"""Geometric constraints and constrained acquisition optimization."""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.exceptions.errors import (
    BotorchError,
    CandidateGenerationError,
    OptimizationGradientError,
)
from botorch.generation.gen import gen_candidates_scipy
from botorch.optim import optimize_acqf_mixed
from botorch.optim.utils.acquisition_utils import fix_features

from .physical import geometry_signature
from .runtime import TENSOR_KWARGS
from .sampling import (
    canonicalise_shape_blocks,
    feasible_mask,
    fixed_features_specific,
)
from .shapes import ellipse, separating_axis_theorem, smooth_min, triangle


def gen_candidates_scipy_single_reconstruction(
    initial_conditions,
    acquisition_function,
    lower_bounds=None,
    upper_bounds=None,
    inequality_constraints=None,
    equality_constraints=None,
    nonlinear_inequality_constraints=None,
    options=None,
    fixed_features=None,
    timeout_sec=None,
    **kwargs,
):
    """Optimize free dimensions and reconstruct fixed features exactly once."""
    if not fixed_features:
        return gen_candidates_scipy(
            initial_conditions=initial_conditions,
            acquisition_function=acquisition_function,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            options=options,
            fixed_features=fixed_features,
            timeout_sec=timeout_sec,
        )
    if inequality_constraints or equality_constraints:
        raise NotImplementedError("Additional linear constraints are not supported.")

    dimension = initial_conditions.shape[-1]
    fixed_indices = sorted(fixed_features)
    free_indices = [index for index in range(dimension) if index not in fixed_features]
    reduced_acquisition = FixedFeatureAcquisitionFunction(
        acq_function=acquisition_function,
        d=dimension,
        columns=fixed_indices,
        values=[fixed_features[index] for index in fixed_indices],
    )
    initial_full = initial_conditions.reshape(-1, dimension)[0]
    reduced_constraints = []
    for constraint, intra_point in nonlinear_inequality_constraints or []:
        initial_slack = float(constraint(initial_full).detach().item())
        feasibility_buffer = max(0.0, min(1e-8, 0.5 * initial_slack))

        def reduced_constraint(
            X, constraint=constraint, feasibility_buffer=feasibility_buffer
        ):
            full = fix_features(
                X, fixed_features=fixed_features, replace_current_value=False
            )
            return constraint(full) - feasibility_buffer

        reduced_constraints.append((reduced_constraint, intra_point))

    scipy_options = dict(options or {})
    scipy_options.pop("batch_limit", None)
    reduced_candidates, values = gen_candidates_scipy(
        initial_conditions=initial_conditions[..., free_indices],
        acquisition_function=reduced_acquisition,
        lower_bounds=(lower_bounds[free_indices] if lower_bounds is not None else None),
        upper_bounds=(upper_bounds[free_indices] if upper_bounds is not None else None),
        nonlinear_inequality_constraints=reduced_constraints,
        options=scipy_options,
        timeout_sec=timeout_sec,
    )
    return (
        fix_features(
            reduced_candidates,
            fixed_features=fixed_features,
            replace_current_value=False,
        ),
        values,
    )


class Bounds:
    """Box and nonlinear feasibility constraints for encoded hole geometries."""

    def __init__(
        self,
        xsize,
        ysize,
        nholes,
        bx,
        by,
        solid_max,
        smin,
        amin,
        raster_tolerance,
        quality_min=None,
        tri_q_min=None,
    ):
        if quality_min is None:
            quality_min = tri_q_min
        if quality_min is None:
            raise ValueError("quality_min (or tri_q_min) is required.")
        self.xsize, self.ysize, self.nholes = xsize, ysize, nholes
        self.bx, self.by = bx, by
        self.solid_max, self.smin, self.amin = solid_max, smin, amin
        self.quality_min = float(quality_min)
        self.raster_tolerance = raster_tolerance
        self.box = (bx, xsize - bx, by, ysize - by)
        self.box_bounds = torch.stack(
            (
                torch.zeros(7 * nholes, **TENSOR_KWARGS),
                torch.ones(7 * nholes, **TENSOR_KWARGS),
            )
        )
        self.nonlinear_constraints = []

        def solid_fraction_constraint(X):
            total_void_area = X.new_zeros(())
            for hole in range(self.nholes):
                offset = 7 * hole
                decoded_triangle = triangle(X, offset, self.xsize, self.ysize)
                decoded_ellipse = ellipse(X, offset, self.xsize, self.ysize)
                triangle_area = 0.5 * (
                    decoded_triangle[2] * decoded_triangle[5]
                    - decoded_triangle[3] * decoded_triangle[4]
                )
                ellipse_area = (
                    torch.pi * decoded_ellipse[2] * decoded_ellipse[3]
                )
                total_void_area += torch.where(
                    X[offset] < 0.5, triangle_area, ellipse_area
                )
            solid_fraction = 1.0 - total_void_area / (self.xsize * self.ysize)
            return self.solid_max - solid_fraction

        self.nonlinear_constraints.append((solid_fraction_constraint, True))

        for hole in range(nholes - 1):

            def centroid_x_constraint(X, hole=hole):
                first, second = 7 * hole, 7 * (hole + 1)
                triangle_a = triangle(X, first, self.xsize, self.ysize)
                triangle_b = triangle(X, second, self.xsize, self.ysize)
                ellipse_a = ellipse(X, first, self.xsize, self.ysize)
                ellipse_b = ellipse(X, second, self.xsize, self.ysize)
                centre_a = torch.where(
                    X[first] < 0.5,
                    triangle_a[0] + (triangle_a[2] + triangle_a[4]) / 3.0,
                    ellipse_a[0],
                )
                centre_b = torch.where(
                    X[second] < 0.5,
                    triangle_b[0] + (triangle_b[2] + triangle_b[4]) / 3.0,
                    ellipse_b[0],
                )
                same_shape = (X[first] < 0.5) == (X[second] < 0.5)
                return torch.where(
                    same_shape,
                    (centre_b - centre_a) / self.xsize,
                    torch.ones_like(centre_a),
                )

            self.nonlinear_constraints.append((centroid_x_constraint, True))

        for hole in range(nholes):

            def minimum_area_constraint(X, hole=hole):
                offset = 7 * hole
                decoded_triangle = triangle(X, offset, self.xsize, self.ysize)
                decoded_ellipse = ellipse(X, offset, self.xsize, self.ysize)
                triangle_area = 0.5 * (
                    decoded_triangle[2] * decoded_triangle[5]
                    - decoded_triangle[3] * decoded_triangle[4]
                )
                ellipse_area = (
                    torch.pi * decoded_ellipse[2] * decoded_ellipse[3]
                )
                area = torch.where(
                    X[offset] < 0.5, triangle_area, ellipse_area
                )
                return (area - self.amin) / (self.xsize * self.ysize)

            self.nonlinear_constraints.append((minimum_area_constraint, True))

            def shape_quality_constraint(X, hole=hole, eps=1e-12):
                offset = 7 * hole
                decoded_triangle = triangle(X, offset, self.xsize, self.ysize)
                first = decoded_triangle[2:4]
                second = decoded_triangle[4:6]
                third = second - first
                signed_twice_area = first[0] * second[1] - first[1] * second[0]
                squared_edge_sum = (
                    torch.dot(first, first)
                    + torch.dot(second, second)
                    + torch.dot(third, third)
                    + eps
                )
                triangle_quality = (
                    2 * np.sqrt(3) * signed_twice_area / squared_edge_sum
                )
                decoded_ellipse = ellipse(X, offset, self.xsize, self.ysize)
                a, b = decoded_ellipse[2], decoded_ellipse[3]
                length = max(self.xsize, self.ysize)
                ellipse_quality = smooth_min(
                    torch.stack(
                        (
                            (a - self.quality_min * b) / length,
                            (b - self.quality_min * a) / length,
                        )
                    )
                )
                return torch.where(
                    X[offset] < 0.5,
                    triangle_quality - self.quality_min,
                    ellipse_quality,
                )

            self.nonlinear_constraints.append((shape_quality_constraint, True))

            def triangle_anchor_constraint(X, hole=hole):
                offset = 7 * hole
                decoded = triangle(X, offset, self.xsize, self.ysize)
                anchor_margin = smooth_min(
                    torch.stack((decoded[2], decoded[4])) / self.xsize
                )
                return torch.where(
                    X[offset] < 0.5,
                    anchor_margin,
                    torch.ones_like(anchor_margin),
                )

            self.nonlinear_constraints.append((triangle_anchor_constraint, True))

            def boundary_constraint(X, hole=hole):
                offset = 7 * hole
                decoded_triangle = triangle(X, offset, self.xsize, self.ysize)
                p0 = decoded_triangle[0:2]
                vector_1 = decoded_triangle[2:4]
                vector_2 = decoded_triangle[4:6]
                length = max(self.xsize, self.ysize)
                vertices = torch.stack((p0, p0 + vector_1, p0 + vector_2))
                triangle_margin = smooth_min(
                    torch.cat(
                        (
                            vertices[:, 0] - self.box[0],
                            self.box[1] - vertices[:, 0],
                            vertices[:, 1] - self.box[2],
                            self.box[3] - vertices[:, 1],
                        )
                    )
                    / length
                )
                cx, cy, rx, ry, theta = ellipse(
                    X, offset, self.xsize, self.ysize
                )
                dx = torch.sqrt(
                    (rx * torch.cos(theta)) ** 2
                    + (ry * torch.sin(theta)) ** 2
                    + 1e-24
                )
                dy = torch.sqrt(
                    (rx * torch.sin(theta)) ** 2
                    + (ry * torch.cos(theta)) ** 2
                    + 1e-24
                )
                ellipse_margin = smooth_min(
                    torch.stack(
                        (
                            cx - dx - self.box[0],
                            self.box[1] - cx - dx,
                            cy - dy - self.box[2],
                            self.box[3] - cy - dy,
                        )
                    )
                    / length
                )
                return torch.where(
                    X[offset] < 0.5, triangle_margin, ellipse_margin
                )

            self.nonlinear_constraints.append((boundary_constraint, True))

        for first in range(nholes):
            for second in range(first + 1, nholes):

                def spacing_constraint(X, first=first, second=second):
                    separation = separating_axis_theorem(
                        X[7 * first : 7 * (first + 1)],
                        X[7 * second : 7 * (second + 1)],
                        self.xsize,
                        self.ysize,
                    )
                    return (separation - self.smin) / max(
                        self.xsize, self.ysize
                    )

                self.nonlinear_constraints.append((spacing_constraint, True))


class Acquisition:
    """Optimize LogEI from labelled starts and retain compact restart metadata."""

    def __init__(
        self,
        acq_function,
        constraints,
        initial_conditions,
        *,
        maxiter: int = 250,
        timeout_sec: float = 60.0,
    ):
        self.acq_function = acq_function
        self.successful_results = []
        self.failed_restarts = []
        entries = self._entries(initial_conditions)
        for restart_index, entry in enumerate(entries):
            initial_point = canonicalise_shape_blocks(
                entry["point"], constraints.nholes
            )
            failure = self._initial_failure(initial_point, constraints)
            if failure:
                self.failed_restarts.append(
                    {
                        "restart_index": restart_index,
                        "source": entry["source"],
                        "reason": failure,
                    }
                )
                continue
            started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    candidate, value = optimize_acqf_mixed(
                        acq_function=self.acq_function,
                        bounds=constraints.box_bounds,
                        q=1,
                        num_restarts=1,
                        raw_samples=None,
                        fixed_features_list=fixed_features_specific(
                            initial_point, constraints.nholes
                        ),
                        nonlinear_inequality_constraints=(
                            constraints.nonlinear_constraints
                        ),
                        batch_initial_conditions=initial_point.reshape(1, 1, -1),
                        options={
                            "batch_limit": 1,
                            "maxiter": maxiter,
                            "ftol": 1e-9,
                        },
                        timeout_sec=timeout_sec,
                        retry_on_optimization_warning=False,
                        gen_candidates=gen_candidates_scipy_single_reconstruction,
                    )
                candidate_row = candidate.detach().reshape(1, -1)
                in_bounds = bool(
                    (
                        (candidate_row >= constraints.box_bounds[0])
                        & (candidate_row <= constraints.box_bounds[1])
                    ).all()
                )
                if not in_bounds or not feasible_mask(candidate_row, constraints).all():
                    raise CandidateGenerationError("infeasible optimizer result")
                scale = (
                    constraints.box_bounds[1] - constraints.box_bounds[0]
                ).clamp_min(1e-12)
                result = {
                    "restart_index": restart_index,
                    "source": entry["source"],
                    "shape_tuple": tuple(
                        int(initial_point[7 * hole].item())
                        for hole in range(constraints.nholes)
                    ),
                    "start_point": initial_point.detach().clone(),
                    "start_distance": float(
                        torch.linalg.vector_norm(
                            (candidate_row.reshape(-1) - initial_point) / scale
                        ).item()
                    ),
                    "candidate": candidate_row,
                    "acquisition_value": value.reshape(-1)[0].detach(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "warning_count": len(caught),
                }
                for key in (
                    "anchor_index",
                    "archive_rank",
                    "start_method",
                    "raw_acquisition_value",
                ):
                    if key in entry:
                        result[key] = entry[key]
                self.successful_results.append(result)
            except (
                BotorchError,
                ValueError,
                RuntimeError,
                CandidateGenerationError,
                OptimizationGradientError,
            ) as error:
                self.failed_restarts.append(
                    {
                        "restart_index": restart_index,
                        "source": entry["source"],
                        "reason": str(error),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )

        self.successful_results.sort(
            key=lambda result: float(result["acquisition_value"].item()),
            reverse=True,
        )
        unique, signatures = [], set()
        for result in self.successful_results:
            signature = geometry_signature(result["candidate"].squeeze(0), constraints)
            if signature not in signatures:
                result["signature"] = signature
                unique.append(result)
                signatures.add(signature)
        self.successful_results = unique

    @staticmethod
    def _entries(initial_conditions):
        if isinstance(initial_conditions, torch.Tensor):
            return [
                {"point": point, "source": "unspecified"}
                for point in initial_conditions
            ]
        entries = []
        for entry in initial_conditions:
            copied = dict(entry)
            copied["source"] = entry.get("source", "unspecified")
            entries.append(copied)
        return entries

    @staticmethod
    def _initial_failure(point, constraints):
        if not torch.isfinite(point).all():
            return "non-finite initial condition"
        if bool(
            ((point < constraints.box_bounds[0]) | (point > constraints.box_bounds[1])).any()
        ):
            return "initial condition outside box bounds"
        if not feasible_mask(point.reshape(1, -1), constraints)[0]:
            return "infeasible initial condition"
        return None
