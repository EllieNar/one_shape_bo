"""Feasible global and local sampling for shaped-hole designs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math

import torch

from .ellipse_features import RAW_ELLIPSE_FEATURES, design_features
from .physical import geometry_signature, rasterized_volume_feasibility
from .runtime import DEVICE, TENSOR_KWARGS
from .shapes import (
    anchor_point,
    canonicalize_ellipses,
    canonicalize_triangles,
    separating_axis_theorem_batch,
)


AREA_NEAR_EQUAL = "near_equal"
AREA_BROAD = "broad"


def _shape_tuple(nholes: int, shape_tuple: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted(int(value) for value in shape_tuple))
    if len(values) != nholes or any(value not in (0, 1) for value in values):
        raise ValueError(f"Shape tuple must contain {nholes} triangle/ellipse indices.")
    return values


def fixed_features(
    nholes: int,
    htype: int | None = None,
    shape_tuples: Iterable[Sequence[int]] | None = None,
):
    """Return fixed-feature maps for allowed canonical shape tuples.

    ``htype`` preserves the current all-one-shape interface.  ``shape_tuples``
    keeps restart selection usable if mixed configurations are enabled later.
    """
    if shape_tuples is None:
        if htype not in (0, 1):
            raise ValueError("htype must be 0 (triangles) or 1 (ellipses).")
        tuples = [(int(htype),) * nholes]
    else:
        tuples = list(dict.fromkeys(_shape_tuple(nholes, item) for item in shape_tuples))

    results = []
    for shapes in tuples:
        features = {}
        for hole, shape in enumerate(shapes):
            offset = 7 * hole
            features[offset] = float(shape)
            if shape == 1:
                features[offset + 6] = 0.0
        results.append(features)
    return results


def fixed_features_specific(point, nholes: int):
    design = canonicalise_shape_blocks(point, nholes)
    features = {}
    for offset in range(0, 7 * nholes, 7):
        shape = int(design[offset].item())
        features[offset] = float(shape)
        if shape == 1:
            features[offset + 6] = 0.0
    return [features]


def canonicalise_shape_blocks(point, nholes: int):
    blocks = point.detach().reshape(nholes, 7)
    order = torch.argsort(blocks[:, 0], stable=True)
    return blocks[order].reshape(-1).clone()


def canonicalise_design(point, nholes: int):
    blocks = canonicalise_shape_blocks(point, nholes).reshape(nholes, 7)
    ntri = int((blocks[:, 0] < 0.5).sum().item())
    triangle_blocks = blocks[:ntri]
    if ntri:
        p0 = triangle_blocks[:, 1:3]
        edge_1 = 2.0 * triangle_blocks[:, 3:5] - 1.0
        edge_2 = 2.0 * triangle_blocks[:, 5:7] - 1.0
        vertices = torch.stack((p0, p0 + edge_1, p0 + edge_2), dim=1)
    else:
        vertices = blocks.new_empty((0, 3, 2))
    triangles = canonicalize_triangles(ntri, vertices)
    ellipses = canonicalize_ellipses(nholes - ntri, blocks[ntri:, 1:6])
    return torch.cat((triangles, ellipses))


def feasible_mask(X, constraints):
    feasible = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for constraint_function, intra_point in constraints.nonlinear_constraints:
        if not intra_point:
            raise NotImplementedError("Only intra-point constraints are supported.")
        values = torch.stack([constraint_function(point) for point in X]).to(X.device)
        feasible &= values >= 0
    return feasible


def _rand(shape=(), *, generator=None):
    return torch.rand(shape, generator=generator, **TENSOR_KWARGS)


def random_shape_area(
    area,
    constraints,
    shape_type: int,
    shape_trials: int = 100,
    normalise: bool = False,
    generator=None,
    eps: float = 1e-12,
):
    xmin, xmax, ymin, ymax = constraints.box
    domain_extent = torch.tensor([xmax - xmin, ymax - ymin], **TENSOR_KWARGS)
    shape_index = torch.tensor([shape_type], **TENSOR_KWARGS)

    if shape_type == 0:
        for _ in range(shape_trials):
            raw_vertices = 2.0 * _rand((3, 2), generator=generator) - 1.0
            edge_1 = raw_vertices[1] - raw_vertices[0]
            edge_2 = raw_vertices[2] - raw_vertices[0]
            edge_3 = raw_vertices[2] - raw_vertices[1]
            raw_area = 0.5 * torch.abs(
                edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
            )
            edge_sum = (
                torch.dot(edge_1, edge_1)
                + torch.dot(edge_2, edge_2)
                + torch.dot(edge_3, edge_3)
            ).clamp_min(eps)
            quality = 4.0 * (3.0**0.5) * raw_area / edge_sum
            if raw_area <= eps or quality < constraints.quality_min:
                continue
            vertices = (raw_vertices - raw_vertices[0]) * torch.sqrt(
                area / raw_area.clamp_min(eps)
            )
            extent = vertices.max(dim=0).values - vertices.min(dim=0).values
            if bool((extent <= domain_extent).all()):
                if normalise:
                    vertices = vertices / vertices.new_tensor(
                        [constraints.xsize, constraints.ysize]
                    )
                return torch.cat((shape_index, vertices.reshape(-1)))
        return None

    for _ in range(shape_trials):
        radii = 0.5 * _rand((2,), generator=generator)
        angle = torch.pi * _rand(generator=generator)
        raw_area = torch.pi * radii[0] * radii[1]
        quality = torch.minimum(radii[0], radii[1]) / torch.maximum(
            radii[0], radii[1]
        ).clamp_min(eps)
        if raw_area <= eps or quality < constraints.quality_min:
            continue
        radii = radii * torch.sqrt(area / raw_area.clamp_min(eps))
        dx = torch.sqrt(
            (radii[0] * torch.cos(angle)) ** 2
            + (radii[1] * torch.sin(angle)) ** 2
        )
        dy = torch.sqrt(
            (radii[0] * torch.sin(angle)) ** 2
            + (radii[1] * torch.cos(angle)) ** 2
        )
        if dx > (xmax - xmin) / 2 or dy > (ymax - ymin) / 2:
            continue
        cx = xmin + dx + _rand(generator=generator) * (xmax - xmin - 2 * dx)
        cy = ymin + dy + _rand(generator=generator) * (ymax - ymin - 2 * dy)
        if normalise:
            cx, cy = cx / constraints.xsize, cy / constraints.ysize
            radii = radii / radii.new_tensor([constraints.xsize, constraints.ysize])
            angle = angle / torch.pi
        return torch.cat(
            (
                shape_index,
                cx.unsqueeze(0),
                cy.unsqueeze(0),
                radii,
                angle.unsqueeze(0),
                torch.zeros(1, **TENSOR_KWARGS),
            )
        )
    return None


def _area_weights(nholes: int, strategy: str, generator=None):
    if strategy == AREA_NEAR_EQUAL:
        return 0.9 + 0.2 * _rand((nholes,), generator=generator)
    if strategy == AREA_BROAD:
        return -torch.log(_rand((nholes,), generator=generator).clamp_min(1e-12))
    raise ValueError(f"Unknown area strategy: {strategy!r}.")


def burn_in_feasibility_limits(constraints, shape_types):
    """Return cheap necessary packing checks for strict burn-in sampling.

    These checks can prove that a request is impossible, but passing them does
    not guarantee that a particular set of shapes can be packed.
    """
    shape_types = tuple(int(value) for value in shape_types)
    if len(shape_types) != constraints.nholes:
        raise ValueError("shape_types must contain one entry per hole.")
    xmin, xmax, ymin, ymax = constraints.box
    width, height = xmax - xmin, ymax - ymin
    interior_area = width * height
    domain_area = constraints.xsize * constraints.ysize
    minimum_void = (1.0 - constraints.solid_max) * domain_area
    required_void = max(minimum_void, constraints.nholes * constraints.amin)
    failures = []
    if width <= 0 or height <= 0:
        failures.append("the usable rectangle has non-positive extent")
    if required_void > interior_area + 1e-12:
        failures.append(
            "required void area exceeds the usable rectangular area"
        )

    triangle_area_limit = 0.5 * interior_area
    ellipse_area_limit = 0.25 * math.pi * interior_area
    if 0 in shape_types and constraints.amin > triangle_area_limit + 1e-12:
        failures.append("amin exceeds the maximum area of an inscribed triangle")
    if 1 in shape_types and constraints.amin > ellipse_area_limit + 1e-12:
        failures.append("amin exceeds the maximum area of an inscribed ellipse")

    clearance_left = None
    clearance_right = None
    if constraints.nholes > 1 and width > 0 and height > 0:
        # Pairwise clearance s implies disjoint s/2 parallel bodies.  Steiner's
        # formula and the isoperimetric inequality give this necessary bound.
        remaining_area = required_void - (constraints.nholes - 1) * constraints.amin
        root_area_sum_lower = math.sqrt(max(remaining_area, 0.0)) + (
            constraints.nholes - 1
        ) * math.sqrt(constraints.amin)
        clearance = constraints.smin
        clearance_left = (
            required_void
            + clearance * math.sqrt(math.pi) * root_area_sum_lower
            + constraints.nholes * math.pi * clearance**2 / 4.0
        )
        clearance_right = (
            interior_area
            + clearance * (width + height)
            + math.pi * clearance**2 / 4.0
        )
        if clearance_left > clearance_right + 1e-12:
            failures.append(
                "the clearance-expanded-hole area exceeds the expanded usable area"
            )

    return {
        "usable_width": float(width),
        "usable_height": float(height),
        "usable_area": float(interior_area),
        "minimum_void_area_from_solid_max": float(minimum_void),
        "minimum_void_area_from_amin": float(
            constraints.nholes * constraints.amin
        ),
        "required_void_area": float(required_void),
        "smin": float(constraints.smin),
        "clearance_area_lower_bound": clearance_left,
        "clearance_available_area_upper_bound": clearance_right,
        "necessary_conditions_passed": not failures,
        "failures": failures,
    }


def _burn_in_target_areas(constraints, strategy, generator=None):
    """Draw one feasible total-area allocation for the burn-in-only sampler."""
    domain_area = constraints.xsize * constraints.ysize
    xmin, xmax, ymin, ymax = constraints.box
    interior_area = (xmax - xmin) * (ymax - ymin)
    minimum_void = (1.0 - constraints.solid_max) * domain_area
    minimum_total = constraints.nholes * constraints.amin
    lower = max(minimum_void, minimum_total)
    preferred_upper = min(interior_area, minimum_void + 0.02 * domain_area)
    upper = max(lower, preferred_upper)
    if lower > interior_area + 1e-12:
        return None
    if upper > interior_area:
        upper = interior_area
    interior_sample = 0.05 + 0.90 * _rand(generator=generator)
    target_void = lower + interior_sample * (upper - lower)
    weights = _area_weights(constraints.nholes, strategy, generator)
    return constraints.amin + (
        target_void - minimum_total
    ) * weights / weights.sum()


def _burn_in_options(points, anchors, shape_type):
    """Translate one normalized shape template to candidate anchor locations."""
    count = anchors.shape[0]
    if shape_type == 0:
        vertices = points[1:].reshape(3, 2).unsqueeze(0) + anchors.unsqueeze(1)
        p0 = vertices[:, 0, :]
        edge_1 = 0.5 * (vertices[:, 1, :] - p0 + 1.0)
        edge_2 = 0.5 * (vertices[:, 2, :] - p0 + 1.0)
        return torch.cat(
            (points[0].expand(count, 1), p0, edge_1, edge_2), dim=1
        )
    options = points.unsqueeze(0).expand(count, -1).clone()
    options[:, 1:3] += anchors
    return options


def _line_intersection(first_normal, first_constant, second_normal, second_constant):
    matrix = torch.stack((first_normal, second_normal))
    determinant = torch.linalg.det(matrix)
    if torch.abs(determinant) <= 1e-12:
        return None
    constants = torch.stack((first_constant, second_constant))
    return torch.linalg.solve(matrix, constants)


def _inset_partition_triangle(vertices, clearance):
    """Inset the two internal edges of a CCW rectangular partition cell."""
    edges = vertices.roll(-1, dims=0) - vertices
    normals = torch.stack((-edges[:, 1], edges[:, 0]), dim=1)
    normals = normals / torch.linalg.vector_norm(
        normals, dim=1, keepdim=True
    ).clamp_min(1e-12)
    constants = (normals * vertices).sum(dim=1)
    constants = constants.clone()
    constants[1:] += clearance
    points = (
        _line_intersection(normals[2], constants[2], normals[0], constants[0]),
        _line_intersection(normals[0], constants[0], normals[1], constants[1]),
        _line_intersection(normals[1], constants[1], normals[2], constants[2]),
    )
    if any(point is None for point in points):
        return None
    result = torch.stack(points)
    edge_1, edge_2 = result[1] - result[0], result[2] - result[0]
    if edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0] <= 0:
        return None
    return result


def _structured_triangle_burn_in_candidate(areas, constraints, generator=None):
    """Pack three triangles in separated cells of a four-way partition."""
    if constraints.nholes != 3:
        return None
    xmin, xmax, ymin, ymax = constraints.box
    boundary_guard = 0.05
    xmin, xmax = xmin + boundary_guard, xmax - boundary_guard
    ymin, ymax = ymin + boundary_guard, ymax - boundary_guard
    width, height = xmax - xmin, ymax - ymin
    jitter = 0.08 * (2.0 * _rand((2,), generator=generator) - 1.0)
    centre = torch.tensor(
        [
            xmin + width * (0.5 + float(jitter[0].item())),
            ymin + height * (0.5 + float(jitter[1].item())),
        ],
        **TENSOR_KWARGS,
    )
    corners = [
        torch.tensor(value, **TENSOR_KWARGS)
        for value in (
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        )
    ]
    cells = (
        torch.stack((corners[0], corners[1], centre)),
        torch.stack((corners[1], corners[2], centre)),
        torch.stack((corners[2], corners[3], centre)),
        torch.stack((corners[3], corners[0], centre)),
    )
    clearance = 0.5 * (constraints.smin + 0.01)
    inset = [
        _inset_partition_triangle(vertices, clearance) for vertices in cells
    ]
    valid_cells = [vertices for vertices in inset if vertices is not None]
    if len(valid_cells) < constraints.nholes:
        return None
    capacities = []
    for vertices in valid_cells:
        first, second = vertices[1] - vertices[0], vertices[2] - vertices[0]
        capacities.append(
            0.5 * torch.abs(first[0] * second[1] - first[1] * second[0])
        )
    cell_order = torch.randperm(
        len(valid_cells), generator=generator, device=DEVICE
    ).tolist()
    selected = cell_order[: constraints.nholes]
    selected.sort(key=lambda index: float(capacities[index].item()), reverse=True)
    area_order = torch.argsort(areas, descending=True).tolist()
    output = [None] * constraints.nholes
    for hole_index, cell_index in zip(area_order, selected):
        capacity = capacities[cell_index]
        if areas[hole_index] > capacity:
            return None
        vertices = valid_cells[cell_index]
        centroid = vertices.mean(dim=0)
        scale = torch.sqrt(areas[hole_index] / capacity)
        output[hole_index] = centroid + scale * (vertices - centroid)
    normalized = torch.stack(output) / torch.tensor(
        [constraints.xsize, constraints.ysize], **TENSOR_KWARGS
    )
    return canonicalize_triangles(constraints.nholes, normalized)


def _structured_ellipse_burn_in_candidate(areas, constraints, generator=None):
    """Pack axis-aligned ellipses into separated randomized vertical cells."""
    xmin, xmax, ymin, ymax = constraints.box
    boundary_guard = 0.05
    xmin, xmax = xmin + boundary_guard, xmax - boundary_guard
    ymin, ymax = ymin + boundary_guard, ymax - boundary_guard
    width, height = xmax - xmin, ymax - ymin
    clearance = constraints.smin + 0.01
    free_width = width - (constraints.nholes - 1) * clearance
    if free_width <= 0:
        return None
    minimum_widths = 4.0 * areas / (math.pi * height)
    spare = free_width - float(minimum_widths.sum().item())
    if spare < 0:
        return None
    slack_weights = _rand((constraints.nholes,), generator=generator)
    cell_widths = minimum_widths + spare * slack_weights / slack_weights.sum()
    hole_order = torch.randperm(
        constraints.nholes, generator=generator, device=DEVICE
    ).tolist()
    cursor = xmin
    ellipses = [None] * constraints.nholes
    for hole_index in hole_order:
        cell_width = cell_widths[hole_index]
        radius_x = 0.5 * cell_width
        radius_y = areas[hole_index] / (math.pi * radius_x)
        if radius_y > 0.5 * height:
            return None
        quality = torch.minimum(radius_x, radius_y) / torch.maximum(
            radius_x, radius_y
        )
        if quality < constraints.quality_min:
            return None
        centre_x = cursor + radius_x
        vertical_slack = height - 2.0 * radius_y
        centre_y = ymin + radius_y + _rand(generator=generator) * vertical_slack
        ellipses[hole_index] = torch.stack(
            (
                centre_x / constraints.xsize,
                centre_y / constraints.ysize,
                radius_x / constraints.xsize,
                radius_y / constraints.ysize,
                areas.new_zeros(()),
            )
        )
        cursor += cell_width + clearance
    return canonicalize_ellipses(constraints.nholes, torch.stack(ellipses))


def _structured_burn_in_candidate(areas, constraints, shape_types, generator=None):
    if not shape_types:
        return None
    if all(shape == 0 for shape in shape_types):
        return _structured_triangle_burn_in_candidate(
            areas, constraints, generator=generator
        )
    if all(shape == 1 for shape in shape_types):
        return _structured_ellipse_burn_in_candidate(
            areas, constraints, generator=generator
        )
    return None


def burn_in_shape_candidate(
    feature_map,
    constraints,
    amin,
    *,
    area_strategy=AREA_BROAD,
    placement_restarts: int = 4,
    template_retries: int = 4,
    anchor_batches: int = 4,
    anchors_per_batch: int = 256,
    shape_trials: int = 200,
    dir_trials: int = 200,
    generator=None,
):
    """Construct one burn-in candidate with bounded placement backtracking.

    This function is deliberately separate from ``random_shape_candidate`` so
    BO/global-pool sampling retains its established behavior.
    """
    if float(amin) != float(constraints.amin):
        raise ValueError("amin must match constraints.amin.")
    shape_types = [
        int(feature_map[7 * hole]) for hole in range(constraints.nholes)
    ]
    areas = _burn_in_target_areas(
        constraints, area_strategy, generator=generator
    )
    if areas is None:
        return None, "area"
    placement_order = torch.argsort(areas, descending=True)

    generated_any_template = False
    for _ in range(int(placement_restarts)):
        placed = {}
        placement_failed = False
        for index_tensor in placement_order:
            index = int(index_tensor.item())
            shape_type = shape_types[index]
            selected = None
            for _ in range(int(template_retries)):
                points = random_shape_area(
                    areas[index],
                    constraints,
                    shape_type,
                    shape_trials,
                    normalise=True,
                    generator=generator,
                )
                if points is None:
                    continue
                generated_any_template = True
                anchor_min = anchor_point(points, False, constraints, True)
                anchor_max = anchor_point(points, True, constraints, True)
                if bool((anchor_min > anchor_max).any()):
                    continue
                for _ in range(int(anchor_batches)):
                    anchors = anchor_min + (anchor_max - anchor_min) * _rand(
                        (anchors_per_batch, 2), generator=generator
                    )
                    options = _burn_in_options(points, anchors, shape_type)
                    valid = torch.ones(
                        anchors_per_batch, dtype=torch.bool, device=DEVICE
                    )
                    for other_points in placed.values():
                        valid &= separating_axis_theorem_batch(
                            options,
                            other_points,
                            constraints.xsize,
                            constraints.ysize,
                            dir_trials,
                        ) >= constraints.smin
                    choices = torch.nonzero(valid, as_tuple=False).flatten()
                    if choices.numel():
                        position = torch.randint(
                            choices.numel(),
                            (),
                            generator=generator,
                            device=DEVICE,
                        )
                        selected = options[choices[position]]
                        break
                if selected is not None:
                    break
            if selected is None:
                placement_failed = True
                break
            placed[index] = selected
        if not placement_failed:
            candidate = torch.stack(
                [placed[index] for index in range(constraints.nholes)]
            ).reshape(-1)
            return canonicalise_design(candidate, constraints.nholes), None
    structured = _structured_burn_in_candidate(
        areas, constraints, shape_types, generator=generator
    )
    if structured is not None:
        return structured, None
    return None, "placement" if generated_any_template else "shape"


def sample_burn_in_feasible(
    number: int,
    nholes: int,
    htype: int,
    constraints,
    amin,
    *,
    area_strategies: Sequence[str] = (AREA_NEAR_EQUAL, AREA_BROAD),
    max_candidate_attempts_per_stratum: int = 5000,
    generator=None,
    return_diagnostics: bool = False,
):
    """Collect the complete initial GP design using the burn-in-only sampler."""
    feature_maps = fixed_features(nholes, htype)
    if len(feature_maps) != 1:
        raise ValueError("Burn-in sampling requires one fixed shape configuration.")
    feature_map = feature_maps[0]
    shape_types = [int(feature_map[7 * hole]) for hole in range(nholes)]
    limits = burn_in_feasibility_limits(constraints, shape_types)
    if not limits["necessary_conditions_passed"]:
        raise ValueError(
            "Burn-in geometry fails necessary feasibility checks: "
            + "; ".join(limits["failures"])
        )

    quotient, remainder = divmod(number, len(area_strategies))
    batches = []
    stratum_records = []
    for stratum_index, strategy in enumerate(area_strategies):
        quota = quotient + int(stratum_index < remainder)
        accepted = []
        failures = {
            "area": 0,
            "shape": 0,
            "placement": 0,
            "encoding": 0,
            "analytical": 0,
            "raster": 0,
        }
        attempts = 0
        while len(accepted) < quota and attempts < max_candidate_attempts_per_stratum:
            attempts += 1
            candidate, failure = burn_in_shape_candidate(
                feature_map,
                constraints,
                amin,
                area_strategy=strategy,
                generator=generator,
            )
            if candidate is None:
                failures[failure] += 1
                continue
            if not torch.isfinite(candidate).all() or bool(
                ((candidate < 0.0) | (candidate > 1.0)).any()
            ):
                failures["encoding"] += 1
                continue
            if not feasible_mask(candidate.unsqueeze(0), constraints)[0]:
                failures["analytical"] += 1
                continue
            raster_valid, _ = rasterized_volume_feasibility(
                candidate.unsqueeze(0), constraints
            )
            if not raster_valid[0]:
                failures["raster"] += 1
                continue
            accepted.append(candidate)

        record = {
            "area_strategy": strategy,
            "requested": quota,
            "accepted": len(accepted),
            "candidate_attempts": attempts,
            "rejections": failures,
        }
        stratum_records.append(record)
        if len(accepted) != quota:
            raise RuntimeError(
                "Could not complete the burn-in stratum "
                f"{strategy!r}: accepted {len(accepted)}/{quota} after "
                f"{attempts} attempts; rejections={failures}."
            )
        if accepted:
            batches.append(torch.stack(accepted))

    designs = (
        torch.cat(batches)
        if batches
        else torch.empty((0, 7 * constraints.nholes), **TENSOR_KWARGS)
    )
    diagnostics = {
        "sampler": "burn_in_clearance_aware",
        "requested": int(number),
        "accepted": int(len(designs)),
        "limits": limits,
        "strata": stratum_records,
    }
    return (designs, diagnostics) if return_diagnostics else designs


def random_shape_candidate(
    feature_map,
    constraints,
    amin,
    anchor_trials: int = 1500,
    shape_trials: int = 200,
    dir_trials: int = 100,
    area_strategy: str = AREA_BROAD,
    generator=None,
):
    """Generate one feasible-by-construction proposal with an explicit area prior."""
    if float(amin) != float(constraints.amin):
        raise ValueError("amin must match constraints.amin.")
    nholes = constraints.nholes
    shape_types = [int(feature_map[7 * hole]) for hole in range(nholes)]
    domain_area = constraints.xsize * constraints.ysize
    minimum_void = (1.0 - constraints.solid_max) * domain_area
    xmin, xmax, ymin, ymax = constraints.box
    interior_area = (xmax - xmin) * (ymax - ymin)
    if minimum_void > interior_area:
        return None
    maximum_void = min(interior_area, minimum_void + 0.02 * domain_area)
    target_void = minimum_void + (0.05 + 0.90 * _rand(generator=generator)) * (
        maximum_void - minimum_void
    )
    minimum_total = nholes * amin
    if target_void < minimum_total:
        return None
    weights = _area_weights(nholes, area_strategy, generator)
    areas = amin + (target_void - minimum_total) * weights / weights.sum()
    placement_order = torch.argsort(areas, descending=True)

    shapes = [None] * nholes
    for index_tensor in placement_order:
        index = int(index_tensor.item())
        shapes[index] = random_shape_area(
            areas[index],
            constraints,
            shape_types[index],
            shape_trials,
            normalise=True,
            generator=generator,
        )
        if shapes[index] is None:
            return None

    placed = {}
    for index_tensor in placement_order:
        index = int(index_tensor.item())
        points = shapes[index]
        anchor_min = anchor_point(points, False, constraints, True)
        anchor_max = anchor_point(points, True, constraints, True)
        if bool((anchor_min > anchor_max).any()):
            return None
        anchors = anchor_min + (anchor_max - anchor_min) * _rand(
            (anchor_trials, 2), generator=generator
        )
        if shape_types[index] == 0:
            vertices = points[1:].reshape(3, 2).unsqueeze(0) + anchors.unsqueeze(1)
            p0 = vertices[:, 0, :]
            edge_1 = 0.5 * (vertices[:, 1, :] - p0 + 1.0)
            edge_2 = 0.5 * (vertices[:, 2, :] - p0 + 1.0)
            options = torch.cat(
                (points[0].expand(anchor_trials, 1), p0, edge_1, edge_2), dim=1
            )
        else:
            options = points.unsqueeze(0).expand(anchor_trials, -1).clone()
            options[:, 1:3] += anchors
        valid = torch.ones(anchor_trials, dtype=torch.bool, device=DEVICE)
        for other_points in placed.values():
            valid &= separating_axis_theorem_batch(
                options,
                other_points,
                constraints.xsize,
                constraints.ysize,
                dir_trials,
            ) >= constraints.smin
        choices = torch.nonzero(valid, as_tuple=False).flatten()
        if not choices.numel():
            return None
        choice_position = torch.randint(
            choices.numel(), (), generator=generator, device=DEVICE
        )
        placed[index] = options[choices[choice_position]]
    return canonicalise_design(
        torch.stack([placed[index] for index in range(nholes)]).reshape(-1), nholes
    )


def sample_random_feasible(
    number: int,
    constraints,
    amin,
    feature_map=None,
    batch_size: int = 200,
    max_batches: int = 250,
    anchor_trials: int = 1000,
    shape_trials: int = 500,
    dir_trials: int = 200,
    allow_partial: bool = False,
    area_strategy: str = AREA_BROAD,
    generator=None,
):
    accepted = []
    if feature_map is None:
        ellipse_count = int(
            torch.randint(
                constraints.nholes + 1, (), generator=generator, device=DEVICE
            ).item()
        )
        shapes = [0] * (constraints.nholes - ellipse_count) + [1] * ellipse_count
        feature_map = {7 * hole: float(shape) for hole, shape in enumerate(shapes)}
    for _ in range(batch_size * max_batches):
        candidate = random_shape_candidate(
            feature_map,
            constraints,
            amin,
            anchor_trials,
            shape_trials,
            dir_trials,
            area_strategy,
            generator,
        )
        if candidate is None or not torch.isfinite(candidate).all():
            continue
        if bool(((candidate < 0.0) | (candidate > 1.0)).any()):
            continue
        if not feasible_mask(candidate.unsqueeze(0), constraints)[0]:
            continue
        raster_valid, _ = rasterized_volume_feasibility(
            candidate.unsqueeze(0), constraints
        )
        if raster_valid[0]:
            accepted.append(candidate)
            if len(accepted) >= number:
                return torch.stack(accepted)
    if allow_partial:
        if accepted:
            return torch.stack(accepted)
        return torch.empty((0, 7 * constraints.nholes), **TENSOR_KWARGS)
    raise RuntimeError(
        "Could not generate enough independently random feasible points. "
        "Increase max_batches or anchor_trials, or relax the constraints."
    )


def sample_stratified_feasible(
    number: int,
    nholes: int,
    htype: int,
    constraints,
    amin,
    area_strategies: Sequence[str] = (AREA_NEAR_EQUAL, AREA_BROAD),
    shape_tuples: Iterable[Sequence[int]] | None = None,
    allow_partial: bool = False,
    generator=None,
):
    """Allocate exact quotas over shape configurations and area policies."""
    strata = [
        (feature_map, strategy)
        for feature_map in fixed_features(nholes, htype, shape_tuples)
        for strategy in area_strategies
    ]
    quotient, remainder = divmod(number, len(strata))
    batches = []
    for index, (feature_map, strategy) in enumerate(strata):
        quota = quotient + int(index < remainder)
        if not quota:
            continue
        result = sample_random_feasible(
            quota,
            constraints,
            amin,
            feature_map=feature_map,
            allow_partial=allow_partial,
            area_strategy=strategy,
            generator=generator,
        )
        if result.numel():
            batches.append(result)
        elif not allow_partial:
            raise RuntimeError("Could not fill every sampling stratum.")
    if not batches:
        raise RuntimeError("Could not generate any feasible points.")
    return torch.cat(batches)


def _model_kernel_distance_matrix(model, points):
    """Return GP-correlation distances between normalized design points."""
    count = len(points)
    if count == 0:
        return points.new_empty((0, 0))
    if count == 1:
        return points.new_zeros((1, 1))
    gp = getattr(model, "gp", model)
    if not hasattr(gp, "covar_module"):
        raise ValueError("The model does not expose a covariance module.")
    input_transform = getattr(gp, "input_transform", None)
    kernel_points = (
        points if input_transform is None else input_transform.transform(points)
    )
    with torch.no_grad():
        covariance = gp.covar_module(kernel_points).to_dense().reshape(count, count)
    diagonal = covariance.diagonal().clamp_min(1e-12)
    denominator = torch.sqrt(diagonal[:, None] * diagonal[None, :])
    correlation = (covariance / denominator).clamp(-1.0, 1.0)
    return torch.sqrt((2.0 - 2.0 * correlation).clamp_min(0.0))


def model_kernel_distance(model, first, second):
    """Return the fitted GP kernel distance between two designs."""
    gp = getattr(model, "gp", model)
    if not hasattr(gp, "covar_module"):
        return float("nan")
    points = torch.stack((first.detach().reshape(-1), second.detach().reshape(-1)))
    return float(_model_kernel_distance_matrix(model, points)[0, 1].item())


def select_archive_local_conditions(
    number: int,
    x_train,
    y_train,
    constraints,
    model,
    elite_multiplier: int = 3,
):
    """Select good, GP-diverse feasible observations as local optimizer starts."""
    if number <= 0 or len(x_train) == 0:
        return (
            torch.empty((0, x_train.shape[-1]), **TENSOR_KWARGS),
            [],
        )

    target = min(int(number), len(x_train))
    elite_limit = min(len(x_train), max(target, elite_multiplier * target))
    ordered_indices = torch.argsort(y_train.reshape(-1))
    elite_indices, signatures = [], set()
    for index_tensor in ordered_indices:
        index = int(index_tensor.item())
        signature = geometry_signature(x_train[index], constraints)
        if signature in signatures:
            continue
        signatures.add(signature)
        elite_indices.append(index)
        if len(elite_indices) >= elite_limit:
            break
    if not elite_indices:
        return (
            torch.empty((0, x_train.shape[-1]), **TENSOR_KWARGS),
            [],
        )

    elite = x_train[elite_indices]
    target = min(target, len(elite))
    distances = _model_kernel_distance_matrix(model, elite)
    shape_tuples = [
        tuple(
            int(elite[position, 7 * hole].item())
            for hole in range(constraints.nholes)
        )
        for position in range(len(elite))
    ]
    selected = [0]
    represented_shapes = {shape_tuples[0]}
    while len(selected) < target:
        available = [index for index in range(len(elite)) if index not in selected]
        unseen_shapes = [
            index
            for index in available
            if shape_tuples[index] not in represented_shapes
        ]
        candidates = unseen_shapes or available
        chosen = max(
            candidates,
            key=lambda index: (
                float(distances[index, selected].min().item()),
                -index,
            ),
        )
        selected.append(chosen)
        represented_shapes.add(shape_tuples[chosen])

    conditions = elite[selected].detach().clone()
    metadata = [
        {
            "allow_observed_start": True,
            "anchor_index": elite_indices[position],
            "archive_rank": position,
            "start_method": "archive",
        }
        for position in selected
    ]
    return conditions, metadata


def sample_local_feasible(
    number: int,
    x_train,
    y_train,
    constraints,
    radius: float = 0.05,
    max_attempts: int = 5000,
    anchor_count: int = 10,
    generator=None,
):
    """Sample the configured-radius neighbourhood of diverse good observations."""
    if number <= 0 or len(x_train) == 0:
        return torch.empty((0, x_train.shape[-1]), **TENSOR_KWARGS)
    top_count = min(max(anchor_count * 3, anchor_count), len(x_train))
    good_indices = torch.topk(
        y_train.squeeze(-1), top_count, largest=False
    ).indices
    # Choose quality-ordered anchors that are not all from one tight cluster.
    scale = (constraints.box_bounds[1] - constraints.box_bounds[0]).clamp_min(1e-12)
    anchors = []
    for index in good_indices:
        candidate = x_train[index]
        if not anchors or min(
            torch.linalg.vector_norm((candidate - prior) / scale) for prior in anchors
        ) >= radius:
            anchors.append(candidate)
        if len(anchors) >= anchor_count:
            break
    if not anchors:
        anchors = [x_train[good_indices[0]]]

    lower, upper = constraints.box_bounds
    accepted, signatures = [], set()
    for attempt in range(max_attempts):
        base = anchors[attempt % len(anchors)]
        output = []
        for hole_index in range(constraints.nholes):
            offset = 7 * hole_index
            shape = int(base[offset].item())
            if shape == 0:
                active = base[offset + 1 : offset + 7]
                noise = torch.randn(
                    active.shape, generator=generator, **TENSOR_KWARGS
                )
                proposal = torch.clamp(
                    active + radius * scale[offset + 1 : offset + 7] * noise,
                    lower[offset + 1 : offset + 7],
                    upper[offset + 1 : offset + 7],
                )
                output.append(torch.cat((base[offset : offset + 1], proposal)))
            else:
                active = base[offset + 1 : offset + 6]
                noise = torch.randn(
                    active.shape, generator=generator, **TENSOR_KWARGS
                )
                proposal = torch.clamp(
                    active + radius * scale[offset + 1 : offset + 6] * noise,
                    lower[offset + 1 : offset + 6],
                    upper[offset + 1 : offset + 6],
                )
                output.append(
                    torch.cat((base[offset : offset + 1], proposal, base.new_zeros(1)))
                )
        candidate = canonicalise_design(torch.cat(output), constraints.nholes)
        if not feasible_mask(candidate.unsqueeze(0), constraints)[0]:
            continue
        raster_valid, _ = rasterized_volume_feasibility(
            candidate.unsqueeze(0), constraints
        )
        if not raster_valid[0]:
            continue
        signature = geometry_signature(candidate, constraints)
        if signature in signatures:
            continue
        signatures.add(signature)
        accepted.append(candidate)
        if len(accepted) >= number:
            break
    if not accepted:
        return torch.empty((0, x_train.shape[-1]), **TENSOR_KWARGS)
    return torch.stack(accepted)


def _unit(values):
    finite = torch.isfinite(values)
    output = torch.zeros_like(values)
    if finite.any():
        subset = values[finite]
        span = subset.max() - subset.min()
        output[finite] = (subset - subset.min()) / span.clamp_min(1e-12)
    return output


def rank_initial_condition_pool(
    acq_function,
    global_conditions,
    local_conditions,
    constraints,
    excluded_signatures,
    attempted_starts,
    minimum_distance: float,
    observed_points=None,
    local_metadata=None,
    acquisition_weight: float = 0.55,
    novelty_weight: float = 0.45,
    number_global: int | None = None,
    global_selection_strategy: str = "greedy_dynamic_novelty",
    ellipse_feature_mode: str = RAW_ELLIPSE_FEATURES,
):
    """Rank valid starts by acquisition and source-appropriate coverage."""
    if global_selection_strategy not in (
        "greedy_dynamic_novelty",
        "static_rank",
    ):
        raise ValueError(
            "global_selection_strategy must be 'greedy_dynamic_novelty' "
            "or 'static_rank'."
        )
    weights = torch.tensor(
        [acquisition_weight, novelty_weight], **TENSOR_KWARGS
    )
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("Ranking weights must be finite and non-negative.")
    if float(weights.sum().item()) <= 0:
        raise ValueError("Ranking weights must have a positive total.")
    acquisition_fraction, novelty_fraction = weights / weights.sum()
    batches, sources, metadata = [], [], []
    for source, conditions in (
        ("global", global_conditions),
        ("local", local_conditions),
    ):
        if conditions.numel():
            batches.append(conditions)
            sources.extend([source] * conditions.shape[0])
            if source == "local" and local_metadata is not None:
                if len(local_metadata) != conditions.shape[0]:
                    raise ValueError("local metadata must align with local conditions")
                metadata.extend(local_metadata)
            else:
                metadata.extend({} for _ in range(conditions.shape[0]))
    if not batches:
        return []
    conditions = torch.cat(batches)
    valid_indices = []
    for index, point in enumerate(conditions):
        if not torch.isfinite(point).all():
            continue
        if bool(
            ((point < constraints.box_bounds[0]) | (point > constraints.box_bounds[1])).any()
        ):
            continue
        if feasible_mask(point.unsqueeze(0), constraints)[0]:
            valid_indices.append(index)
    if not valid_indices:
        return []
    conditions = conditions[valid_indices]
    sources = [sources[index] for index in valid_indices]
    metadata = [metadata[index] for index in valid_indices]
    with torch.no_grad():
        acquisition = acq_function(conditions.unsqueeze(1)).reshape(-1)

    lower, upper = constraints.box_bounds
    scale = (upper - lower).clamp_min(1e-12)

    def distance_features(points):
        if ellipse_feature_mode == RAW_ELLIPSE_FEATURES:
            return (points - lower) / scale
        return design_features(
            points,
            constraints.nholes,
            constraints.xsize,
            constraints.ysize,
            mode=ellipse_feature_mode,
        )

    attempted_normalized = [
        distance_features(point.detach().reshape(1, -1)).reshape(-1)
        for point in attempted_starts
    ]
    observed_normalized = []
    if observed_points is not None:
        observed_normalized = [
            distance_features(point.detach().reshape(1, -1)).reshape(-1)
            for point in observed_points
        ]
    normalized = distance_features(conditions)
    if observed_normalized:
        observed_distance = torch.cdist(
            normalized, torch.stack(observed_normalized)
        ).min(dim=1).values
    else:
        observed_distance = torch.full(
            (len(conditions),), float("inf"), **TENSOR_KWARGS
        )
    observed_start = [
        metadata[index].get("allow_observed_start", False)
        and bool(observed_distance[index] <= 1e-12)
        for index in range(len(conditions))
    ]

    if global_selection_strategy == "greedy_dynamic_novelty":
        accepted_start_signatures = set()
        eligible_indices = []
        signatures = {}
        for index, point in enumerate(conditions):
            if not torch.isfinite(acquisition[index]):
                continue
            signature = geometry_signature(point, constraints)
            allow_observed = observed_start[index]
            if signature in accepted_start_signatures:
                continue
            if signature in excluded_signatures and not allow_observed:
                continue
            accepted_start_signatures.add(signature)
            signatures[index] = signature
            eligible_indices.append(index)

        if eligible_indices:
            raster_feasible, _ = rasterized_volume_feasibility(
                conditions[eligible_indices], constraints
            )
            eligible_indices = [
                index
                for index, is_feasible in zip(eligible_indices, raster_feasible)
                if bool(is_feasible)
            ]

        global_remaining = [
            index
            for index in eligible_indices
            if sources[index] == "global"
        ]
        selected_global = []
        global_entries = []
        global_target = (
            len(global_remaining)
            if number_global is None
            else max(0, int(number_global))
        )
        while global_remaining:
            references = attempted_normalized + observed_normalized + selected_global
            if references:
                reference_tensor = torch.stack(references)
                distances = torch.cdist(
                    normalized[global_remaining], reference_tensor
                ).min(dim=1).values
            else:
                distances = torch.full(
                    (len(global_remaining),), float("inf"), **TENSOR_KWARGS
                )
            distance_eligible = [
                position
                for position, value in enumerate(distances)
                if value >= minimum_distance
            ]
            if not distance_eligible:
                break
            candidate_indices = [
                global_remaining[position] for position in distance_eligible
            ]
            candidate_distances = distances[distance_eligible]
            acquisition_score = _unit(acquisition[candidate_indices])
            novelty_score = _unit(candidate_distances)
            combined_score = (
                acquisition_fraction * acquisition_score
                + novelty_fraction * novelty_score
            )
            chosen_position = max(
                range(len(candidate_indices)),
                key=lambda position: (
                    float(combined_score[position].item()),
                    float(acquisition_score[position].item()),
                    float(novelty_score[position].item()),
                    -candidate_indices[position],
                ),
            )
            index = candidate_indices[chosen_position]
            point = conditions[index].detach().reshape(-1)
            selection_step = len(global_entries) + 1
            entry = {
                "point": point,
                "source": "global",
                "shape_tuple": tuple(
                    int(point[7 * hole].item())
                    for hole in range(constraints.nholes)
                ),
                "raw_acquisition_value": acquisition[index].detach(),
                "normalized_acquisition_score": acquisition_score[
                    chosen_position
                ].detach(),
                "raw_novelty_distance": candidate_distances[
                    chosen_position
                ].detach(),
                "normalized_novelty_score": novelty_score[
                    chosen_position
                ].detach(),
                "rank_score": combined_score[chosen_position].detach(),
                "global_selection_step": selection_step,
                "global_selection_strategy": global_selection_strategy,
                "reserve_start": selection_step > global_target,
                "signature": signatures[index],
            }
            if ellipse_feature_mode != RAW_ELLIPSE_FEATURES:
                entry["distance_feature"] = normalized[index].detach()
            entry.update(metadata[index])
            global_entries.append(entry)
            selected_global.append(normalized[index])
            global_remaining.remove(index)

        local_indices = [
            index
            for index in eligible_indices
            if sources[index] == "local"
        ]
        local_acquisition_score = _unit(acquisition[local_indices])
        local_entries = []
        for position in torch.argsort(local_acquisition_score, descending=True):
            local_position = int(position.item())
            index = local_indices[local_position]
            allow_observed = observed_start[index]
            references = list(attempted_normalized)
            if not allow_observed:
                references.extend(observed_normalized)
            novelty = conditions.new_tensor(1.0)
            if references:
                novelty = torch.cdist(
                    normalized[index : index + 1], torch.stack(references)
                ).min()
            if references and novelty < minimum_distance:
                continue
            point = conditions[index].detach().reshape(-1)
            entry = {
                "point": point,
                "source": "local",
                "shape_tuple": tuple(
                    int(point[7 * hole].item())
                    for hole in range(constraints.nholes)
                ),
                "raw_acquisition_value": acquisition[index].detach(),
                "normalized_acquisition_score": local_acquisition_score[
                    local_position
                ].detach(),
                "rank_score": local_acquisition_score[local_position].detach(),
                "global_selection_strategy": global_selection_strategy,
                "signature": signatures[index],
            }
            if ellipse_feature_mode != RAW_ELLIPSE_FEATURES:
                entry["distance_feature"] = normalized[index].detach()
            entry.update(metadata[index])
            local_entries.append(entry)
        return global_entries + local_entries

    novelty = torch.ones(len(conditions), **TENSOR_KWARGS)
    for index in range(len(conditions)):
        references = list(attempted_normalized)
        if not observed_start[index]:
            references.extend(observed_normalized)
        if references:
            novelty[index] = torch.cdist(
                normalized[index : index + 1], torch.stack(references)
            ).min()
    acquisition_score = _unit(acquisition)
    novelty_score = _unit(novelty)
    score = torch.stack(
        [
            acquisition_fraction * acquisition_score[index]
            + novelty_fraction * novelty_score[index]
            if source == "global"
            else acquisition_score[index]
            for index, source in enumerate(sources)
        ]
    )

    accepted_start_signatures = set()
    ranked = []
    for index_tensor in torch.argsort(score, descending=True):
        index = int(index_tensor.item())
        if not torch.isfinite(acquisition[index]):
            continue
        point = conditions[index].detach().reshape(-1)
        signature = geometry_signature(point, constraints)
        allow_observed = observed_start[index]
        if signature in accepted_start_signatures:
            continue
        if signature in excluded_signatures and not allow_observed:
            continue
        has_references = bool(attempted_normalized) or (
            bool(observed_normalized) and not allow_observed
        )
        if has_references and novelty[index] < minimum_distance:
            continue
        entry = {
            "point": point,
            "source": sources[index],
            "shape_tuple": tuple(
                int(point[7 * hole].item()) for hole in range(constraints.nholes)
            ),
            "raw_acquisition_value": acquisition[index].detach(),
            "rank_score": score[index].detach(),
            "global_selection_strategy": global_selection_strategy,
            "signature": signature,
        }
        if ellipse_feature_mode != RAW_ELLIPSE_FEATURES:
            entry["distance_feature"] = normalized[index].detach()
        entry.update(metadata[index])
        ranked.append(entry)
        accepted_start_signatures.add(signature)
    return ranked


def take_restart_batch(
    ranked_pool,
    number_global: int,
    number_local: int,
    nholes: int,
    diversity: bool = True,
    global_selection_strategy: str = "static_rank",
):
    """Take a source-balanced, spatially diverse batch across shape tuples."""
    target_total = min(number_global + number_local, len(ranked_pool))
    if not target_total:
        return [], list(ranked_pool)
    if global_selection_strategy == "greedy_dynamic_novelty":
        global_indices = sorted(
            (
                index
                for index, entry in enumerate(ranked_pool)
                if entry["source"] == "global"
            ),
            key=lambda index: ranked_pool[index].get(
                "global_selection_step", float("inf")
            ),
        )[:number_global]
        local_pool = [
            entry for entry in ranked_pool if entry["source"] == "local"
        ]
        selected_local, _ = take_restart_batch(
            local_pool,
            0,
            number_local,
            nholes,
            diversity=diversity,
            global_selection_strategy="static_rank",
        )
        selected_ids = {id(entry) for entry in selected_local}
        selected_indices = list(global_indices) + [
            index
            for index, entry in enumerate(ranked_pool)
            if id(entry) in selected_ids
        ]
        if len(selected_indices) < target_total:
            selected_set = set(selected_indices)
            selected_indices.extend(
                index
                for index in range(len(ranked_pool))
                if index not in selected_set
            )
            selected_indices = selected_indices[:target_total]
        selected_set = set(selected_indices)
        return (
            [
                entry
                for index, entry in enumerate(ranked_pool)
                if index in selected_set
            ],
            [
                entry
                for index, entry in enumerate(ranked_pool)
                if index not in selected_set
            ],
        )
    if global_selection_strategy != "static_rank":
        raise ValueError(
            "global_selection_strategy must be 'greedy_dynamic_novelty' "
            "or 'static_rank'."
        )
    source_remaining = {"global": number_global, "local": number_local}
    selected_indices, selected_points, represented_shapes = [], [], set()

    def distance(entry):
        if not selected_points:
            return float("inf")
        feature = entry.get("distance_feature", entry["point"])
        return min(
            float(torch.linalg.vector_norm(feature - point).item())
            for point in selected_points
        )

    while len(selected_indices) < target_total:
        available = [
            index
            for index, entry in enumerate(ranked_pool)
            if index not in selected_indices
            and source_remaining.get(entry["source"], 0) > 0
        ]
        if not available:
            available = [
                index for index in range(len(ranked_pool)) if index not in selected_indices
            ]
        if diversity:
            unseen = []
            for index in available:
                entry = ranked_pool[index]
                key = entry.get("shape_tuple") or tuple(
                    int(entry["point"][7 * hole].item()) for hole in range(nholes)
                )
                if key not in represented_shapes:
                    unseen.append(index)
            if unseen:
                available = unseen
            chosen = max(
                available,
                key=lambda index: (
                    distance(ranked_pool[index]),
                    float(ranked_pool[index].get("rank_score", 0.0)),
                ),
            )
        else:
            chosen = available[0]
        entry = ranked_pool[chosen]
        selected_indices.append(chosen)
        selected_points.append(entry.get("distance_feature", entry["point"]))
        key = entry.get("shape_tuple") or tuple(
            int(entry["point"][7 * hole].item()) for hole in range(nholes)
        )
        represented_shapes.add(key)
        if source_remaining.get(entry["source"], 0) > 0:
            source_remaining[entry["source"]] -= 1

    selected_set = set(selected_indices)
    return (
        [entry for index, entry in enumerate(ranked_pool) if index in selected_set],
        [entry for index, entry in enumerate(ranked_pool) if index not in selected_set],
    )
