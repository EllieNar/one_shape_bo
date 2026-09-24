# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:03:58 2026

@author: pemb6626
"""
import torch
import numpy as np
from botorch.optim import optimize_acqf_mixed
from botorch.generation.gen import gen_candidates_scipy
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.utils.acquisition_utils import fix_features
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP
from botorch.models.transforms import Log, Normalize, Standardize
from botorch.exceptions.errors import (
    BotorchError,
    CandidateGenerationError,
    OptimizationGradientError,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.priors import GammaPrior
from dataclasses import dataclass
from typing import Optional, Sequence, Literal, Tuple
import warnings
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {
    "device": device,
    "dtype": torch.double,
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
from Functions.Shape import triangle, ellipse, Area_Fraction, Perimeter_Fraction, separating_axis_theorem, smooth_min
from Functions.Sampling import canonicalise_shape_blocks, fixed_features_specific, feasible_mask
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
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
        **kwargs):
    """Run SLSQP in free dimensions and reconstruct fixed features exactly once."""
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
        raise NotImplementedError(
            "This mixed acquisition wrapper supports the nonlinear constraints "
            "used by Bounds, but not additional linear constraints."
        )

    dimension = initial_conditions.shape[-1]
    fixed_indices = sorted(fixed_features)
    free_indices = [i for i in range(dimension) if i not in fixed_features]
    reduced_acquisition = FixedFeatureAcquisitionFunction(
        acq_function=acquisition_function,
        d=dimension,
        columns=fixed_indices,
        values=[fixed_features[i] for i in fixed_indices],
    )

    initial_full = initial_conditions.reshape(-1, dimension)[0]
    reduced_nonlinear_constraints = []
    for constraint, intra_point in nonlinear_inequality_constraints or []:
        initial_slack = float(constraint(initial_full).detach().item())
        feasibility_buffer = max(0.0, min(1e-8, 0.5 * initial_slack))

        def reduced_constraint(
                X,
                constraint=constraint,
                feasibility_buffer=feasibility_buffer):
            full_X = fix_features(
                X,
                fixed_features=fixed_features,
                replace_current_value=False,
            )
            return constraint(full_X) - feasibility_buffer

        reduced_nonlinear_constraints.append((reduced_constraint, intra_point))

    scipy_options = dict(options or {})
    scipy_options.pop("batch_limit", None)
    reduced_candidates, values = gen_candidates_scipy(
        initial_conditions=initial_conditions[..., free_indices],
        acquisition_function=reduced_acquisition,
        lower_bounds=(
            lower_bounds[free_indices] if lower_bounds is not None else None
        ),
        upper_bounds=(
            upper_bounds[free_indices] if upper_bounds is not None else None
        ),
        nonlinear_inequality_constraints=reduced_nonlinear_constraints,
        options=scipy_options,
        timeout_sec=timeout_sec,
    )
    candidates = fix_features(
        reduced_candidates,
        fixed_features=fixed_features,
        replace_current_value=False,
    )
    return candidates, values


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class Bounds:
    def __init__(self, xsize, ysize, nholes, bx, by, solid_max, smin, amin, raster_tolerance, quality_min):
        
        self.xsize      = xsize
        self.ysize      = ysize
        self.nholes     = nholes
        self.bx         = bx
        self.by         = by
        self.solid_max  = solid_max
        self.smin       = smin
        self.amin       = amin
        self.quality_min= float(quality_min)
        self.nonlinear_constraints  = []
        self.box        = (bx, xsize - bx, by, ysize - by)
        self.raster_tolerance = raster_tolerance
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        # Bounds per hole        
        lower = torch.zeros(7*nholes, **tkwargs)
        upper = torch.ones(7*nholes, **tkwargs)
        self.box_bounds = torch.stack([lower, upper],)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        # Solid fraction constraint
        def solid_fraction_constraint(X):
            total_void_area = X.new_zeros(())

            for i in range(self.nholes):
                j = 7*i
            
                T = triangle(X, j, self.xsize, self.ysize)
                E = ellipse(X, j, self.xsize, self.ysize)
            
                tri_area = 0.5*(T[2]*T[5] - T[3]*T[4])
                ell_area = torch.pi*E[2]*E[3]
            
                total_void_area = total_void_area + torch.where(
                    X[j] < 0.5,
                    tri_area,
                    ell_area,
                )
            
            solid_fraction = 1.0 - total_void_area/(self.xsize*self.ysize)
            return self.solid_max - solid_fraction

        self.nonlinear_constraints.append((solid_fraction_constraint, True))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        # Permutation constraints
        # (holes ordered by centroid, NB holes already canonical in tri, ell order as X input)
        for i in range(nholes - 1):
            def centroid_x_constraint(X, i=i):
                a = 7 * i
                b = 7 * (i + 1)
            
                T_a = triangle(X, a, self.xsize, self.ysize)
                T_b = triangle(X, b, self.xsize, self.ysize)
                E_a = ellipse(X, a, self.xsize, self.ysize)
                E_b = ellipse(X, b, self.xsize, self.ysize)

                cx_a_tri = T_a[0] + (T_a[2] + T_a[4])/3.0
                cx_b_tri = T_b[0] + (T_b[2] + T_b[4])/3.0
                cx_a_ell = E_a[0]
                cx_b_ell = E_b[0]
            
                # Shape is fixed during each mixed optimization
                cx_a = torch.where(X[a] < 0.5, cx_a_tri, cx_a_ell)
                cx_b = torch.where(X[b] < 0.5, cx_b_tri, cx_b_ell)
            
                same_shape = (X[a] < 0.5) == (X[b] < 0.5)
                margin = (cx_b - cx_a)/self.xsize
                return torch.where(same_shape, margin, torch.ones_like(cx_a))
            
            self.nonlinear_constraints.append((centroid_x_constraint, True))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        for i in range(nholes):   
            # Minimum area constraint
            def minimum_area_constraint(X, i=i):
                j = 7*i
                
                T = triangle(X, j, xsize, ysize)
                E = ellipse(X, j, xsize, ysize)       
                
                tri_area = 0.5*(T[2]*T[5] - T[3]*T[4])
                ell_area = torch.pi*E[2]*E[3]
                
                A = torch.where(X[j] < 0.5, tri_area, ell_area,)
                return (A - self.amin)/(self.xsize*self.ysize)
            
            self.nonlinear_constraints.append((minimum_area_constraint, True))            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            # Shape-specific quality constraint
            def shape_quality_constraint(X, i=i, eps=1e-12):
                j = 7*i
                T = triangle(X, j, self.xsize, self.ysize)
                v1 = T[2:4]
                v2 = T[4:6]
                signed_twice_area = v1[0]*v2[1] - v1[1]*v2[0]
                v3 = v2 - v1
                squared_edge_sum = (torch.dot(v1, v1) + torch.dot(v2, v2) + torch.dot(v3, v3) + eps)
                tri_quality = 2*np.sqrt(3)*signed_twice_area/squared_edge_sum

                E = ellipse(X, j, self.xsize, self.ysize)
                a = E[2]
                b = E[3]
                
                L = max(self.xsize, self.ysize)
                
                ell_margins = torch.stack((
                    (a - self.quality_min*b)/L,
                    (b - self.quality_min*a)/L,))
                
                ell_quality = smooth_min(ell_margins)
                
                quality = torch.where(X[j] < 0.5, tri_quality-self.quality_min, ell_quality)
                return quality

            self.nonlinear_constraints.append((shape_quality_constraint, True))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            # Keep p0 as the leftmost triangle vertex during local refinement.
            def triangle_anchor_constraint(X, i=i):
                j = 7*i
                T = triangle(X, j, self.xsize, self.ysize)
                anchor_margins = torch.stack((T[2], T[4]))/self.xsize
                anchor_margin = smooth_min(anchor_margins)
                
                return torch.where(X[j] < 0.5, anchor_margin, torch.ones_like(anchor_margin))

            self.nonlinear_constraints.append((triangle_anchor_constraint, True))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            # Boundary constraint 1 - to the plate edge
            def boundary_constraint(X, i=i):
                # Triangle
                T = triangle(X, 7*i, self.xsize, self.ysize)
                p0 = T[0:2]
                v1 = T[2:4]
                v2 = T[4:6]
                
                length_scale = max(self.xsize, self.ysize)
                
                vertices = torch.stack((p0, p0 + v1, p0 + v2))
                
                tri_margins = torch.cat((vertices[:,0] - self.box[0], self.box[1] - vertices[:, 0], vertices[:, 1] - self.box[2], self.box[3] - vertices[:, 1],))/length_scale
                
                tri_min = smooth_min(tri_margins)
                
                # Ellipse
                E = ellipse(X, 7*i, self.xsize, self.ysize)
                cx, cy, rx, ry, theta = E
                
                dx = torch.sqrt((rx*torch.cos(theta))**2 + (ry*torch.sin(theta))**2 + 1e-24)
                dy = torch.sqrt((rx*torch.sin(theta))**2 + (ry*torch.cos(theta))**2 + 1e-24)
                
                ell_margins = torch.stack((cx - dx - self.box[0], self.box[1] - cx - dx, cy - dy - self.box[2], self.box[3] - cy - dy))/length_scale
                
                ell_min = smooth_min(ell_margins)
                
                return torch.where(X[7*i] < 0.5, tri_min, ell_min)
            
            self.nonlinear_constraints.append((boundary_constraint, True))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        # Pairwise minimum-clearance constraints in physical co-ordinates.
        for i in range(nholes):
            for j in range(i + 1, nholes):
                def spacing_constraint(X, i=i, j=j):
                    points_i = X[7*i:7*(i + 1)]
                    points_j = X[7*j:7*(j + 1)]
                    separation = separating_axis_theorem(
                        points_i,
                        points_j,
                        self.xsize,
                        self.ysize,
                    )
                    return (separation - self.smin)/max(self.xsize, self.ysize)

                self.nonlinear_constraints.append((spacing_constraint, True))
        
class Acquisition:
    # This uses optimize_acqf_mixed
    def __init__(self, acq_function, constraints, initial_conditions):
        self.acq_function = acq_function
        self.successful_results    = []
        self.failed_restarts        = []
        
        for restart_index in range(initial_conditions.shape[0]):
            initial_point       = canonicalise_shape_blocks(initial_conditions[restart_index], constraints.nholes)
            restart_initial_condition = initial_point.reshape(1, 1, -1)
            
            print(f"Restart Index = {restart_index}")
            print(f"Initial Point = {initial_point}")
            
            if not torch.isfinite(initial_point).all():
                self.failed_restarts.append({
                    "restart_index": restart_index,
                    "initial_point": initial_point,
                    "reason": "non-finite initial condition",
                })
                continue
            if bool(((initial_point < constraints.box_bounds[0]) | (initial_point > constraints.box_bounds[1])).any()):
                self.failed_restarts.append({
                    "restart_index": restart_index,
                    "initial_point": initial_point,
                    "reason": "initial condition outside box bounds",
                })
                continue
            if not bool(feasible_mask(initial_point.reshape(1, -1), constraints)[0]):
                self.failed_restarts.append({
                    "restart_index": restart_index,
                    "initial_point": initial_point,
                    "reason": "infeasible initial condition",
                })
                continue
            
            #= pd.DataFrame({"Index":[], "Acq":[], "Acq Grad Norm": [], "Failed?"[]})
            # Bound pathological SLSQP restarts; the ranked feasible raw point remains available as a safe fallback
            
            # May wish to change the options/timeout_sec/retry_on_optimisation warning
            # At present, this restricts the computational effort of each local acquisition optimisation (not the search space itself):
                # maxiter = 50: SLSQP may stop before fully converging to the local acquisition maximum
                # timeout_sec = 5: similarly terminates a local solve if it takes too long
            # However, there is a risk on optimisation quality as a promising restart could be terminated prematurely, and return a suboptimal candidate.

            try:
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    candidate, value = optimize_acqf_mixed(
                        acq_function = self.acq_function,
                        bounds = constraints.box_bounds,
                        q=1,
                        num_restarts=1,
                        raw_samples = None,
                        fixed_features_list = fixed_features_specific(initial_point, constraints.nholes),
                        nonlinear_inequality_constraints = constraints.nonlinear_constraints,
                        batch_initial_conditions=restart_initial_condition,
                        options = {"batch_limit":1, "maxiter":250, "ftol":1e-9},
                        timeout_sec = 60.0,
                        retry_on_optimization_warning = False,
                        gen_candidates=gen_candidates_scipy_single_reconstruction,)
                
                elapsed = time.perf_counter() - start
                print(f"Optimisation time: {elapsed:.2f} s")
                for w in caught:
                    print(f"Optimiser warning: {w.message}")
                shape = tuple(int(initial_point[7*h]) for h in range(constraints.nholes))
                print(
                    f"shape={shape}, "
                    f"time={elapsed:.2f}s, "
                    f"acq={value.item():.4g}"
                )
                
                successful_restart = restart_index

                
                candidate_row = candidate.detach().reshape(1, -1)
                candidate_in_bounds = bool(
                    ((candidate_row >= constraints.box_bounds[0]) &
                     (candidate_row <= constraints.box_bounds[1])).all()
                )
                
                print(f"Candidate row = {candidate_row} and candidate in bounds is {candidate_in_bounds}")
                if candidate_in_bounds and feasible_mask(candidate_row, constraints).all():
                    self.successful_results.append({
                        "restart_index": restart_index,
                        "initial_point": initial_point,
                        "candidate": candidate_row,
                        "acquisition_value": value.reshape(-1)[0].detach(),})
                    
                    
                else:
                    self.failed_restarts.append({
                        "restart_index": restart_index,
                        "initial_point": initial_point,
                        "reason": "infeasible optimizer result",})
                    
                    
            except (
                    BotorchError,
                    ValueError,
                    RuntimeError,
                    CandidateGenerationError,
                    OptimizationGradientError) as error:
                self.failed_restarts.append({
                    "restart_index": restart_index,
                    "initial_point": initial_point,
                    "reason": str(error),
                })
                
                
        
        self.successful_results.sort(
            key=lambda result: float(
                result["acquisition_value"].item()
            ),
            reverse=True,
        )
        
