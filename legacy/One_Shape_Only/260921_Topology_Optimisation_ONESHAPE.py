# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 09:59:38 2026

@author: pemb6626
"""

# Import functions:
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import gpytorch
from botorch.acquisition import AnalyticAcquisitionFunction, ExpectedImprovement, LogExpectedImprovement
import matplotlib.pyplot as plt
from shapely.affinity import rotate, scale
from shapely.geometry import Point
from shapely.geometry import Polygon

import sys

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {
    "device": device,
    "dtype": torch.double,
}

SHAPE = "Triangle"

dir_save = "/home/eleno/projects/one_shape_bo/outputs"
fol_save = SHAPE + "/260923/"
#fil_name = "260923"
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            if not stream.closed:
                stream.write(text)
                stream.flush()

    def flush(self):
        for stream in self.streams:
            if not stream.closed:
                stream.flush()


original_stdout = sys.stdout
original_stderr = sys.stderr

with open(dir_save + fol_save + f"output_log.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)
    
    from Functions.Gaussian import Pred_Objective_Multi_Task
    from Functions.Bayesian import Acquisition, Bounds
    from Functions.Sampling import sample_stratified_feasible, rank_initial_condition_pool, take_restart_batch, feasible_mask, sample_local_feasible
    from Functions.Shape import Read_Shape
    from Functions.Physical import geometry_signature, rasterized_volume_feasibility, Evaluate_Designs, Generate_Geometry, analytical_solid_fraction


    try:
        print("OUTPUT LOG")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        def plot(fig, ax, X, nholes):
            T, E = Read_Shape(X, nholes, xsize, ysize)
            for triangle in T:
                p0 = triangle[0,:]
                p1 = p0 + triangle[1,:]
                p2 = p0 + triangle[2,:]
                
                tri = Polygon([(p0[0].detach().cpu().numpy(), p0[1].detach().cpu().numpy()),
                                (p1[0].detach().cpu().numpy(), p1[1].detach().cpu().numpy()),
                                (p2[0].detach().cpu().numpy(), p2[1].detach().cpu().numpy())])
                x, y = tri.exterior.xy
                ax.plot(x, y, color="r", alpha=0.6)
                ax.fill(x, y, color="r", alpha=0.6)
                
            for ellipse in E:
                cx = ellipse[0].detach().cpu().numpy()
                cy = ellipse[1].detach().cpu().numpy()
                rx = ellipse[2].detach().cpu().numpy()
                ry = ellipse[3].detach().cpu().numpy()
                th = np.degrees(ellipse[4].detach().cpu().numpy())
                
                circ = Point(cx, cy).buffer(1, resolution=64)
                ell = scale(circ, xfact=rx, yfact=ry)
                ell = rotate(ell, th, origin=(cx, cy))
                x, y = ell.exterior.xy
                ax.plot(x, y, color="b", alpha=0.6)
                ax.fill(x, y, color="b", alpha=0.6)
                
            ax.set_xlim(0, xsize)
            ax.set_ylim(0, ysize)
        
        
        def is_novel(candidate, existing_points, atol=1e-8):
            candidate = candidate.detach().reshape(1, -1)
            if existing_points.numel() == 0:
                return True
            duplicate = torch.isclose(existing_points, candidate, rtol=0.0, atol=atol).all(dim=1)
            return not bool(duplicate.any())
            
        if __name__ == "__main__":
                
            xsize               = 100
            ysize               = 50
            amin                = 50
            smin                = 1
            nholes              = 3
            design_dimension    = 7*nholes
            raw_pool_multiplier = 3
            max_restart_batches = 6
            ng_rst              = 8
            nl_rst              = 8
            minimum_restart_distance = 0.02
            plot_every          = 10
            local_radius        = 0.05
            nbrin               = 5*design_dimension
            niter               = 25*design_dimension
            bx                  = 1
            by                  = 1
            solid_max           = 0.5
            raster_tolerance    = 0.005
            tri_q_min           = 0.10
            Emin                = 1e-9
            E0                  = 1.0
            penal               = 3
            
            if SHAPE == "Triangle":
                htype = 0
            elif SHAPE == "Ellipse":
                htype = 1
            else:
                print("Error htype!")
            
            print(f"""Number of x-pixels                          = {xsize}
Number of y-pixels                          = {ysize}
Minimum hole area                           = {amin}
Number of holes                             = {nholes}
Design dimension                            = {design_dimension}
Number of burn-in                           = {nbrin}
Number of acquisition evaluations           = {niter}
Number of global random restarts            = {ng_rst}
Number of local random restarts             = {nl_rst}
Raw pool multiplier                         = {raw_pool_multiplier}
Maximum number of restart batch trials      = {max_restart_batches}
Minimum scalar hole distance                = {minimum_restart_distance}
Local radius (local exploitation value)     = {local_radius}
Minimum Young's Modulus                     = {Emin}
E0                                          = {E0}
penal                                       = {penal}
bx (border from x-edge)                     = {bx}
by (border from y-edge)                     = {by}
smin (min spacing between perforations)     = {smin}
Solid ratio                                 = {solid_max}
Raster tolerance                            = {raster_tolerance}
Triangle quality min                        = {tri_q_min}""")
            
            Constraints = Bounds(xsize, ysize, nholes, bx, by, solid_max, smin, amin, raster_tolerance, tri_q_min)
            
            # Ensure that there is a spread of shape multiset, so that the GP sees each a similar number of times
            x_train = sample_stratified_feasible(nbrin, nholes, htype, Constraints, amin, Print=True, stage="Burn_In")
            y_train, raster_train = Evaluate_Designs(x_train, xsize, ysize, nholes, Emin, E0, penal, Constraints, return_solid_fractions=True, Print=True, dir_save=dir_save, fol_save=fol_save)
            analytical_burn     = analytical_solid_fraction(x_train, Constraints, Print=True)
            best_history        = [min(y_train).item()]
            initial_best            = best_history[0]
            observed_signatures     = {geometry_signature(design, Constraints) for design in x_train}
            previous_model          = None
            bo_record_improvements  = 0
            bo_record_magnitudes    = []
            
            print(f". . . . . . . . . . . . . B U R N    I N    C O M P L E T E! . . . . . . . . . . . . . ")
            
            for iteration in range(1, niter + 1):
                model               = Pred_Objective_Multi_Task(x_train, y_train, nholes)
                x_new               = None
                last_error          = None
                candidate_signature = None
                acq_function        = LogExpectedImprovement(model=model.gp, best_f=model.best_f)
                attempted_starts = []
                iteration_rejected_signatures = set()
                restart_failure_records = []
                
                try:
                    global_raw_conditions = sample_stratified_feasible(ng_rst*raw_pool_multiplier, nholes, htype, Constraints, amin, allow_partial = True)
                except RuntimeError as error:
                    global_raw_conditions = torch.empty((0, design_dimension), **tkwargs)
                    last_error = error
                    print(f"Iteration {iteration}: initial global raw-pool generation failed: {error}")
                
                local_raw_conditions = sample_local_feasible(nl_rst*raw_pool_multiplier,x_train,y_train,Constraints,radius=local_radius,)
                ranked_restart_pool = rank_initial_condition_pool(acq_function,global_raw_conditions,local_raw_conditions,Constraints,observed_signatures,attempted_starts,minimum_restart_distance,)       
                print(
                    f"Iteration {iteration}: feasible raw pool "
                    f"global={global_raw_conditions.shape[0]}, "
                    f"local={local_raw_conditions.shape[0]}, "
                    f"unique ranked={len(ranked_restart_pool)}."
                )
                
                for restart_batch in range(max_restart_batches):
                    if not ranked_restart_pool:
                        try:
                            global_raw_conditions = sample_stratified_feasible(ng_rst, nholes, htype, Constraints, amin, allow_partial = True)
                        except RuntimeError as error:
                            global_raw_conditions = torch.empty((0, design_dimension), **tkwargs)
                            last_error = error
                        
                        local_raw_conditions = sample_local_feasible(nl_rst, x_train, y_train, Constraints, radius=local_radius,)
                        excluded_pool_signatures = (observed_signatures | iteration_rejected_signatures)
                        ranked_restart_pool = rank_initial_condition_pool(acq_function, global_raw_conditions, local_raw_conditions, Constraints, excluded_pool_signatures, attempted_starts, minimum_restart_distance,)
                        
                    selected_starts, ranked_restart_pool = take_restart_batch(ranked_restart_pool, ng_rst, nl_rst, nholes, diversity=(iteration<=10),)
                    
                    if not selected_starts:
                        last_error = RuntimeError("No unseen feasible restart points were available.")
                        print(
                            f"Iteration {iteration}: restart batch "
                            f"{restart_batch + 1}/{max_restart_batches} "
                            f"has no unseen starts."
                        )
                        continue
                        
                    initial_conditions = torch.stack([entry["point"] for entry in selected_starts]).unsqueeze(1)
                    attempted_starts.extend([entry["point"].detach().clone()for entry in selected_starts])
                    acquisition = Acquisition(acq_function=acq_function,constraints=Constraints,initial_conditions=initial_conditions,)
                    restart_failure_records.extend(acquisition.failed_restarts)
                    rejection_reasons = []
                    
                    for result in acquisition.successful_results:
                        candidate = result["candidate"]
                        raster_valid, candidate_raster = (rasterized_volume_feasibility(candidate, Constraints))
                        signature = geometry_signature(candidate.squeeze(0), Constraints)
                        
                        if not raster_valid.all():
                            iteration_rejected_signatures.add(signature)
                            rejection_reasons.append(f"rasterized volume {candidate_raster.item():.6f}")
                            continue
                        
                        if (signature in observed_signatures or signature in iteration_rejected_signatures):
                            iteration_rejected_signatures.add(signature)
                            rejection_reasons.append("duplicate rasterized geometry")
                            continue
                        
                        x_new = candidate
                        candidate_signature = signature
                        break
                    
                    if x_new is not None:
                        break
                    
                    failure_count = len(acquisition.failed_restarts)
                    reason_summary = ", ".join(rejection_reasons[:3])
                    if not reason_summary:
                        reason_summary = "no successful local optima"
                    last_error = RuntimeError(reason_summary)
                    print(f"Iteration {iteration}: restart batch {restart_batch + 1}/{max_restart_batches} rejected ({failure_count} optimizer failures; {reason_summary}).")
                
                if x_new is None:
                    print(f"""Could not obtain a feasible, novel acquisition candidate after {len(attempted_starts)} unique restart optimisations across {max_restart_batches} batches.
        Optimiser failures: {len(restart_failure_records)}.
        Last error: {last_error}""")
                    break
                
                y_new, raster_new = Evaluate_Designs(x_new, xsize, ysize, nholes, Emin, E0, penal, Constraints, return_solid_fractions=True, dir_save=dir_save, fol_save=fol_save)
                x_train = torch.cat([x_train, x_new], dim=0)
                y_train = torch.cat([y_train, y_new], dim=0)
                raster_train = torch.cat([raster_train, raster_new], dim=0)
                
                previous_best = best_history[-1]
                current_best = y_train.min().item()
                best_history.append(current_best)
                observed_signatures.add(candidate_signature)
                
                # Minimisation, so improvement if less
                if current_best < previous_best:
                    bo_record_improvements += 1
                    bo_record_magnitudes.append(previous_best - current_best)
                    # CHANGED: expand after success to explore the improving basin.
                    local_radius = min(0.20, 1.20*local_radius)
                else:
                    # CHANGED: shrink after failure to refine locally, while unchanged
                    # global restarts preserve exploration of other basins.
                    local_radius = max(0.01, 0.80*local_radius)
                print(
                    f"BO iteration {iteration:4d}/{niter}: "
                    f"new compliance={y_new.item():.6f}, "
                    f"best={current_best:.6f}, "
                    f"binary solid={raster_new.item():.6f}, "
                    f"local radius={local_radius:.4f}"
                )
        
                # CHANGED: plot only the current best binary geometry periodically,
                # rather than opening a figure for every objective evaluation.
                if iteration % plot_every == 0:
                    current_best_index = y_train.argmin().item()
                    Generate_Geometry(xsize, ysize, x_train[current_best_index], nholes, plot=True, namesave=False, title=(f"Best binary geometry after BO iteration {iteration}, (compliance={current_best:.4f})"), dir_save=dir_save, fol_save=fol_save)
            
            best_index = y_train.argmin().item()
            best_design = x_train[best_index]
            best_compliance = y_train[best_index].item()
            best_analytical_solid = analytical_solid_fraction(best_design.unsqueeze(0), Constraints,)[0]
            best_rasterized_solid = raster_train[best_index].item()
            best_is_feasible = bool(feasible_mask(best_design.unsqueeze(0), Constraints).all() and best_rasterized_solid <= Constraints.solid_max + Constraints.raster_tolerance)
            best_source = (
                "initial design"
                if best_index < nbrin
                else f"BO iteration {best_index - nbrin + 1}"
            )
            improvement = 100.0 * (
                initial_best - best_compliance
            ) / max(abs(initial_best), 1e-12)
            
            # CHANGED: quantitative final report requested for reproducible comparison
            # with the TEMPLATE and for detecting volume/feasibility regressions.
            print("\nFinal Bayesian optimization report")
            print(f"  Best compliance:             {best_compliance:.8f}")
            print(f"  Initial best compliance:     {initial_best:.8f}")
            print(f"  Relative improvement:        {improvement:.3f}%")
            print(f"  Analytical solid fraction:   {best_analytical_solid:.6f}")
            print(f"  Rasterized solid fraction:   {best_rasterized_solid:.6f}")
            print(f"  Feasible:                    {best_is_feasible}")
            print(f"  Best design source:          {best_source}")
            print(f"  Total objective evaluations: {len(y_train)}")
            print(f"  BO record improvements:      {bo_record_improvements}")
            print(
                f"  Largest BO record gain:      "
                f"{max(bo_record_magnitudes, default=0.0):.8f}"
            )
            
            # CHANGED: display the convergence history and final thresholded geometry.
            fig, ax = plt.subplots()
            ax.plot(range(len(best_history)), best_history)
            ax.set_xlabel("BO iteration")
            ax.set_ylabel("Best compliance")
            ax.set_title("Bayesian optimization convergence")
            ax.grid(True, alpha=0.3)
            plt.savefig(dir_save + fol_save + "Convergence_History", dpi=500)
            plt.show()
            
            best_geometry = Generate_Geometry(
                xsize,
                ysize,
                best_design,
                nholes,
                plot=True,
                namesave="Best_Geometry",
                title=(
                    f"Final best binary geometry "
                    f"(compliance={best_compliance:.4f}, "
                    f"solid={best_rasterized_solid:.4f})"
                ),
                dir_save=dir_save,
                fol_save=fol_save,
            )
            
            print(f"THE BEST DESIGN IS X = {best_design}")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr