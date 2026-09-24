# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:04:06 2026

@author: pemb6626
"""

import torch
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {
    "device": device,
    "dtype": torch.double,
}

from Functions.Shape import anchor_point, separating_axis_theorem_batch, canonicalize_triangles, canonicalize_ellipses
from Functions.Physical import rasterized_volume_feasibility, geometry_signature

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def fixed_features(nholes, htype):
    "Return the all-triangle and all-ellipse shape configurations"
    fixed_features_list = []
    
    # Every design must contain only one shape type; mixed configurations are excluded.
    shapes = [htype]*nholes
    ff = {}
    for h, shape in enumerate(shapes):
        j = 7*h
        ff[j] = float(shape)
        if shape == 1:
            ff[j+6] = 0
    fixed_features_list.append(ff)
    
    return fixed_features_list
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def fixed_features_specific(point, nholes):
    X = canonicalise_shape_blocks(point, nholes)
    ff = {}
    for i in range(0, 7*nholes, 7):
        if int(X[i].item()) == 0:
            ff[i] = 0
        elif int(X[i].item()) == 1:
            ff[i] = 1
            ff[i + 6] = 0
    return [ff]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def canonicalise_shape_blocks(point, nholes):
    "Move complete hole blocks into triangle-then-ellipse order"
    blocks = point.detach().reshape(nholes, 7)
    order = torch.argsort(blocks[:, 0], stable=True)
    return blocks[order].reshape(-1).clone()


def canonicalise_design(point, nholes):
    """Canonicalise shape blocks, triangle vertices, and within-shape order."""
    blocks = canonicalise_shape_blocks(point, nholes).reshape(nholes, 7)
    ntri = int((blocks[:, 0] < 0.5).sum().item())

    tri_blocks = blocks[:ntri]
    if ntri:
        p0 = tri_blocks[:, 1:3]
        edge_1 = 2.0*tri_blocks[:, 3:5] - 1.0
        edge_2 = 2.0*tri_blocks[:, 5:7] - 1.0
        tri_vertices = torch.stack((p0, p0 + edge_1, p0 + edge_2), dim=1)
    else:
        tri_vertices = blocks.new_empty((0, 3, 2))

    tri_points = canonicalize_triangles(ntri, tri_vertices)
    ell_points = canonicalize_ellipses(nholes - ntri, blocks[ntri:, 1:6])
    return torch.cat((tri_points, ell_points))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def feasible_mask(X, constraints):
    feasible = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for constraint_function, intra_point in constraints.nonlinear_constraints:
        if not intra_point:
            raise NotImplementedError
        vals = torch.stack([constraint_function(x) for x in X]).to(X.device)
        feasible &= vals >= 0

    return feasible        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def random_shape_area(area, constraints, shape_type, shape_trials=100, Normalise=False, eps=1e-12):
    xmin, xmax, ymin, ymax  = constraints.box
    xsize = constraints.xsize
    ysize = constraints.ysize
    domain_extent           = torch.tensor([xmax - xmin, ymax - ymin], **tkwargs)
    shape_index  = torch.as_tensor([shape_type], **tkwargs)
    
    # Triangle
    if shape_type == 0:
        # Samples three random 2D points from a {-1, 1} square and computes the triangle area from these three points
        for _ in range(shape_trials):
            # Note that torch.rand generates random points between 0 and 1, hence *2 then sub 1 to get into desired -1 to 1 interval
            raw_vertices = 2.0* torch.rand((3, 2), **tkwargs) - 1.0
            edge_1       = raw_vertices[1] - raw_vertices[0]
            edge_2       = raw_vertices[2] - raw_vertices[0]
            edge_3       = raw_vertices[2] - raw_vertices[1]
            raw_area     = 0.5*torch.abs(edge_1[0]*edge_2[1] - edge_1[1]*edge_2[0])
            
            squared_edge_sum = (torch.dot(edge_1, edge_1) + torch.dot(edge_2, edge_2) + torch.dot(edge_3, edge_3)).clamp_min(eps)
            raw_quality      = 4.0*np.sqrt(3.0)*raw_area/squared_edge_sum
            
            # Triangle quality is invariant to scaling and directly excludes almost collinear shapes
            if raw_area <= eps or raw_quality< constraints.quality_min:
                continue
        
            vertices = (raw_vertices - raw_vertices[0])*torch.sqrt(area / raw_area.clamp_min(eps))
            extent   = vertices.max(dim=0).values - vertices.min(dim=0).values
            
            if bool((extent <= domain_extent).all()):
                
                if Normalise:
                    T = torch.stack((vertices[0, 0]/xsize, vertices[0, 1]/ysize, vertices[1, 0]/xsize, vertices[1, 1]/ysize, vertices[2, 0]/xsize, vertices[2, 1]/ysize)) 
                    return torch.cat((shape_index, T), dim=0)
                
                return torch.cat((shape_index, vertices.reshape(-1)), dim=0)
        return None
    
    # Ellipse
    else:
        # Samples three random 2D points from a {-1, 1} square and computes the triangle area from these three points
        for _ in range(shape_trials):
            # Random radius between {0, 0.5}
            random_radii    = 0.5*torch.rand((1, 2), **tkwargs).reshape(-1)
            # Random angle between 0 and pi
            random_angle    = torch.pi * torch.rand((), **tkwargs)
            
            raw_area        = torch.pi*random_radii[0]*random_radii[1]
            raw_quality = (torch.minimum(random_radii[0], random_radii[1])/ torch.maximum(random_radii[0], random_radii[1]).clamp_min(eps))
            
            if raw_area <= eps or raw_quality < constraints.quality_min:
                continue
            
            scaled_radii = (random_radii)*torch.sqrt(area / raw_area.clamp_min(eps))
            
            # Bounding box half extents of rotated ellipse
            dx = torch.sqrt((scaled_radii[0]*torch.cos(random_angle))**2 + (scaled_radii[1]*torch.sin(random_angle))**2)
            dy = torch.sqrt((scaled_radii[0]*torch.sin(random_angle))**2 + (scaled_radii[1]*torch.cos(random_angle))**2)
            
            if dx > (xmax - xmin)/2 or dy > (ymax - ymin)/2:
                continue
            
            # Sample the centre only from the valid region
            cx = xmin + dx + torch.rand((), **tkwargs)*(xmax - xmin - 2*dx)
            cy = ymin + dy + torch.rand((), **tkwargs)*(ymax - ymin - 2*dy)
            
            if Normalise:
                return torch.cat((shape_index, (cx/xsize).unsqueeze(0), (cy/ysize).unsqueeze(0), (scaled_radii[0]/xsize).unsqueeze(0), (scaled_radii[1]/ysize).unsqueeze(0), (random_angle/torch.pi).unsqueeze(0), torch.tensor([0.0], **tkwargs),))
            
            else:
                return torch.cat((shape_index, cx.unsqueeze(0), cy.unsqueeze(0), (scaled_radii[0]).unsqueeze(0), (scaled_radii[1]).unsqueeze(0), (random_angle).unsqueeze(0), torch.tensor([0.0], **tkwargs),))
            
        return None 

def random_shape_candidate(fixed_features, constraints, amin, anchor_trials=1500, shape_trials=200, dir_trials=100, stage="BO_Eval"):
    # fixed_features is dictionary e.g. [0, 0, 1] i.e. # tri and ell
    if float(amin) != float(constraints.amin):
        raise ValueError("amin must match constraints.amin.")

    nholes = constraints.nholes
    fixed_features_list = [int(fixed_features[7*h]) for h in range(nholes)]

    domain_area = constraints.xsize*constraints.ysize
    minimum_void_area = (1 - constraints.solid_max)*domain_area
    interior_area = (constraints.xsize - 2*constraints.bx)*(constraints.ysize - 2*constraints.by)
    if minimum_void_area > interior_area:
        return None
    maximum_sampled_void_area = min(interior_area, minimum_void_area + 0.02*domain_area)
    # Stay inside, rather than exactly on, the active volume constraint. This
    # gives SLSQP a useful feasible margin while keeping samples near the
    # compliance-favourable material limit.
    interior_sample = 0.05 + 0.90*torch.rand((), **tkwargs)
    target_void_area = minimum_void_area + interior_sample*(maximum_sampled_void_area - minimum_void_area)
    min_tot_area = nholes*amin
    
    if target_void_area < min_tot_area:
        return None
    
    remaining_area      = target_void_area - min_tot_area
    
    if stage == "BO_Eval":
        # Exponentially distributed random weights
        weights             = -torch.log(torch.rand(nholes, **tkwargs).clamp_min(1e-12))
    elif stage == "Burn_In":
        # Random similar weights = similar area
        weights             = 0.9 + 0.2 * torch.rand(nholes, **tkwargs)
    
    proportions         = weights/weights.sum()
    areas               = amin + remaining_area*proportions
    placement_order     = torch.argsort(areas, descending=True)
    shapes              = [None]*nholes
    
    # DOES THIS ENSURE THAT TRIANGLES AREN'T ALWAYS THE LARGEST? OR DO I NEED TO SHUFFLE THE FIXED FIEATURES LIST?
    for i in placement_order:
        index = int(i.item())
        # random_shape_area returns LOCAL (0 to 1) vertices as a 1D tensor length 7 i.e. [0, a, b, c, d, e, f]
        shapes[index] = random_shape_area(areas[index], constraints = constraints, shape_type = fixed_features_list[index], shape_trials=shape_trials, Normalise=True)
        if shapes[index] is None:
            return None
    
    placed = {}
    for index_tensor in placement_order:
        index = int(index_tensor.item())
        points = shapes[index]
        
        anchor_min = anchor_point(points=points, maximum=False, constraints=constraints, Normalise=True)
        anchor_max = anchor_point(points=points, maximum=True, constraints=constraints, Normalise=True)
        
        if bool((anchor_min > anchor_max).any()):
            return None
        
        # Decide where to locate the shapes by sampling many possible anchor points.
        # Each anchor shifts the origin anchor from random_triangle_shape into the domain.
        anchors     = anchor_min + (anchor_max - anchor_min)*torch.rand((anchor_trials, 2), **tkwargs)
        
        if fixed_features_list[index] == 0:
            vertices = (points[1:].reshape(3, 2)).unsqueeze(0) + anchors.unsqueeze(1)
            p0 = vertices[:, 0, :]
            encoded_edge_1 = 0.5*(vertices[:, 1, :] - p0 + 1.0)
            encoded_edge_2 = 0.5*(vertices[:, 2, :] - p0 + 1.0)
            options = torch.cat((points[0].expand(anchor_trials, 1), p0, encoded_edge_1, encoded_edge_2), dim=1)
            valid    = torch.ones(anchor_trials, dtype = torch.bool, device=device)
        
        else:
            #centre   = (points[1:3].reshape(1, 2)).unsqueeze(0) + anchors.unsqueeze(1)
            #options  = torch.cat((points[0], centre, points[3:]), dim=0)
            options = points.unsqueeze(0).expand(anchor_trials, -1).clone()
            options[:, 1:3] += anchors
            valid    = torch.ones(anchor_trials, dtype = torch.bool, device=device)
            
        for other_points in placed.values():
            valid &= separating_axis_theorem_batch(
                options,
                other_points,
                constraints.xsize,
                constraints.ysize,
                dir_trials,
            ) >= constraints.smin
            
        choices = torch.nonzero(valid, as_tuple=False).flatten()
        if choices.numel() == 0:
            return None
        
        # A valid placement is found and appended to placed. Continue to iterate through the remaining shapes
        choice = choices[torch.randint(choices.numel(), (), device=device)]
        placed[index] = options[choice]
    
    points = torch.stack([placed[i] for i in range(nholes)]).reshape(-1)
    print(f"The stage is {stage}")
    return canonicalise_design(points, nholes)

def sample_random_feasible(number, constraints, amin, fixed_features=None, batch_size=200, max_batches=250, anchor_trials=1000, shape_trials=500, dir_trials=200, allow_partial=False, Print=False, iters=None, stage="BO_Eval"):
    accepted = []
    max_attempts = batch_size*max_batches
    iters = 0 if iters is None else iters

    if fixed_features is None:
        nellipse = random.randint(0, constraints.nholes)
        shapes = [0]*(constraints.nholes - nellipse) + [1]*nellipse
        fixed_features = {7*h: float(shape) for h, shape in enumerate(shapes)}
    
    for _ in range(max_attempts):
        candidate = random_shape_candidate(fixed_features, constraints, amin, anchor_trials, shape_trials, dir_trials, stage=stage)
    
        if candidate is None:
            continue

        if not torch.isfinite(candidate).all():
            continue
        if bool(((candidate < 0.0) | (candidate > 1.0)).any()):
            continue
        
        if feasible_mask(candidate.unsqueeze(0), constraints)[0]:
            raster_valid, _ = rasterized_volume_feasibility(candidate.unsqueeze(0), constraints)
            if raster_valid[0]:
                accepted.append(candidate)
                iters += 1
                if Print:
                    print(f"Iteration {iters}: SUCCESSFUL")
                
        if len(accepted) >= number:
            return torch.stack(accepted[:number]), iters
    
    if allow_partial:
        if accepted:
            return torch.stack(accepted), iters
        return torch.empty((0, 7*constraints.nholes), **tkwargs), iters

    raise RuntimeError(
        "Could not generate enough independently random feasible points. "
        "Increase max_batches or anchor_trials, or relax the constraints."
    )


def sample_stratified_feasible(number, nholes, htype, constraints, amin, allow_partial=False, Print=False, stage="BO_Eval"):
    "Guarantee raw-pool coverage of every canonical shape multiset, thus permitting optimisation over discrete space (shape combinations)"
    "PROVIDED that number >= number of combinations (numc). For reference:"
    "nholes = 1 --> numc = 2"
    "nholes = 2 --> numc = 3"
    "nholes = 3 --> numc = 4"
    "nholes = 4 --> numc = 5"
    "i.e. number >= nholes + 1"
    shape_features = fixed_features(nholes, htype)
    quotient, remainder = divmod(number, len(shape_features))
    batches = []
    iters = 0
    
    for index, features in enumerate(shape_features):
        batch_size = quotient + int(index < remainder)
        if batch_size == 0:
            continue
        
        result, iters = sample_random_feasible(
            number = batch_size,
            constraints = constraints,
            amin = amin,
            fixed_features = features,
            allow_partial = allow_partial,
            Print=Print,
            iters=iters,
            stage=stage)
        
        if result.numel() == 0:
            if allow_partial:
                continue
            raise RuntimeError("Could not generate a feasible point for every shape multiset.")

        batches.append(result)

    if not batches:
        raise RuntimeError("Could not generate any feasible points.")
    return torch.cat(batches, dim=0)

def sample_local_feasible(number, x_train, y_train, constraints, radius=0.05, max_attempts=5000):
    # Unlike the global (sample_random_feasible), the local sampler does not (should not) have to be stratified (i.e. partitioned into 'equal' groups accross the design space)
    if number <= 0:
        return torch.empty((0, x_train.shape[-1]), **tkwargs)
    
    if len(x_train) == 0:
        return torch.empty((0, x_train.shape[-1]), **tkwargs)

    top_count       = min(10, len(x_train))
    top_indices     = torch.topk(y_train.squeeze(-1), top_count, largest=False).indices
    lower, upper    = constraints.box_bounds
    accepted        = []
    
    for attempt in range(max_attempts):
        base = x_train[top_indices[attempt % top_count]]
        # Try the requested neighbourhood first, then contract it if the mixed-shape constraints make acceptance sparse. This avoids spending thousands of expensive SAT evaluations at an unsuitable radius.
        contraction = min(3, attempt // max(1, 2*top_count))
        proposal_radius = radius*(0.5**contraction)
        output = []
        
        for h in range(constraints.nholes):
            shape_index = base[7*h]
            scale = upper[7*h:7*(h+1)] - lower[7*h:7*(h+1)] #1
            
            if shape_index == 0: # triangle
                active = base[7*h + 1 : 7*(h + 1)]
                proposal = torch.maximum(torch.minimum(active + proposal_radius*scale[1:]*torch.randn_like(active),upper[7*h + 1:7*(h+1)],),lower[7*h + 1:7*(h+1)],)
                hole = torch.cat((shape_index.unsqueeze(0), proposal))
            elif shape_index == 1: #ellipse
                active = base[7*h + 1: 7*h + 6]
                proposal = torch.maximum(torch.minimum(active + proposal_radius*scale[1:-1]*torch.randn_like(active),upper[7*h + 1:7*h + 6],),lower[7*h + 1:7*h + 6],)
                hole = torch.cat((shape_index.unsqueeze(0), proposal, base.new_zeros(1)))
            output.append(hole)
            
        candidate = canonicalise_design(torch.cat(output), constraints.nholes)
        
        if feasible_mask(candidate.unsqueeze(0), constraints)[0]:
            raster_valid, _ = rasterized_volume_feasibility(candidate.unsqueeze(0), constraints)
            if raster_valid[0]:
                accepted.append(candidate)
                if len(accepted) >= number:
                    break
    
    if not accepted:
        return torch.empty((0, x_train.shape[-1]), **tkwargs)
    
    return torch.stack(accepted)      
    
def rank_initial_condition_pool(acq_function, global_conditions, local_conditions, constraints, excluded_signatures, attempted_starts, minimum_distance):
    condition_batches = []
    sources = []
    
    if global_conditions.numel() > 0:
        condition_batches.append(global_conditions)
        sources.extend(["global"]*global_conditions.shape[0])
    if local_conditions.numel() > 0:
        condition_batches.append(local_conditions)
        sources.extend(["local"]*local_conditions.shape[0])
    
    if not condition_batches:
        return []
    
    conditions = torch.cat(condition_batches, dim=0)
    valid_indices = []
    for index, point in enumerate(conditions):
        if not torch.isfinite(point).all():
            continue
        if bool(((point < constraints.box_bounds[0]) | (point > constraints.box_bounds[1])).any()):
            continue
        if not bool(feasible_mask(point.unsqueeze(0), constraints)[0]):
            continue
        valid_indices.append(index)

    if not valid_indices:
        return []

    conditions = conditions[valid_indices]
    sources = [sources[index] for index in valid_indices]
    with torch.no_grad():
        acquisition_values = acq_function(conditions.unsqueeze(1)).reshape(-1)
    
    lower, upper = constraints.box_bounds
    scale = (upper - lower).clamp_min(1e-12)
    accepted_normalized = [
        (start.detach().reshape(-1) - lower)/scale
        for start in attempted_starts
    ]
    accepted_signatures = set(excluded_signatures)
    ranked_pool = []
    
    for index_tensor in torch.argsort(acquisition_values, descending=True):
        index = int(index_tensor.item())
        value = acquisition_values[index]
        if not torch.isfinite(value):
            continue
        
        a = conditions[index].detach().reshape(-1)
        signature = geometry_signature(a, constraints)
        if signature in accepted_signatures:
            continue
        
        normalized = (a - lower)/scale
        if accepted_normalized:
            distances = torch.stack([
                torch.linalg.vector_norm(normalized - previous)
                for previous in accepted_normalized
            ])
            if bool((distances < minimum_distance).any()):
                continue
        
        ranked_pool.append({
            "point": a,
            "source": sources[index],
            "raw_acquisition_value": value.detach(),
            "signature": signature,
        })
        
        accepted_normalized.append(normalized)
        accepted_signatures.add(signature)
    
    return ranked_pool

def take_restart_batch(ranked_pool, number_global, number_local, nholes, diversity=False):
    """Pop a source-balanced batch, optionally seeding every shape multiset."""
    selected_indices = []
    source_targets = (
        ("global", number_global),
        ("local", number_local),
    )
    target_total = min(number_global + number_local, len(ranked_pool))

    if diversity:
        selected_shapes = set()
        for index, entry in enumerate(ranked_pool):
            point = entry["point"]
            shape_key = tuple(int(point[7*h].item()) for h in range(nholes))
            if shape_key in selected_shapes:
                continue
            
            selected_indices.append(index)
            selected_shapes.add(shape_key)
            if len(selected_indices) >= target_total:
                break

    # Honour the requested source balance after the diversity seeds.
    for source, target in source_targets:
        already_selected = sum(
            ranked_pool[index]["source"] == source for index in selected_indices        )
        shortage = max(0, target - already_selected)
        source_indices = [index for index, entry in enumerate(ranked_pool) if entry["source"] == source and index not in selected_indices]
        available_slots = target_total - len(selected_indices)
        selected_indices.extend(source_indices[:min(shortage, available_slots)])

    if len(selected_indices) < target_total:
        selected_indices.extend([index for index in range(len(ranked_pool)) if index not in selected_indices][:target_total - len(selected_indices)])

    selected_index_set = set(selected_indices[:target_total])
    selected = [entry for index, entry in enumerate(ranked_pool) if index in selected_index_set]
    remaining = [entry for index, entry in enumerate(ranked_pool) if index not in selected_index_set]
    
    return selected, remaining
