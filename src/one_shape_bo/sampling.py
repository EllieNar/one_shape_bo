"""Feasible global and local sampling for shaped-hole designs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

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
    interior_area = (constraints.xsize - 2 * constraints.bx) * (
        constraints.ysize - 2 * constraints.by
    )
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
):
    """Rank valid starts by acquisition and source-appropriate coverage."""
    batches, sources = [], []
    for source, conditions in (
        ("global", global_conditions),
        ("local", local_conditions),
    ):
        if conditions.numel():
            batches.append(conditions)
            sources.extend([source] * conditions.shape[0])
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
    with torch.no_grad():
        acquisition = acq_function(conditions.unsqueeze(1)).reshape(-1)

    lower, upper = constraints.box_bounds
    scale = (upper - lower).clamp_min(1e-12)
    reference = list(attempted_starts)
    if observed_points is not None:
        reference.extend(point for point in observed_points)
    reference_normalized = [
        (point.detach().reshape(-1) - lower) / scale for point in reference
    ]
    normalized = (conditions - lower) / scale
    if reference_normalized:
        references = torch.stack(reference_normalized)
        novelty = torch.cdist(normalized, references).min(dim=1).values
    else:
        novelty = torch.ones(len(conditions), **TENSOR_KWARGS)
    acquisition_score = _unit(acquisition)
    novelty_score = _unit(novelty)
    score = torch.stack(
        [
            0.55 * acquisition_score[index] + 0.45 * novelty_score[index]
            if source == "global"
            else acquisition_score[index]
            for index, source in enumerate(sources)
        ]
    )

    accepted_signatures = set(excluded_signatures)
    ranked = []
    for index_tensor in torch.argsort(score, descending=True):
        index = int(index_tensor.item())
        if not torch.isfinite(acquisition[index]):
            continue
        point = conditions[index].detach().reshape(-1)
        signature = geometry_signature(point, constraints)
        if signature in accepted_signatures:
            continue
        if reference_normalized and novelty[index] < minimum_distance:
            continue
        ranked.append(
            {
                "point": point,
                "source": sources[index],
                "shape_tuple": tuple(
                    int(point[7 * hole].item()) for hole in range(constraints.nholes)
                ),
                "raw_acquisition_value": acquisition[index].detach(),
                "rank_score": score[index].detach(),
                "signature": signature,
            }
        )
        accepted_signatures.add(signature)
    return ranked


def take_restart_batch(
    ranked_pool,
    number_global: int,
    number_local: int,
    nholes: int,
    diversity: bool = True,
):
    """Take a source-balanced, spatially diverse batch across shape tuples."""
    target_total = min(number_global + number_local, len(ranked_pool))
    if not target_total:
        return [], list(ranked_pool)
    source_remaining = {"global": number_global, "local": number_local}
    selected_indices, selected_points, represented_shapes = [], [], set()

    def distance(entry):
        if not selected_points:
            return float("inf")
        return min(
            float(torch.linalg.vector_norm(entry["point"] - point).item())
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
        selected_points.append(entry["point"])
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
