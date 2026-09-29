import torch

from one_shape_bo import sampling


class SimpleConstraints:
    nholes = 1
    box_bounds = torch.stack(
        (torch.zeros(7, dtype=torch.double), torch.ones(7, dtype=torch.double))
    )
    nonlinear_constraints = []


class CandidateConstraints:
    nholes = 3
    xsize = 100.0
    ysize = 50.0
    bx = 1.0
    by = 1.0
    solid_max = 0.5
    amin = 50.0
    quality_min = 0.1
    smin = -1e6
    box = (bx, xsize - bx, by, ysize - by)


def test_random_candidate_applies_both_explicit_area_policies(monkeypatch):
    recorded = []

    def fake_shape(area, constraints, shape_type, *args, **kwargs):
        recorded.append(float(area))
        return torch.tensor(
            [0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.02], dtype=torch.double
        )

    monkeypatch.setattr(sampling, "random_shape_area", fake_shape)
    monkeypatch.setattr(
        sampling,
        "separating_axis_theorem_batch",
        lambda points_i, *args, **kwargs: torch.ones(
            len(points_i), dtype=torch.double
        ),
    )
    feature_map = {0: 0.0, 7: 0.0, 14: 0.0}
    constraints = CandidateConstraints()

    near = sampling.random_shape_candidate(
        feature_map,
        constraints,
        constraints.amin,
        anchor_trials=2,
        area_strategy=sampling.AREA_NEAR_EQUAL,
        generator=torch.Generator().manual_seed(5),
    )
    near_areas = recorded.copy()
    recorded.clear()
    broad = sampling.random_shape_candidate(
        feature_map,
        constraints,
        constraints.amin,
        anchor_trials=2,
        area_strategy=sampling.AREA_BROAD,
        generator=torch.Generator().manual_seed(5),
    )

    assert near is not None and broad is not None
    assert max(near_areas) / min(near_areas) < 1.2
    assert max(recorded) / min(recorded) > 1.2


def _mock_local_raster_and_signatures(monkeypatch):
    monkeypatch.setattr(
        sampling,
        "rasterized_volume_feasibility",
        lambda X, constraints: (
            torch.ones(len(X), dtype=torch.bool),
            torch.zeros(len(X), dtype=torch.double),
        ),
    )
    monkeypatch.setattr(
        sampling,
        "geometry_signature",
        lambda design, constraints: design.detach().cpu().numpy().tobytes(),
    )


def test_local_sampler_starts_at_configured_radius(monkeypatch):
    constraints = SimpleConstraints()
    base = torch.tensor(
        [[0.0, 0.25, 0.25, 0.55, 0.55, 0.60, 0.60]], dtype=torch.double
    )
    values = torch.tensor([[1.0]], dtype=torch.double)
    diagnostics = {}
    monkeypatch.setattr(
        sampling.torch,
        "randn",
        lambda shape, **kwargs: torch.ones(shape, dtype=torch.double),
    )
    _mock_local_raster_and_signatures(monkeypatch)

    result = sampling.sample_local_feasible(
        1,
        base,
        values,
        constraints,
        radius=0.05,
        max_attempts=10,
        diagnostics=diagnostics,
    )

    assert result.shape == (1, 7)
    assert torch.allclose(result[0, 1:], base[0, 1:] + 0.05)
    assert diagnostics["attempts"] == 1
    assert diagnostics["accepted"] == 1


def test_local_sampler_contracts_through_radius_ladder(monkeypatch):
    constraints = SimpleConstraints()
    base = torch.tensor(
        [[0.0, 0.25, 0.25, 0.55, 0.55, 0.60, 0.60]], dtype=torch.double
    )
    values = torch.tensor([[1.0]], dtype=torch.double)
    diagnostics = {}
    seen_radii = []
    monkeypatch.setattr(
        sampling.torch,
        "randn",
        lambda shape, **kwargs: torch.ones(shape, dtype=torch.double),
    )
    def accept_smallest_radius(X, constraints):
        seen_radii.append(float(X[0, 1] - base[0, 1]))
        return torch.tensor(
            [torch.allclose(X[0, 1:], base[0, 1:] + 0.00625)]
        )

    monkeypatch.setattr(sampling, "feasible_mask", accept_smallest_radius)
    _mock_local_raster_and_signatures(monkeypatch)

    result = sampling.sample_local_feasible(
        1,
        base,
        values,
        constraints,
        radius=0.05,
        max_attempts=7,
        diagnostics=diagnostics,
    )

    assert result.shape == (1, 7)
    assert torch.allclose(
        torch.tensor(seen_radii),
        torch.tensor([0.05, 0.05, 0.025, 0.025, 0.0125, 0.0125, 0.00625]),
    )
    assert diagnostics == {
        "requested": 1,
        "attempts": 7,
        "accepted": 1,
        "minimum_radius": 0.00625,
    }


def test_local_sampler_returns_partial_pool_at_attempt_limit(monkeypatch):
    constraints = SimpleConstraints()
    base = torch.tensor(
        [[0.0, 0.25, 0.25, 0.55, 0.55, 0.60, 0.60]], dtype=torch.double
    )
    values = torch.tensor([[1.0]], dtype=torch.double)
    diagnostics = {}
    calls = iter((1.0, -1.0))

    def alternating_noise(shape, **kwargs):
        return torch.full(shape, next(calls), dtype=torch.double)

    monkeypatch.setattr(sampling.torch, "randn", alternating_noise)
    _mock_local_raster_and_signatures(monkeypatch)

    result = sampling.sample_local_feasible(
        3,
        base,
        values,
        constraints,
        radius=0.05,
        max_attempts=2,
        diagnostics=diagnostics,
    )

    assert result.shape == (2, 7)
    assert diagnostics["requested"] == 3
    assert diagnostics["accepted"] == 2
    assert diagnostics["attempts"] == 2
    assert diagnostics["minimum_radius"] == 0.05


def test_rank_pool_keeps_sources_and_filters_near_observations(monkeypatch):
    constraints = SimpleConstraints()
    global_points = torch.tensor(
        [
            [0.0, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
            [0.0, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10],
        ],
        dtype=torch.double,
    )
    local_points = torch.tensor(
        [[0.0, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40]], dtype=torch.double
    )
    observed = torch.tensor(
        [[0.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]], dtype=torch.double
    )

    def acquisition(X):
        return X[..., 1:].sum(dim=-1)

    monkeypatch.setattr(
        sampling,
        "geometry_signature",
        lambda design, constraints: design.detach().cpu().numpy().tobytes(),
    )
    ranked = sampling.rank_initial_condition_pool(
        acquisition,
        global_points,
        local_points,
        constraints,
        excluded_signatures=set(),
        attempted_starts=[],
        minimum_distance=0.05,
        observed_points=observed,
    )

    assert {entry["source"] for entry in ranked} == {"global", "local"}
    assert all(not torch.equal(entry["point"], global_points[1]) for entry in ranked)
    assert all("rank_score" in entry for entry in ranked)
