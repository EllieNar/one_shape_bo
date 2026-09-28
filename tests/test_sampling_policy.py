import torch

from one_shape_bo import sampling


def _entry(source, shape_tuple, value, score):
    blocks = []
    for shape in shape_tuple:
        blocks.extend([shape, value, value, value, value, value, 0.0])
    return {
        "point": torch.tensor(blocks, dtype=torch.double),
        "source": source,
        "shape_tuple": shape_tuple,
        "rank_score": torch.tensor(score, dtype=torch.double),
    }


def test_fixed_features_supports_mixed_shape_tuples():
    feature_maps = sampling.fixed_features(
        3,
        shape_tuples=[(0, 0, 1), (1, 0, 0), (1, 1, 0)],
    )

    assert len(feature_maps) == 2
    assert feature_maps[0][0] == 0.0
    assert feature_maps[0][7] == 0.0
    assert feature_maps[0][14] == 1.0
    assert feature_maps[0][20] == 0.0
    assert [feature_maps[1][index] for index in (0, 7, 14)] == [0.0, 1.0, 1.0]


def test_stratified_sampler_allocates_area_policy_quotas(monkeypatch):
    calls = []

    def fake_sample(number, constraints, amin, feature_map, **kwargs):
        calls.append((number, kwargs["area_strategy"], feature_map[0]))
        return torch.zeros((number, 7), dtype=torch.double)

    monkeypatch.setattr(sampling, "sample_random_feasible", fake_sample)
    constraints = type("Constraints", (), {"nholes": 1})()
    result = sampling.sample_stratified_feasible(
        5,
        1,
        0,
        constraints,
        1.0,
        area_strategies=(sampling.AREA_NEAR_EQUAL, sampling.AREA_BROAD),
    )

    assert result.shape == (5, 7)
    assert calls == [
        (3, sampling.AREA_NEAR_EQUAL, 0.0),
        (2, sampling.AREA_BROAD, 0.0),
    ]


def test_restart_batch_balances_sources_and_preserves_mixed_tuples():
    pool = [
        _entry("global", (0, 0, 1), 0.10, 0.9),
        _entry("global", (0, 1, 1), 0.85, 0.8),
        _entry("global", (0, 0, 1), 0.50, 0.7),
        _entry("local", (0, 0, 1), 0.20, 0.95),
        _entry("local", (0, 1, 1), 0.75, 0.6),
    ]

    selected, remaining = sampling.take_restart_batch(pool, 2, 2, 3, diversity=True)

    assert len(selected) == 4
    assert len(remaining) == 1
    assert sum(entry["source"] == "global" for entry in selected) == 2
    assert sum(entry["source"] == "local" for entry in selected) == 2
    assert {entry["shape_tuple"] for entry in selected} == {(0, 0, 1), (0, 1, 1)}
