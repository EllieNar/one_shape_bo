from types import SimpleNamespace

import torch

from one_shape_bo import optimization
from one_shape_bo.optimization import StagnationRestartController


def test_radius_restart_strategy_remains_default():
    assert optimization.OptimizationConfig().local_start_strategy == "radius"


def test_stagnation_changes_split_without_changing_budget_and_resets():
    controller = StagnationRestartController(
        ng_rst=8,
        nl_rst=8,
        patience=3,
        exploration_fraction=0.75,
    )

    assert controller.allocation() == (8, 8)
    controller.observe(False)
    controller.observe(False)
    assert controller.allocation() == (8, 8)
    controller.observe(False)
    assert controller.allocation() == (12, 4)
    assert sum(controller.allocation()) == 16
    controller.observe(True)
    assert controller.allocation() == (8, 8)


def test_archive_restart_pool_requests_multiplier_times_local_budget(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        optimization,
        "sample_stratified_feasible",
        lambda number, *args, **kwargs: torch.zeros(
            (number, 7), dtype=torch.double
        ),
    )

    def fake_archive(number, x_train, y_train, constraints, model):
        captured.update(number=number, model=model)
        return (
            x_train[:number].clone(),
            [{"allow_observed_start": True} for _ in range(number)],
        )

    monkeypatch.setattr(
        optimization, "select_archive_local_conditions", fake_archive
    )
    config = optimization.OptimizationConfig(
        nholes=1,
        raw_pool_multiplier=3,
        local_start_strategy="archive",
    )
    x_train = torch.rand((8, 7), dtype=torch.double)
    y_train = torch.arange(8, dtype=torch.double).unsqueeze(-1)
    model = object()

    global_pool, local_pool, metadata = optimization._sample_restart_pool(
        config,
        SimpleNamespace(),
        x_train,
        y_train,
        global_count=2,
        local_count=2,
        generator=torch.Generator().manual_seed(3),
        model=model,
    )

    assert global_pool.shape == (6, 7)
    assert local_pool.shape == (6, 7)
    assert len(metadata) == 6
    assert captured == {"number": 6, "model": model}
