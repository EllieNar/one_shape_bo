from types import SimpleNamespace

import torch

from one_shape_bo import optimization
from one_shape_bo.gaussian import GPDiagnostics


def test_one_iteration_writes_compact_source_and_gp_diagnostics(monkeypatch, tmp_path):
    initial_x = torch.tensor(
        [
            [0.0, 0.10, 0.10, 0.55, 0.50, 0.50, 0.55],
            [0.0, 0.20, 0.20, 0.60, 0.50, 0.50, 0.60],
        ],
        dtype=torch.double,
    )
    start = torch.tensor(
        [0.0, 0.30, 0.30, 0.60, 0.50, 0.50, 0.60], dtype=torch.double
    )
    candidate = torch.tensor(
        [[0.0, 0.35, 0.35, 0.60, 0.50, 0.50, 0.60]], dtype=torch.double
    )

    class DummyBounds:
        def __init__(self, *args, **kwargs):
            self.nholes = 1
            self.solid_max = 0.5
            self.raster_tolerance = 0.005
            self.box_bounds = torch.stack(
                (torch.zeros(7, dtype=torch.double), torch.ones(7, dtype=torch.double))
            )
            self.nonlinear_constraints = []

    class DummyModel:
        def __init__(self, *args, **kwargs):
            self.gp = object()
            self.best_f = torch.tensor(0.0)

        def posterior(self, X):
            return SimpleNamespace(variance=torch.tensor([[0.04]]))

        def diagnostics(self):
            return GPDiagnostics({"kernel": [0.3]}, {"kernel": 1.2}, 0.01)

    class DummyAcquisition:
        def __init__(self, *args, **kwargs):
            self.failed_restarts = []
            self.successful_results = [
                {
                    "candidate": candidate,
                    "signature": b"candidate",
                    "source": "global",
                    "acquisition_value": torch.tensor(1.5),
                }
            ]

    evaluation_count = 0

    def fake_evaluate(X, *args, **kwargs):
        nonlocal evaluation_count
        evaluation_count += 1
        if evaluation_count == 1:
            return (
                torch.tensor([[10.0], [12.0]], dtype=torch.double),
                torch.tensor([0.5, 0.5], dtype=torch.double),
            )
        return torch.tensor([[8.0]], dtype=torch.double), torch.tensor([0.5])

    monkeypatch.setattr(optimization, "Bounds", DummyBounds)
    monkeypatch.setattr(
        optimization, "sample_stratified_feasible", lambda *args, **kwargs: initial_x
    )
    monkeypatch.setattr(
        optimization,
        "_sample_restart_pool",
        lambda *args, **kwargs: (start.unsqueeze(0), torch.empty((0, 7))),
    )
    monkeypatch.setattr(
        optimization,
        "rank_initial_condition_pool",
        lambda *args, **kwargs: [{"point": start, "source": "global"}],
    )
    monkeypatch.setattr(
        optimization,
        "take_restart_batch",
        lambda pool, *args, **kwargs: (pool, []),
    )
    monkeypatch.setattr(optimization, "Pred_Objective_Multi_Task", DummyModel)
    monkeypatch.setattr(optimization, "LogExpectedImprovement", lambda **kwargs: object())
    monkeypatch.setattr(optimization, "Acquisition", DummyAcquisition)
    monkeypatch.setattr(optimization, "evaluate_designs", fake_evaluate)
    monkeypatch.setattr(
        optimization,
        "geometry_signature",
        lambda design, constraints: design.detach().cpu().numpy().tobytes(),
    )
    monkeypatch.setattr(
        optimization,
        "rasterized_volume_feasibility",
        lambda X, constraints: (torch.tensor([True]), torch.tensor([0.5])),
    )
    monkeypatch.setattr(
        optimization,
        "feasible_mask",
        lambda X, constraints: torch.ones(len(X), dtype=torch.bool),
    )
    monkeypatch.setattr(
        optimization,
        "analytical_solid_fraction",
        lambda X, constraints: [torch.tensor(0.5)],
    )
    monkeypatch.setattr(optimization, "generate_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimization, "_plot_diagnostics", lambda *args: None)

    result = optimization.run_optimization(
        optimization.OptimizationConfig(
            nholes=1,
            burn_in=2,
            iterations=1,
            output_dir=tmp_path,
            plot_every=0,
            audit_reference=False,
        )
    )

    diagnostics = (tmp_path / "iteration_diagnostics.csv").read_text()
    output_log = (tmp_path / "output_log.txt").read_text()
    assert result["best_source"] == "global"
    assert "posterior_score_std" in diagnostics
    assert "winner_source" in diagnostics
    assert "winner=global" in output_log
    assert "tensor([" not in output_log
