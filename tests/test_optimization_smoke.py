import json
from types import SimpleNamespace

import torch

from one_shape_bo import optimization
from one_shape_bo.gaussian import GPDiagnostics, GPHealthDiagnostics


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
            self.train_score = -torch.log(args[1])
            self.noise_mode = "learned"
            self.prescribed_observation_noise_variance = None

        def posterior(self, X):
            return SimpleNamespace(
                mean=torch.full((len(X), 1), -2.2, dtype=torch.double),
                variance=torch.full((len(X), 1), 0.04, dtype=torch.double),
            )

        def diagnostics(self):
            return GPDiagnostics({"kernel": [0.3]}, {"kernel": 1.2}, 0.01)

        def transform_parameters(self):
            return {"mean": [-2.0], "std": [0.2], "transform_class": "Dummy"}

        def persistence_state(self):
            return {"model_class": "Dummy", "model_state_dict": {}}

        def observation_noise_variance_raw_response(self):
            return 0.001

        def health_diagnostics(self, points):
            return GPHealthDiagnostics(
                signal_variance=1.2,
                observation_noise_variance=0.01,
                signal_to_noise_ratio=120.0,
                signal_fraction_nu=1.2 / 1.21,
                representative_point_count=len(points),
                posterior_mean_spread_ratio=0.4,
                posterior_std_ratio=0.2,
            )

    class DummyAcquisition:
        def __init__(self, *args, **kwargs):
            self.failed_restarts = []
            self.successful_results = [
                {
                    "candidate": candidate,
                    "signature": b"signature",
                    "source": "global",
                    "start_point": start,
                    "start_distance": 0.05,
                    "raw_acquisition_value": torch.tensor(1.0),
                    "acquisition_value": torch.tensor(1.5),
                }
            ]

    evaluation_count = 0
    ranking_kwargs = {}

    def fake_evaluate(X, *args, **kwargs):
        nonlocal evaluation_count
        evaluation_count += 1
        if evaluation_count == 1:
            return (
                torch.tensor([[10.0], [12.0]], dtype=torch.double),
                torch.tensor([0.5, 0.5], dtype=torch.double),
            )
        return (
            torch.tensor([[8.0]], dtype=torch.double),
            torch.tensor([0.5], dtype=torch.double),
        )

    def fake_rank(*args, **kwargs):
        ranking_kwargs.update(kwargs)
        return [{"point": start, "source": "global"}]

    monkeypatch.setattr(optimization, "Bounds", DummyBounds)
    monkeypatch.setattr(
        optimization, "sample_stratified_feasible", lambda *args, **kwargs: initial_x
    )
    monkeypatch.setattr(
        optimization,
        "_sample_restart_pool",
        lambda *args, **kwargs: (
            start.unsqueeze(0),
            torch.empty((0, 7)),
            [],
        ),
    )
    monkeypatch.setattr(
        optimization,
        "rank_initial_condition_pool",
        fake_rank,
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
        lambda design, constraints: bytes(
            [int(round(float(design[1].item()) * 100.0))]
        )
        * len(b"signature"),
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
            acquisition_weight=0.7,
            novelty_weight=0.3,
            audit_reference=False,
        )
    )

    diagnostics = (tmp_path / "iteration_diagnostics.csv").read_text()
    restart_attempts = (tmp_path / "restart_attempts.csv").read_text()
    output_log = (tmp_path / "output_log.txt").read_text()
    archive = torch.load(
        tmp_path / "evaluation_archive.pt", map_location="cpu", weights_only=False
    )
    assert result["best_source"] == "global"
    assert "posterior_score_std" in diagnostics
    assert "posterior_incumbent_score_std" in diagnostics
    assert "posterior_fixed_burnin_score_std" in diagnostics
    assert "winner_start_distance" in diagnostics
    assert "winner_start_method" in diagnostics
    assert "winner_acquisition_gain" in diagnostics
    assert "local_start_strategy" in diagnostics
    assert "global_restart_selection_strategy" in diagnostics
    assert "global_start_pairwise_distance_min" in diagnostics
    assert "actual_global_restart_attempts" in diagnostics
    assert "actual_local_restart_attempts" in diagnostics
    assert "adaptive_restart_allocation" in diagnostics
    assert "ranking_runtime_seconds" in diagnostics
    assert "optimized_acquisition_value" in restart_attempts
    assert "final_best_compliance" in restart_attempts
    assert "winner_source" in diagnostics
    assert "winner=global" in output_log
    assert "tensor([" not in output_log
    assert evaluation_count == 2
    assert ranking_kwargs["acquisition_weight"] == 0.7
    assert ranking_kwargs["novelty_weight"] == 0.3
    assert ranking_kwargs["number_global"] == 10
    assert ranking_kwargs["global_selection_strategy"] == (
        "greedy_dynamic_novelty"
    )
    assert archive["x"].shape == (3, 7)
    assert archive["burn_in_index"].tolist() == [1, 2, -1]
    assert archive["bo_iteration"].tolist() == [-1, -1, 1]
    assert (tmp_path / "gp_fits.json").exists()
    assert (tmp_path / "fit_states" / "fit_0001.pt").exists()
    fit_records = json.loads((tmp_path / "gp_fits.json").read_text())
    assert fit_records[0]["noise_mode"] == "learned"
    assert fit_records[0]["health_warning_triggered"] is False
