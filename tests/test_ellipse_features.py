import math

import pytest
import torch
from botorch.acquisition import LogExpectedImprovement

from one_shape_bo.ellipse_features import (
    design_features,
    shape_matrix_design_features,
)
from one_shape_bo.gaussian import Pred_Objective_Multi_Task, build_objective_gp
from one_shape_bo.optimization import OptimizationConfig, run_optimization
from one_shape_bo import sampling


def _ellipse(a, b, theta, *, xsize=100.0, ysize=50.0):
    return torch.tensor(
        [1.0, 0.4, 0.6, a / xsize, b / ysize, theta / math.pi, 0.0],
        dtype=torch.double,
    )


def test_shape_matrix_features_respect_ellipse_symmetries_and_circle_limit():
    original = _ellipse(20.0, 8.0, 0.3)
    half_turn = _ellipse(20.0, 8.0, 0.3 + math.pi)
    swapped = _ellipse(8.0, 20.0, 0.3 + math.pi / 2.0)
    circle_a = _ellipse(12.0, 12.0, 0.1)
    circle_b = _ellipse(12.0, 12.0, 1.2)

    features = shape_matrix_design_features(
        torch.stack((original, half_turn, swapped, circle_a, circle_b)),
        nholes=1,
        xsize=100.0,
        ysize=50.0,
    )

    assert torch.allclose(features[0], features[1], atol=1e-12, rtol=1e-12)
    assert torch.allclose(features[0], features[2], atol=1e-12, rtol=1e-12)
    assert torch.allclose(features[3], features[4], atol=1e-12, rtol=1e-12)


def test_shape_matrix_features_have_finite_gradients_at_circle():
    design = _ellipse(12.0, 12.0, 0.7).requires_grad_(True)

    transformed = shape_matrix_design_features(
        design, nholes=1, xsize=100.0, ysize=50.0
    )
    transformed.square().sum().backward()

    assert torch.isfinite(transformed).all()
    assert torch.isfinite(design.grad).all()


def test_shape_matrix_gp_and_log_ei_are_invariant_but_use_raw_candidates():
    train_x = torch.stack(
        (
            _ellipse(18.0, 7.0, 0.1),
            _ellipse(14.0, 9.0, 0.5),
            _ellipse(10.0, 6.0, 1.0),
        )
    )
    train_y = torch.tensor([[10.0], [12.0], [9.0]], dtype=torch.double)
    gp, train_score = build_objective_gp(
        train_x,
        train_y,
        nholes=1,
        ellipse_feature_mode="shape_matrix",
        xsize=100.0,
        ysize=50.0,
    )
    gp.eval()
    gp.likelihood.eval()
    original = _ellipse(20.0, 8.0, 0.3)
    equivalent = _ellipse(8.0, 20.0, 0.3 + math.pi / 2.0)

    posterior = gp.posterior(torch.stack((original, equivalent)))
    acquisition = LogExpectedImprovement(gp, best_f=train_score.max())
    values = acquisition(torch.stack((original, equivalent)).unsqueeze(1))
    circle = _ellipse(12.0, 12.0, 0.7).requires_grad_(True)
    circle_value = acquisition(circle.reshape(1, 1, -1))
    circle_value.backward()

    assert torch.allclose(posterior.mean[0], posterior.mean[1], atol=1e-12)
    assert torch.allclose(posterior.variance[0], posterior.variance[1], atol=1e-12)
    assert torch.allclose(values[0], values[1], atol=1e-12)
    assert torch.isfinite(circle.grad).all()


def test_invariant_fit_state_records_lossless_transformed_training_inputs(monkeypatch):
    train_x = torch.stack(
        (_ellipse(18.0, 7.0, 0.1), _ellipse(14.0, 9.0, 0.5))
    )
    train_y = torch.tensor([[10.0], [12.0]], dtype=torch.double)
    monkeypatch.setattr("one_shape_bo.gaussian.fit_objective_gp", lambda gp: None)

    model = Pred_Objective_Multi_Task(
        train_x,
        train_y,
        nholes=1,
        ellipse_feature_mode="shape_matrix",
        xsize=100.0,
        ysize=50.0,
    )
    stored = model.persistence_state()["input_features"]

    assert stored["mode"] == "shape_matrix"
    assert torch.equal(
        stored["transformed_train_x"],
        design_features(train_x, 1, 100.0, 50.0, mode="shape_matrix"),
    )


def test_global_novelty_uses_invariant_distance_for_equivalent_ellipses(monkeypatch):
    original = _ellipse(20.0, 8.0, 0.3)
    equivalent = _ellipse(8.0, 20.0, 0.3 + math.pi / 2.0)

    class Constraints:
        nholes = 1
        xsize = 100.0
        ysize = 50.0
        box_bounds = torch.stack(
            (torch.zeros(7, dtype=torch.double), torch.ones(7, dtype=torch.double))
        )

    class Acquisition:
        def __call__(self, points):
            return torch.ones(points.shape[0], dtype=points.dtype)

    monkeypatch.setattr(
        sampling,
        "feasible_mask",
        lambda points, constraints: torch.ones(len(points), dtype=torch.bool),
    )
    monkeypatch.setattr(
        sampling,
        "rasterized_volume_feasibility",
        lambda points, constraints: (
            torch.ones(len(points), dtype=torch.bool),
            torch.zeros(len(points), dtype=torch.double),
        ),
    )
    monkeypatch.setattr(
        sampling,
        "geometry_signature",
        lambda point, constraints: point.detach().cpu().numpy().tobytes(),
    )

    ranked = sampling.rank_initial_condition_pool(
        Acquisition(),
        equivalent.unsqueeze(0),
        torch.empty((0, 7), dtype=torch.double),
        Constraints(),
        excluded_signatures=set(),
        attempted_starts=[],
        minimum_distance=0.0,
        observed_points=original.unsqueeze(0),
        number_global=1,
        ellipse_feature_mode="shape_matrix",
    )

    assert len(ranked) == 1
    assert ranked[0]["raw_novelty_distance"].item() == pytest.approx(
        0.0, abs=1e-12
    )


def test_triangle_default_retains_raw_feature_path(tmp_path):
    config = OptimizationConfig(htype=0)
    triangle = torch.tensor(
        [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.double
    )
    gp, _ = build_objective_gp(
        triangle.repeat(2, 1),
        torch.tensor([[10.0], [11.0]], dtype=torch.double),
        nholes=1,
    )

    assert config.ellipse_feature_mode == "raw"
    assert not hasattr(gp, "input_transform")
    assert torch.equal(
        design_features(triangle, 1, 100.0, 50.0, mode="shape_matrix"),
        triangle,
    )

    with pytest.raises(ValueError, match="available only when htype=1"):
        run_optimization(
            OptimizationConfig(
                htype=0,
                ellipse_feature_mode="shape_matrix",
                burn_in=1,
                iterations=0,
                output_dir=tmp_path,
                audit_reference=False,
            )
        )
