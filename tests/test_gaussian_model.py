import torch

from one_shape_bo.gaussian import Pred_Objective_Multi_Task
from one_shape_bo.sampling import model_kernel_distance


def test_mixed_gp_fits_and_reports_diagnostics():
    generator = torch.Generator().manual_seed(4)
    train_x = torch.rand((6, 7), generator=generator, dtype=torch.double)
    train_x[:, 0] = 0.0
    train_y = torch.linspace(100.0, 150.0, 6, dtype=torch.double).unsqueeze(-1)

    model = Pred_Objective_Multi_Task(train_x, train_y, nholes=1)
    diagnostics = model.diagnostics()
    posterior = model.posterior(train_x[:1])
    distance = model_kernel_distance(model, train_x[0], train_x[1])

    assert diagnostics.lengthscales
    assert diagnostics.outputscales
    assert diagnostics.noise > 0.0
    assert torch.isfinite(posterior.variance).all()
    assert 0.0 <= distance <= 2.0


def test_prescribed_noise_and_health_metrics_are_finite():
    generator = torch.Generator().manual_seed(5)
    train_x = torch.rand((6, 7), generator=generator, dtype=torch.double)
    train_x[:, 0] = 0.0
    train_y = torch.linspace(90.0, 140.0, 6, dtype=torch.double).unsqueeze(-1)

    model = Pred_Objective_Multi_Task(
        train_x,
        train_y,
        nholes=1,
        observation_noise_variance=1e-6,
    )
    health = model.health_diagnostics(train_x[:3])

    assert model.noise_mode == "prescribed"
    assert model.observation_noise_variance_raw_response() == 1e-6
    assert health.signal_variance > 0.0
    assert health.observation_noise_variance > 0.0
    assert health.signal_to_noise_ratio > 0.0
    assert 0.0 <= health.signal_fraction_nu <= 1.0
    assert health.representative_point_count == 3
    assert health.posterior_mean_spread_ratio is not None
    assert health.posterior_std_ratio is not None

    next_x = torch.cat(
        (train_x, torch.rand((1, 7), generator=generator, dtype=torch.double))
    )
    next_x[-1, 0] = 0.0
    next_y = torch.cat((train_y, torch.tensor([[145.0]], dtype=torch.double)))
    warm_model = Pred_Objective_Multi_Task(
        next_x,
        next_y,
        nholes=1,
        previous_model=model,
        observation_noise_variance=1e-6,
    )
    assert warm_model.gp.likelihood.noise.numel() == len(next_x)
    assert warm_model.observation_noise_variance_raw_response() == 1e-6
