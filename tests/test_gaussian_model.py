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
