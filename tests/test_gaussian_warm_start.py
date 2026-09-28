import torch

from one_shape_bo.gaussian import _warm_start


class DummyGP:
    def __init__(self, value):
        self.mean_module = torch.nn.Linear(1, 1)
        self.covar_module = torch.nn.Linear(1, 1)
        self.likelihood = torch.nn.Linear(1, 1)
        for module in (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ):
            torch.nn.init.constant_(module.weight, value)
            torch.nn.init.constant_(module.bias, value)


def test_warm_start_copies_gp_hyperparameter_modules():
    source = DummyGP(3.0)
    target = DummyGP(0.0)

    _warm_start(target, source)

    for module in (target.mean_module, target.covar_module, target.likelihood):
        assert torch.equal(module.weight, torch.full_like(module.weight, 3.0))
        assert torch.equal(module.bias, torch.full_like(module.bias, 3.0))
