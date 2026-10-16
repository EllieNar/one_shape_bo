from types import SimpleNamespace

import torch

from one_shape_bo.gaussian import GPDiagnostics, _warm_start, warm_start_metadata


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


def test_warm_start_metadata_identifies_source_and_inherited_values():
    source = DummyGP(3.0)
    previous = SimpleNamespace(
        gp=source,
        diagnostics=lambda: GPDiagnostics(
            lengthscales={"kernel": [0.25]},
            outputscales={"kernel": 1.5},
            noise=0.02,
        ),
    )

    metadata = warm_start_metadata(previous, source_fit_index=2)

    assert metadata["initialization_type"] == "warm"
    assert metadata["warm_start_source_fit_index"] == 2
    assert metadata["warm_start_state_reference"] == (
        "fit_states/fit_0002.pt:model_state"
    )
    assert metadata["inherited_lengthscales"] == {"kernel": [0.25]}
    assert metadata["inherited_outputscales"] == {"kernel": 1.5}
    assert metadata["inherited_likelihood_noise_variance"] == 0.02
    assert metadata["inherited_mean_module_state"]["weight"] == [3.0]
