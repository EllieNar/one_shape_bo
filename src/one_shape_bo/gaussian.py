"""Gaussian-process surrogate used by Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


OBJECTIVE_CLAMP_MIN = 1e-12


def compliance_to_gp_response(compliance):
    """Return the exact response modelled by the GP."""
    return -torch.log(torch.as_tensor(compliance).clamp_min(OBJECTIVE_CLAMP_MIN))


def matern_factory(batch_shape, ard_num_dims, active_dims):
    return MaternKernel(
        nu=1.5,
        batch_shape=batch_shape,
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
        lengthscale_prior=GammaPrior(3.0, 6.0),
        lengthscale_constraint=Interval(0.05, 5.0),
    )


def _warm_start(target_gp, previous_model):
    """Reuse fitted hyperparameters, but not outcome-transform statistics."""
    if previous_model is None:
        return
    source_gp = getattr(previous_model, "gp", previous_model)
    for module_name in ("mean_module", "covar_module", "likelihood"):
        getattr(target_gp, module_name).load_state_dict(
            getattr(source_gp, module_name).state_dict()
        )


def _module_state_summary(module):
    return {
        name: value.detach().cpu().reshape(-1).tolist()
        for name, value in module.state_dict().items()
    }


def warm_start_metadata(previous_model, source_fit_index=None):
    """Describe the state supplied to a fit without changing that state."""
    if previous_model is None:
        return {
            "initialization_type": "cold",
            "warm_start_source_fit_index": None,
            "warm_start_state_reference": None,
            "inherited_lengthscales": None,
            "inherited_outputscales": None,
            "inherited_likelihood_noise_variance": None,
            "inherited_mean_module_state": None,
        }
    diagnostics = previous_model.diagnostics()
    source_gp = getattr(previous_model, "gp", previous_model)
    return {
        "initialization_type": "warm",
        "warm_start_source_fit_index": source_fit_index,
        "warm_start_state_reference": (
            None
            if source_fit_index is None
            else f"fit_states/fit_{source_fit_index:04d}.pt:model_state"
        ),
        "inherited_lengthscales": diagnostics.lengthscales,
        "inherited_outputscales": diagnostics.outputscales,
        "inherited_likelihood_noise_variance": diagnostics.noise,
        "inherited_mean_module_state": _module_state_summary(source_gp.mean_module),
    }


@dataclass(frozen=True)
class GPDiagnostics:
    lengthscales: dict[str, list[float]]
    outputscales: dict[str, float]
    noise: float


class Pred_Objective_Multi_Task:
    """Mixed-variable GP over negative log compliance."""

    def __init__(self, train_x, train_y, nholes: int, previous_model=None):
        train_score = compliance_to_gp_response(train_y)
        self.train_score = train_score.detach().clone()
        categorical = list(range(0, 7 * nholes, 7))
        self.gp = MixedSingleTaskGP(
            train_X=train_x,
            train_Y=train_score,
            cat_dims=categorical,
            cont_kernel_factory=matern_factory,
            outcome_transform=Standardize(m=1),
        )
        _warm_start(self.gp, previous_model)
        self.best_f = train_score.max()
        marginal_likelihood = ExactMarginalLogLikelihood(
            likelihood=self.gp.likelihood, model=self.gp
        )
        fit_gpytorch_mll(marginal_likelihood)

    def posterior(self, X, **kwargs):
        return self.gp.posterior(X, **kwargs)

    def diagnostics(self) -> GPDiagnostics:
        lengthscales, outputscales = {}, {}
        for name, module in self.gp.covar_module.named_modules():
            if hasattr(module, "lengthscale") and module.lengthscale is not None:
                lengthscales[name or "covar_module"] = (
                    module.lengthscale.detach().reshape(-1).cpu().tolist()
                )
            if hasattr(module, "outputscale"):
                outputscales[name or "covar_module"] = float(
                    module.outputscale.detach().reshape(-1)[0].cpu().item()
                )
        return GPDiagnostics(
            lengthscales=lengthscales,
            outputscales=outputscales,
            noise=float(
                self.gp.likelihood.noise.detach().reshape(-1)[0].cpu().item()
            ),
        )

    def transform_parameters(self):
        """Return the fitted outcome-standardisation parameters."""
        transform = self.gp.outcome_transform
        mean = getattr(transform, "means", None)
        std = getattr(transform, "stdvs", None)
        return {
            "mean": None if mean is None else mean.detach().cpu().reshape(-1).tolist(),
            "std": None if std is None else std.detach().cpu().reshape(-1).tolist(),
            "transform_class": type(transform).__name__,
        }

    def persistence_state(self):
        """Return fitted state needed to audit or restore this GP fit."""
        return {
            "model_class": type(self.gp).__name__,
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.gp.state_dict().items()
            },
        }
