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


def build_objective_gp(
    train_x,
    train_y,
    nholes: int,
    observation_noise_variance: float | None = None,
):
    """Construct an unfitted objective GP with the production model definition.

    This factory deliberately does not accept a previous model. Ordinary BO
    applies its established warm-start policy after construction, whereas
    offline numerical studies can apply independent controlled initial states.
    """
    train_score = compliance_to_gp_response(train_y)
    train_yvar = (
        None
        if observation_noise_variance is None
        else torch.full_like(train_score, float(observation_noise_variance))
    )
    categorical = list(range(0, 7 * nholes, 7))
    gp = MixedSingleTaskGP(
        train_X=train_x,
        train_Y=train_score,
        train_Yvar=train_yvar,
        cat_dims=categorical,
        cont_kernel_factory=matern_factory,
        outcome_transform=Standardize(m=1),
    )
    return gp, train_score


def fit_objective_gp(gp, **fit_options):
    """Fit a constructed objective GP using the production marginal likelihood."""
    marginal_likelihood = ExactMarginalLogLikelihood(
        likelihood=gp.likelihood, model=gp
    )
    return fit_gpytorch_mll(marginal_likelihood, **fit_options)


def _warm_start(target_gp, previous_model, *, copy_likelihood=True):
    """Reuse fitted hyperparameters, but not outcome-transform statistics."""
    if previous_model is None:
        return
    source_gp = getattr(previous_model, "gp", previous_model)
    module_names = ["mean_module", "covar_module"]
    if copy_likelihood:
        module_names.append("likelihood")
    for module_name in module_names:
        getattr(target_gp, module_name).load_state_dict(
            getattr(source_gp, module_name).state_dict()
        )


def _module_state_summary(module):
    return {
        name: value.detach().cpu().reshape(-1).tolist()
        for name, value in module.state_dict().items()
    }


def warm_start_metadata(
    previous_model, source_fit_index=None, *, inherit_likelihood=True
):
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
        "inherited_likelihood_noise_variance": (
            diagnostics.noise if inherit_likelihood else None
        ),
        "inherited_mean_module_state": _module_state_summary(source_gp.mean_module),
    }


@dataclass(frozen=True)
class GPDiagnostics:
    lengthscales: dict[str, list[float]]
    outputscales: dict[str, float]
    noise: float


@dataclass(frozen=True)
class GPHealthDiagnostics:
    signal_variance: float
    observation_noise_variance: float
    signal_to_noise_ratio: float
    signal_fraction_nu: float
    representative_point_count: int
    posterior_mean_spread_ratio: float | None
    posterior_std_ratio: float | None


class Pred_Objective_Multi_Task:
    """Mixed-variable GP over negative log compliance."""

    def __init__(
        self,
        train_x,
        train_y,
        nholes: int,
        previous_model=None,
        observation_noise_variance: float | None = None,
    ):
        self.gp, train_score = build_objective_gp(
            train_x,
            train_y,
            nholes,
            observation_noise_variance=observation_noise_variance,
        )
        self.train_score = train_score.detach().clone()
        self.prescribed_observation_noise_variance = observation_noise_variance
        _warm_start(
            self.gp,
            previous_model,
            copy_likelihood=observation_noise_variance is None,
        )
        self.best_f = train_score.max()
        fit_objective_gp(self.gp)

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
                self.gp.likelihood.noise.detach().reshape(-1).median().cpu().item()
            ),
        )

    @property
    def noise_mode(self):
        return (
            "learned"
            if self.prescribed_observation_noise_variance is None
            else "prescribed"
        )

    def observation_noise_variance_raw_response(self):
        """Return likelihood noise variance in pre-standardisation score units."""
        if self.prescribed_observation_noise_variance is not None:
            return float(self.prescribed_observation_noise_variance)
        standard_deviation = self.gp.outcome_transform.stdvs.detach().reshape(-1)[0]
        standardized_noise = self.gp.likelihood.noise.detach().reshape(-1)[0]
        return float((standardized_noise * standard_deviation**2).item())

    def health_diagnostics(self, representative_points) -> GPHealthDiagnostics:
        """Compute scale-consistent health metrics without changing GP state."""
        points = torch.as_tensor(
            representative_points,
            device=self.train_score.device,
            dtype=self.train_score.dtype,
        )
        if points.ndim == 1:
            points = points.unsqueeze(0)
        representative_count = len(points)
        kernel_points = points if representative_count else self.gp.train_inputs[0]
        with torch.no_grad():
            kernel_diagonal = self.gp.covar_module(kernel_points).diagonal(
                dim1=-2, dim2=-1
            )
            signal_variance = float(kernel_diagonal.reshape(-1).median().item())
            observation_noise = self.diagnostics().noise
            signal_to_noise = signal_variance / max(observation_noise, 1e-300)
            signal_fraction = signal_variance / max(
                signal_variance + observation_noise, 1e-300
            )
            mean_spread_ratio = None
            posterior_std_ratio = None
            if representative_count >= 2:
                posterior = self.posterior(points)
                response_scale = self.train_score.reshape(-1).std(correction=1)
                if torch.isfinite(response_scale) and response_scale > 0:
                    posterior_means = posterior.mean.reshape(-1)
                    posterior_stds = (
                        posterior.variance.clamp_min(0).sqrt().reshape(-1)
                    )
                    mean_spread_ratio = float(
                        (posterior_means.std(correction=1) / response_scale).item()
                    )
                    posterior_std_ratio = float(
                        (torch.quantile(posterior_stds, 0.9) / response_scale).item()
                    )
        return GPHealthDiagnostics(
            signal_variance=signal_variance,
            observation_noise_variance=observation_noise,
            signal_to_noise_ratio=signal_to_noise,
            signal_fraction_nu=signal_fraction,
            representative_point_count=representative_count,
            posterior_mean_spread_ratio=mean_spread_ratio,
            posterior_std_ratio=posterior_std_ratio,
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
