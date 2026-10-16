"""End-to-end Bayesian optimization driver for one-shape perforations."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement

from .bayesian import Acquisition, Bounds
from .benchmark import audit_reference_image
from .gaussian import (
    Pred_Objective_Multi_Task,
    compliance_to_gp_response,
    warm_start_metadata,
)
from .physical import (
    analytical_solid_fraction,
    evaluate_designs,
    generate_geometry,
    geometry_signature,
    rasterized_volume_feasibility,
)
from .runtime import TENSOR_KWARGS, make_generator
from .records import RunRecorder, capture_rng_state
from .sampling import (
    AREA_BROAD,
    AREA_NEAR_EQUAL,
    feasible_mask,
    model_kernel_distance,
    rank_initial_condition_pool,
    sample_local_feasible,
    sample_stratified_feasible,
    select_archive_local_conditions,
    take_restart_batch,
)


@dataclass
class OptimizationConfig:
    xsize: int = 100
    ysize: int = 50
    bx: float = 1.0
    by: float = 1.0
    amin: float = 50.0
    smin: float = 1.0
    solid_max: float = 0.5
    nholes: int = 3
    htype: int = 0
    tri_q_min: float = 0.10
    raw_pool_multiplier: int = 3
    ng_rst: int = 10
    nl_rst: int = 5
    max_restart_batches: int = 6
    minimum_restart_distance: float = 0.02
    local_radius: float = 0.05
    local_start_strategy: str = "radius"
    acquisition_weight: float = 0.55
    novelty_weight: float = 0.45
    raster_tolerance: float = 0.005
    emin: float = 1e-9
    e0: float = 1.0
    penal: float = 3.0
    observation_noise_variance: float | None = None
    warm_start_policy: str | None = None
    health_nu_min: float = 0.01
    health_signal_variance_min: float | None = None
    health_mean_spread_min: float | None = None
    health_posterior_std_min: float | None = None
    health_predictive_z_max: float | None = None
    health_predictive_window: int = 5
    burn_in: int | None = None
    iterations: int | None = None
    stagnation_patience: int | None = None
    exploration_fraction: float = 0.75
    seed: int = 0
    output_dir: str | Path | None = None
    audit_reference: bool = True

    @property
    def dimension(self):
        return 7 * self.nholes

    @property
    def resolved_burn_in(self):
        return self.burn_in if self.burn_in is not None else 5 * self.dimension

    @property
    def resolved_iterations(self):
        return self.iterations if self.iterations is not None else 25 * self.dimension

    @property
    def resolved_stagnation_patience(self):
        if self.stagnation_patience is not None:
            return self.stagnation_patience
        return 5 * self.dimension

    @property
    def resolved_warm_start_policy(self):
        if self.warm_start_policy is not None:
            return self.warm_start_policy
        return "always"

    @property
    def resolved_restart_ranking_weights(self):
        total = self.acquisition_weight + self.novelty_weight
        return self.acquisition_weight / total, self.novelty_weight / total

class StagnationRestartController:
    """Shift a fixed restart budget toward global starts after stagnation."""

    def __init__(self, ng_rst, nl_rst, patience, exploration_fraction=0.75):
        self.baseline = (int(ng_rst), int(nl_rst))
        self.patience = int(patience)
        self.exploration_fraction = float(exploration_fraction)
        self.stagnation = 0

    def allocation(self):
        total = sum(self.baseline)
        if self.stagnation < self.patience or total == 0:
            return self.baseline
        global_count = min(total, max(1, math.ceil(total * self.exploration_fraction)))
        return global_count, total - global_count

    def observe(self, improved: bool):
        self.stagnation = 0 if improved else self.stagnation + 1


class GPWarmStartController:
    """Apply a health-triggered cold start to the next fit only."""

    def __init__(self):
        self.force_cold_next = False

    def model_for_fit(self, previous_model):
        if self.force_cold_next:
            self.force_cold_next = False
            return None
        return previous_model

    def observe_health(self, health):
        self.force_cold_next = bool(health["next_fit_forced_cold"])


class RunLog:
    def __init__(self, path: Path):
        self.stream = path.open("w", encoding="utf-8")

    def write(self, message=""):
        print(message)
        print(message, file=self.stream, flush=True)

    def close(self):
        self.stream.close()


def _output_directory(config: OptimizationConfig):
    if config.output_dir is not None:
        return Path(config.output_dir)
    shape = "Triangle" if config.htype == 0 else "Ellipse"
    return Path("outputs") / f"{datetime.now():%y%m%d_%H%M%S}_{shape}"


def _flatten_diagnostics(diagnostics):
    lengthscales = [
        value for values in diagnostics.lengthscales.values() for value in values
    ]
    outputscales = list(diagnostics.outputscales.values())
    return lengthscales, outputscales


def _validate_health_configuration(config):
    if config.observation_noise_variance is not None and (
        not math.isfinite(config.observation_noise_variance)
        or config.observation_noise_variance <= 0
    ):
        raise ValueError("observation_noise_variance must be finite and positive.")
    if config.warm_start_policy not in (None, "health_reset", "always"):
        raise ValueError("warm_start_policy must be 'health_reset', 'always', or None.")
    if not 0.0 <= config.health_nu_min <= 1.0:
        raise ValueError("health_nu_min must lie in [0, 1].")
    for name in (
        "health_signal_variance_min",
        "health_mean_spread_min",
        "health_posterior_std_min",
        "health_predictive_z_max",
    ):
        value = getattr(config, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative, or None.")
    if config.health_predictive_window < 1:
        raise ValueError("health_predictive_window must be at least 1.")


def _validate_restart_ranking_configuration(config):
    weights = (config.acquisition_weight, config.novelty_weight)
    if not all(math.isfinite(weight) and weight >= 0 for weight in weights):
        raise ValueError(
            "acquisition_weight and novelty_weight must be finite and non-negative."
        )
    if sum(weights) <= 0:
        raise ValueError(
            "acquisition_weight and novelty_weight must have a positive total."
        )


def _representative_feasible_points(global_conditions, local_conditions, dimension):
    pools = [pool for pool in (global_conditions, local_conditions) if len(pool)]
    if not pools:
        return torch.empty((0, dimension), **TENSOR_KWARGS)
    return torch.unique(torch.cat(pools), dim=0)


def _assess_gp_health(
    model,
    representative_points,
    config,
    predictive_z_history,
    *,
    withheld_predictive_mean=None,
    withheld_predictive_std=None,
    withheld_observed_response=None,
):
    diagnostics = asdict(model.health_diagnostics(representative_points))
    reasons = []
    core_metrics = (
        diagnostics["signal_variance"],
        diagnostics["observation_noise_variance"],
        diagnostics["signal_to_noise_ratio"],
        diagnostics["signal_fraction_nu"],
    )
    if not all(math.isfinite(value) for value in core_metrics):
        reasons.append("nonfinite_signal_or_noise_metric")
    elif model.noise_mode == "learned":
        if diagnostics["signal_fraction_nu"] < config.health_nu_min:
            reasons.append("signal_fraction_nu_below_threshold")
    elif (
        config.health_signal_variance_min is not None
        and diagnostics["signal_variance"] < config.health_signal_variance_min
    ):
        reasons.append("signal_variance_below_threshold")

    mean_spread = diagnostics["posterior_mean_spread_ratio"]
    if (
        config.health_mean_spread_min is not None
        and mean_spread is not None
        and mean_spread < config.health_mean_spread_min
    ):
        reasons.append("posterior_mean_spread_below_threshold")
    posterior_std = diagnostics["posterior_std_ratio"]
    if (
        config.health_posterior_std_min is not None
        and posterior_std is not None
        and posterior_std < config.health_posterior_std_min
    ):
        reasons.append("posterior_std_below_threshold")

    standardized_residual = None
    predictive_rms_z = None
    if (
        withheld_predictive_mean is not None
        and withheld_predictive_std is not None
        and withheld_observed_response is not None
        and withheld_predictive_std > 0
    ):
        standardized_residual = (
            withheld_observed_response - withheld_predictive_mean
        ) / withheld_predictive_std
        predictive_z_history.append(standardized_residual)
        window = predictive_z_history[-config.health_predictive_window :]
        predictive_rms_z = math.sqrt(sum(value**2 for value in window) / len(window))
        if (
            config.health_predictive_z_max is not None
            and len(window) == config.health_predictive_window
            and predictive_rms_z > config.health_predictive_z_max
        ):
            reasons.append("predictive_rms_z_above_threshold")

    warning = bool(reasons)
    if not warning:
        action = "none"
    elif config.resolved_warm_start_policy == "health_reset":
        action = "cold_fit_next_iteration"
    else:
        action = "warn_only"
    diagnostics.update(
        {
            "noise_mode": model.noise_mode,
            "prescribed_observation_noise_variance": (
                model.prescribed_observation_noise_variance
            ),
            "withheld_predictive_mean": withheld_predictive_mean,
            "withheld_predictive_std": withheld_predictive_std,
            "withheld_observed_response": withheld_observed_response,
            "withheld_standardized_residual": standardized_residual,
            "predictive_window_count": min(
                len(predictive_z_history), config.health_predictive_window
            ),
            "predictive_rms_z": predictive_rms_z,
            "health_warning_triggered": warning,
            "health_warning_reasons": reasons,
            "health_action": action,
            "next_fit_forced_cold": action == "cold_fit_next_iteration",
        }
    )
    return diagnostics


def _sample_restart_pool(
    config,
    constraints,
    x_train,
    y_train,
    global_count,
    local_count,
    generator,
    model=None,
):
    try:
        global_conditions = sample_stratified_feasible(
            global_count * config.raw_pool_multiplier,
            config.nholes,
            config.htype,
            constraints,
            config.amin,
            area_strategies=(AREA_BROAD,),
            allow_partial=True,
            generator=generator,
        )
    except RuntimeError:
        global_conditions = torch.empty((0, config.dimension), **TENSOR_KWARGS)
    requested_local = local_count * config.raw_pool_multiplier
    if config.local_start_strategy == "archive":
        local_conditions, local_metadata = select_archive_local_conditions(
            requested_local,
            x_train,
            y_train,
            constraints,
            model,
        )
    else:
        local_conditions = sample_local_feasible(
            requested_local,
            x_train,
            y_train,
            constraints,
            radius=config.local_radius,
            generator=generator,
        )
        local_metadata = [
            {"start_method": "radius"} for _ in range(len(local_conditions))
        ]
    return global_conditions, local_conditions, local_metadata


def _write_diagnostics(output_dir: Path, records):
    if not records:
        return
    path = output_dir / "iteration_diagnostics.csv"
    fields = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot_diagnostics(output_dir: Path, best_history, records):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(range(len(best_history)), best_history)
    axes[0, 0].set(title="Best compliance", xlabel="BO iteration")
    axes[0, 0].grid(alpha=0.3)
    if records:
        iterations = [record["iteration"] for record in records]
        axes[0, 1].plot(
            iterations,
            [record["posterior_score_std"] for record in records],
            color="tab:orange",
            label="selected candidate",
        )
        axes[0, 1].plot(
            iterations,
            [record["posterior_incumbent_score_std"] for record in records],
            label="incumbent at model fit",
        )
        axes[0, 1].plot(
            iterations,
            [record["posterior_fixed_burnin_score_std"] for record in records],
            label="fixed best burn-in",
        )
        axes[0, 1].set(
            title="Posterior uncertainty in -log(compliance) score",
            xlabel="BO iteration",
        )
        axes[0, 1].legend(fontsize=8)
        sources = [record["winner_source"] for record in records]
        source_codes = [0 if source == "local" else 1 for source in sources]
        axes[1, 0].scatter(iterations, source_codes, s=14)
        axes[1, 0].set(
            title="Winning restart source",
            xlabel="BO iteration",
            yticks=[0, 1],
            yticklabels=["local", "global"],
        )
        axes[1, 1].plot(
            iterations,
            [record["lengthscale_median"] for record in records],
            label="median lengthscale",
        )
        axes[1, 1].plot(
            iterations,
            [record["noise"] for record in records],
            label="noise",
        )
        axes[1, 1].plot(
            iterations,
            [record["outputscale_mean"] for record in records],
            label="mean outputscale",
        )
        axes[1, 1].set_yscale("log")
        axes[1, 1].set(title="GP diagnostics", xlabel="BO iteration")
        axes[1, 1].legend(fontsize=8)
    figure.savefig(output_dir / "Optimization_Diagnostics.png", dpi=300)
    plt.close(figure)


def run_optimization(config: OptimizationConfig | None = None):
    """Run constrained GP Bayesian optimization and persist a compact audit trail."""
    config = config or OptimizationConfig()
    if config.htype not in (0, 1):
        raise ValueError("htype must be 0 (triangles) or 1 (ellipses).")
    if config.local_start_strategy not in ("radius", "archive"):
        raise ValueError("local_start_strategy must be 'radius' or 'archive'.")
    _validate_health_configuration(config)
    _validate_restart_ranking_configuration(config)
    output_dir = _output_directory(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_config = asdict(config)
    serializable_config["output_dir"] = str(config.output_dir) if config.output_dir else None
    generator = make_generator(config.seed)
    recorder = RunRecorder(
        output_dir,
        serializable_config,
        {
            "dimension": config.dimension,
            "burn_in": config.resolved_burn_in,
            "iterations": config.resolved_iterations,
            "stagnation_patience": config.resolved_stagnation_patience,
            "noise_mode": (
                "learned"
                if config.observation_noise_variance is None
                else "prescribed"
            ),
            "warm_start_policy": config.resolved_warm_start_policy,
            "restart_ranking_weights": {
                "acquisition": config.resolved_restart_ranking_weights[0],
                "novelty": config.resolved_restart_ranking_weights[1],
            },
            "tensor_device": str(TENSOR_KWARGS["device"]),
            "tensor_dtype": str(TENSOR_KWARGS["dtype"]),
        },
        Path(__file__).resolve().parents[2],
    )
    recorder.record_initial_rng(capture_rng_state(generator))
    log = RunLog(output_dir / "output_log.txt")

    try:
        constraints = Bounds(
            config.xsize,
            config.ysize,
            config.nholes,
            config.bx,
            config.by,
            config.solid_max,
            config.smin,
            config.amin,
            config.raster_tolerance,
            tri_q_min=config.tri_q_min,
        )
        log.write("Bayesian optimization run")
        local_radius_label = (
            f"{config.local_radius:.4f}"
            if config.local_start_strategy == "radius"
            else "unused"
        )
        log.write(
            f"seed={config.seed} burn_in={config.resolved_burn_in} "
            f"iterations={config.resolved_iterations} restart_budget="
            f"{config.ng_rst + config.nl_rst} "
            f"local_start_strategy={config.local_start_strategy} "
            f"local_radius={local_radius_label} "
            f"noise_mode="
            f"{'learned' if config.observation_noise_variance is None else 'prescribed'} "
            f"warm_start_policy={config.resolved_warm_start_policy}"
        )
        x_train = sample_stratified_feasible(
            config.resolved_burn_in,
            config.nholes,
            config.htype,
            constraints,
            config.amin,
            area_strategies=(AREA_NEAR_EQUAL, AREA_BROAD),
            generator=generator,
        )
        y_train, raster_train = evaluate_designs(
            x_train,
            config.xsize,
            config.ysize,
            config.nholes,
            config.emin,
            config.e0,
            config.penal,
            constraints,
            return_solid_fractions=True,
        )
        burn_in_signatures = [
            geometry_signature(design, constraints) for design in x_train
        ]
        burn_in_analytical_feasible = feasible_mask(x_train, constraints)
        burn_in_raster_feasible = (
            raster_train <= constraints.solid_max + constraints.raster_tolerance
        )
        recorder.append_evaluations(
            x_train,
            y_train,
            compliance_to_gp_response(y_train),
            raster_train,
            burn_in_signatures,
            burn_in_analytical_feasible,
            burn_in_raster_feasible,
            phase="burn-in",
            sources="burn-in",
            burn_in_indices=list(range(1, len(x_train) + 1)),
        )
        recorder.record_burn_in_summary()
        initial_best = float(y_train.min().item())
        fixed_burnin_point = x_train[int(y_train.argmin().item())].detach().clone()
        best_history = [initial_best]
        observed_signatures = set(burn_in_signatures)
        evaluation_sources = ["burn-in"] * len(x_train)
        controller = StagnationRestartController(
            config.ng_rst,
            config.nl_rst,
            config.resolved_stagnation_patience,
            config.exploration_fraction,
        )
        previous_model = None
        diagnostics_records = []
        record_improvements = 0
        predictive_z_history = []
        warm_start_controller = GPWarmStartController()

        for iteration in range(1, config.resolved_iterations + 1):
            fit_rng_before = capture_rng_state(generator)
            fit_warnings = []
            fit_previous_model = warm_start_controller.model_for_fit(previous_model)
            initialization_metadata = warm_start_metadata(
                fit_previous_model,
                source_fit_index=(
                    None if fit_previous_model is None else iteration - 1
                ),
                inherit_likelihood=config.observation_noise_variance is None,
            )
            try:
                with warnings.catch_warnings(record=True) as fit_warnings:
                    warnings.simplefilter("always")
                    model = Pred_Objective_Multi_Task(
                        x_train,
                        y_train,
                        config.nholes,
                        previous_model=fit_previous_model,
                        observation_noise_variance=(
                            config.observation_noise_variance
                        ),
                    )
            except Exception as error:
                recorder.record_fit(
                    bo_iteration=iteration,
                    train_x=x_train,
                    train_compliance=y_train,
                    train_gp_response=compliance_to_gp_response(y_train),
                    rng_before=fit_rng_before,
                    rng_after=capture_rng_state(generator),
                    warnings_recorded=fit_warnings,
                    initialization_metadata=initialization_metadata,
                    noise_mode=(
                        "learned"
                        if config.observation_noise_variance is None
                        else "prescribed"
                    ),
                    prescribed_observation_noise_variance=(
                        config.observation_noise_variance
                    ),
                    failure={"type": type(error).__name__, "message": str(error)},
                )
                raise
            gp_diagnostics = model.diagnostics()
            current_fit_index = recorder.record_fit(
                bo_iteration=iteration,
                train_x=x_train,
                train_compliance=y_train,
                train_gp_response=model.train_score,
                rng_before=fit_rng_before,
                rng_after=capture_rng_state(generator),
                warnings_recorded=fit_warnings,
                model_state=model.persistence_state(),
                diagnostics=asdict(gp_diagnostics),
                transform_parameters=model.transform_parameters(),
                initialization_metadata=initialization_metadata,
                noise_mode=model.noise_mode,
                prescribed_observation_noise_variance=(
                    config.observation_noise_variance
                ),
            )
            previous_model = model
            acquisition_function = LogExpectedImprovement(
                model=model.gp, best_f=model.best_f
            )
            global_count, local_count = controller.allocation()
            attempted_starts = []
            rejected_signatures = set()
            optimizer_failures = 0
            optimizer_failure_records = []
            x_new = None
            winner = None

            global_conditions, local_conditions, local_metadata = _sample_restart_pool(
                config,
                constraints,
                x_train,
                y_train,
                global_count,
                local_count,
                generator,
                model=model,
            )
            representative_points = _representative_feasible_points(
                global_conditions, local_conditions, config.dimension
            )
            ranked_pool = rank_initial_condition_pool(
                acquisition_function,
                global_conditions,
                local_conditions,
                constraints,
                observed_signatures,
                attempted_starts,
                config.minimum_restart_distance,
                observed_points=x_train,
                local_metadata=local_metadata,
                acquisition_weight=config.acquisition_weight,
                novelty_weight=config.novelty_weight,
            )
            raw_global, raw_local = len(global_conditions), len(local_conditions)

            for _ in range(config.max_restart_batches):
                if not ranked_pool:
                    global_conditions, local_conditions, local_metadata = (
                        _sample_restart_pool(
                            config,
                            constraints,
                            x_train,
                            y_train,
                            global_count,
                            local_count,
                            generator,
                            model=model,
                        )
                    )
                    ranked_pool = rank_initial_condition_pool(
                        acquisition_function,
                        global_conditions,
                        local_conditions,
                        constraints,
                        observed_signatures | rejected_signatures,
                        attempted_starts,
                        config.minimum_restart_distance,
                        observed_points=x_train,
                        local_metadata=local_metadata,
                        acquisition_weight=config.acquisition_weight,
                        novelty_weight=config.novelty_weight,
                    )
                starts, ranked_pool = take_restart_batch(
                    ranked_pool,
                    global_count,
                    local_count,
                    config.nholes,
                    diversity=True,
                )
                if not starts:
                    continue
                attempted_starts.extend(entry["point"].detach().clone() for entry in starts)
                acquisition = Acquisition(
                    acquisition_function,
                    constraints,
                    starts,
                )
                optimizer_failures += len(acquisition.failed_restarts)
                optimizer_failure_records.extend(acquisition.failed_restarts)
                for result in acquisition.successful_results:
                    candidate = result["candidate"]
                    raster_valid, _ = rasterized_volume_feasibility(
                        candidate, constraints
                    )
                    signature = result["signature"]
                    if not raster_valid.all() or signature in observed_signatures:
                        rejected_signatures.add(signature)
                        continue
                    x_new, winner = candidate, result
                    break
                if x_new is not None:
                    break

            if x_new is None:
                health = _assess_gp_health(
                    model,
                    representative_points,
                    config,
                    predictive_z_history,
                )
                recorder.update_fit_health(current_fit_index, health)
                warm_start_controller.observe_health(health)
                if health["health_warning_triggered"]:
                    log.write(
                        f"GP HEALTH WARNING iter={iteration:04d} "
                        f"reasons={','.join(health['health_warning_reasons'])} "
                        f"action={health['health_action']}"
                    )
                recorder.record_iteration_event(
                    {
                        "bo_iteration": iteration,
                        "status": "stopped_no_novel_feasible_candidate",
                        "attempted_start_count": len(attempted_starts),
                        "optimizer_failures": optimizer_failure_records,
                    }
                )
                log.write(
                    f"iter={iteration:04d} stopped: no novel feasible candidate; "
                    f"starts={len(attempted_starts)} failures={optimizer_failures}"
                )
                break

            incumbent_point = x_train[int(y_train.argmin().item())]
            diagnostic_points = torch.cat(
                (
                    x_new,
                    incumbent_point.unsqueeze(0),
                    fixed_burnin_point.unsqueeze(0),
                )
            )
            with torch.no_grad():
                diagnostic_posterior = model.posterior(diagnostic_points)
                posterior_stds = (
                    diagnostic_posterior.variance.clamp_min(0).sqrt().reshape(-1)
                )
                withheld_predictive_mean = float(
                    diagnostic_posterior.mean.reshape(-1)[0].item()
                )
                withheld_predictive_std = math.sqrt(
                    float(posterior_stds[0].item()) ** 2
                    + model.observation_noise_variance_raw_response()
                )
            posterior_std = float(posterior_stds[0].item())
            posterior_incumbent_std = float(posterior_stds[1].item())
            posterior_fixed_burnin_std = float(posterior_stds[2].item())
            start_acquisition = winner.get("raw_acquisition_value")
            if start_acquisition is None:
                start_acquisition_value = math.nan
                acquisition_gain = math.nan
            else:
                start_acquisition_value = float(start_acquisition.item())
                acquisition_gain = (
                    float(winner["acquisition_value"].item())
                    - start_acquisition_value
                )
            winner_kernel_distance = model_kernel_distance(
                model,
                winner.get("start_point", x_new.reshape(-1)),
                x_new.reshape(-1),
            )
            y_new, raster_new = evaluate_designs(
                x_new,
                config.xsize,
                config.ysize,
                config.nholes,
                config.emin,
                config.e0,
                config.penal,
                constraints,
                return_solid_fractions=True,
            )
            gp_response_new = compliance_to_gp_response(y_new)
            new_analytical_feasible = feasible_mask(x_new, constraints)
            new_raster_feasible = (
                raster_new <= constraints.solid_max + constraints.raster_tolerance
            )
            recorder.append_evaluations(
                x_new,
                y_new,
                gp_response_new,
                raster_new,
                [winner["signature"]],
                new_analytical_feasible,
                new_raster_feasible,
                phase="bo",
                sources=[winner["source"]],
                bo_iterations=[iteration],
            )
            health = _assess_gp_health(
                model,
                representative_points,
                config,
                predictive_z_history,
                withheld_predictive_mean=withheld_predictive_mean,
                withheld_predictive_std=withheld_predictive_std,
                withheld_observed_response=float(gp_response_new.item()),
            )
            recorder.update_fit_health(current_fit_index, health)
            warm_start_controller.observe_health(health)
            if health["health_warning_triggered"]:
                log.write(
                    f"GP HEALTH WARNING iter={iteration:04d} "
                    f"reasons={','.join(health['health_warning_reasons'])} "
                    f"action={health['health_action']}"
                )
            recorder.record_iteration_event(
                {
                    "bo_iteration": iteration,
                    "status": "evaluated",
                    "winner_source": winner["source"],
                    "winner_start_method": winner.get("start_method", "global"),
                    "optimizer_failures": optimizer_failure_records,
                }
            )
            x_train = torch.cat((x_train, x_new))
            y_train = torch.cat((y_train, y_new))
            raster_train = torch.cat((raster_train, raster_new))
            observed_signatures.add(winner["signature"])
            evaluation_sources.append(winner["source"])

            previous_best = best_history[-1]
            current_best = float(y_train.min().item())
            improved = current_best < previous_best
            record_improvements += int(improved)
            best_history.append(current_best)
            controller.observe(improved)

            lengthscales, outputscales = _flatten_diagnostics(gp_diagnostics)
            record = {
                "iteration": iteration,
                "new_compliance": float(y_new.item()),
                "best_compliance": current_best,
                "raster_solid_fraction": float(raster_new.item()),
                "winner_source": winner["source"],
                "winner_acquisition": float(winner["acquisition_value"].item()),
                "winner_start_acquisition": start_acquisition_value,
                "winner_acquisition_gain": acquisition_gain,
                "winner_start_distance": winner.get("start_distance", math.nan),
                "winner_kernel_distance": winner_kernel_distance,
                "winner_start_method": winner.get("start_method", "global"),
                "winner_anchor_index": winner.get("anchor_index"),
                "winner_archive_rank": winner.get("archive_rank"),
                "posterior_score_std": posterior_std,
                "posterior_incumbent_score_std": posterior_incumbent_std,
                "posterior_fixed_burnin_score_std": posterior_fixed_burnin_std,
                "noise": gp_diagnostics.noise,
                "outputscale_mean": float(np.mean(outputscales)) if outputscales else math.nan,
                "lengthscale_min": min(lengthscales, default=math.nan),
                "lengthscale_median": float(np.median(lengthscales)) if lengthscales else math.nan,
                "lengthscale_max": max(lengthscales, default=math.nan),
                "lengthscales": json.dumps(gp_diagnostics.lengthscales),
                "outputscales": json.dumps(gp_diagnostics.outputscales),
                "global_restarts": global_count,
                "local_restarts": local_count,
                "raw_global": raw_global,
                "raw_local": raw_local,
                "requested_raw_local": local_count * config.raw_pool_multiplier,
                "local_start_strategy": config.local_start_strategy,
                "optimizer_failures": optimizer_failures,
                "stagnation": controller.stagnation,
            }
            diagnostics_records.append(record)
            log.write(
                f"iter={iteration:04d}/{config.resolved_iterations} "
                f"new={record['new_compliance']:.6f} best={current_best:.6f} "
                f"solid={record['raster_solid_fraction']:.4f} "
                f"winner={winner['source']} restarts={global_count}G/{local_count}L "
                f"move={record['winner_start_distance']:.3g} "
                f"acq_gain={record['winner_acquisition_gain']:.3g} "
                f"score_std={posterior_std:.3g} noise={gp_diagnostics.noise:.3g} "
                f"ls={record['lengthscale_min']:.3g}/"
                f"{record['lengthscale_median']:.3g}/"
                f"{record['lengthscale_max']:.3g}"
            )

        best_index = int(y_train.argmin().item())
        best_design = x_train[best_index]
        best_compliance = float(y_train[best_index].item())
        analytical_solid = float(
            analytical_solid_fraction(best_design.unsqueeze(0), constraints)[0].item()
        )
        raster_solid = float(raster_train[best_index].item())
        feasible = bool(
            feasible_mask(best_design.unsqueeze(0), constraints).all()
            and raster_solid <= constraints.solid_max + constraints.raster_tolerance
        )
        improvement = 100.0 * (initial_best - best_compliance) / max(
            abs(initial_best), 1e-12
        )
        best_source = evaluation_sources[best_index]
        _write_diagnostics(output_dir, diagnostics_records)
        _plot_diagnostics(output_dir, best_history, diagnostics_records)
        generate_geometry(
            config.xsize,
            config.ysize,
            best_design,
            config.nholes,
            plot=True,
            namesave="Best_Geometry",
            title=f"Best binary geometry (c={best_compliance:.4f}, solid={raster_solid:.4f})",
            fol_save=output_dir,
        )

        reference = Path(
            "outputs/Expected_Output_TRIANGLE/"
            "Three_Hole_Expected_Result_vf05_nx100_ny50_c96_Rasterised.png"
        )
        if config.audit_reference and config.htype == 0 and reference.exists():
            audit = audit_reference_image(
                reference,
                nelx=config.xsize,
                nely=config.ysize,
                solid_max=config.solid_max,
                raster_tolerance=config.raster_tolerance,
                emin=config.emin,
                e0=config.e0,
                penal=config.penal,
                report_path=output_dir / "benchmark_audit.json",
            )
            log.write(
                "benchmark audit: "
                f"binary_solid={audit.solid_fraction:.4f} "
                f"permitted={audit.permitted_solid_fraction:.4f} "
                f"binary_compliance={audit.recomputed_binary_compliance:.4f}; "
                "continuous logged compliance is not directly comparable"
            )

        log.write("Final Bayesian optimization report")
        log.write(f"best_compliance={best_compliance:.8f}")
        log.write(f"initial_best={initial_best:.8f} improvement={improvement:.3f}%")
        log.write(
            f"analytical_solid={analytical_solid:.6f} "
            f"raster_solid={raster_solid:.6f} feasible={feasible}"
        )
        log.write(
            f"best_source={best_source} evaluations={len(y_train)} "
            f"record_improvements={record_improvements}"
        )
        recorder.finalize(
            "completed",
            best_compliance=best_compliance,
            best_evaluation_index=best_index,
            best_source=best_source,
            feasible=feasible,
        )
        return {
            "best_design": best_design,
            "best_compliance": best_compliance,
            "best_source": best_source,
            "feasible": feasible,
            "output_dir": output_dir,
            "diagnostics": diagnostics_records,
        }
    except BaseException as error:
        recorder.finalize(
            "failed",
            failure={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        log.close()
