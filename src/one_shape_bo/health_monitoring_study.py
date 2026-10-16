"""Offline calibration of GP health thresholds from persistent BO archives.

The replay uses no FEM calls.  For a selected fixed standardized observation
noise it fits each archived training prefix in sequence, using production-style
warm starts and an independent cold reference, then evaluates the next recorded
response.  Recommendations remain provisional and are never written into the
runtime configuration automatically.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import statistics
import time
import warnings

import torch

from .gaussian import Pred_Objective_Multi_Task
from .numerical_noise_study import (
    decide_provisional_fixed_variance,
    load_archived_run,
    representative_probe_points,
    standardized_to_raw_noise_variance,
    validate_archived_runs,
)
from .records import (
    _git_metadata,
    atomic_csv_write,
    atomic_json_write,
    atomic_torch_save,
)
from .runtime import DEVICE


STUDY_SCHEMA_VERSION = 1
DEFAULT_WINDOWS = (3, 5, 10)
REPLAY_FIELDS = (
    "seed",
    "bo_iteration",
    "source_run_directory",
    "source_run_id",
    "selected_standardized_noise_variance",
    "prescribed_raw_response_noise_variance",
    "effective_standardized_noise_variance",
    "training_response_sample_variance",
    "train_count",
    "probe_count",
    "initialization_type",
    "warm_start_source_bo_iteration",
    "status",
    "fit_duration_seconds",
    "warning_count",
    "warnings",
    "failure_type",
    "failure_message",
    "signal_variance",
    "observation_noise_variance_standardized",
    "signal_to_noise_ratio",
    "signal_fraction_nu",
    "posterior_mean_spread_ratio",
    "posterior_std_ratio",
    "one_step_predictive_mean",
    "one_step_latent_std",
    "one_step_predictive_std",
    "one_step_observed_response",
    "one_step_standardized_residual",
    "one_step_absolute_error_response_sd",
    "one_step_negative_log_predictive_density",
    "cold_reference_status",
    "cold_reference_warning_count",
    "cold_reference_predictive_mean",
    "cold_reference_predictive_std",
    "warm_cold_mean_disagreement_response_sd",
    "warm_lengthscales",
    "warm_outputscales",
    "cold_lengthscales",
    "cold_outputscales",
    "state_file",
)


def _json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_value(value), separators=(",", ":"), sort_keys=True)
    return value


def _write_rows(rows, fields, path):
    atomic_csv_write(
        [{field: _csv_value(row.get(field)) for field in fields} for row in rows],
        fields,
        path,
    )


def _read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(row, field):
    value = row.get(field)
    if value in (None, "", "N/A"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _software_manifest():
    packages = {}
    for package in ("torch", "botorch", "gpytorch", "numpy", "scipy"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {"packages": packages, "device": str(DEVICE)}


def _resolve_selected_variance(numerical_study_directory, selected_variance):
    numerical_study_directory = Path(numerical_study_directory)
    if selected_variance is not None:
        value = float(selected_variance)
        if value <= 0:
            raise ValueError("selected_standardized_variance must be positive.")
        return value, "explicit_argument"
    manifest_path = numerical_study_directory / "study_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing numerical study manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(
            "The numerical-noise study must complete successfully before its "
            "automatic decision can drive health calibration."
        )
    decision_path = numerical_study_directory / "decision.json"
    if not decision_path.exists():
        decide_provisional_fixed_variance(numerical_study_directory)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    value = decision.get("selected_standardized_noise_variance")
    if decision.get("status") != "provisional_selection" or value is None:
        raise ValueError(
            "The numerical-noise study did not select a fixed variance. Review "
            "decision.json before running health calibration."
        )
    return float(value), str(decision_path.resolve())


def _load_source_runs(numerical_study_directory):
    config_path = Path(numerical_study_directory) / "study_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing numerical study config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_directories = config.get("run_directories")
    if not run_directories:
        raise ValueError("Numerical study config does not identify source runs.")
    runs = [
        load_archived_run(int(seed), path)
        for seed, path in run_directories.items()
    ]
    runs.sort(key=lambda item: item.seed)
    validate_archived_runs(runs)
    return runs


def _maximum_bo_iteration(run):
    values = run.archive["bo_iteration"].reshape(-1)
    values = values[values > 0]
    return 0 if not len(values) else int(values.max().item())


def _training_prefix_before(run, bo_iteration):
    bo_values = run.archive["bo_iteration"].reshape(-1)
    train_mask = (bo_values < 0) | (bo_values < int(bo_iteration))
    train_indices = torch.nonzero(train_mask, as_tuple=False).reshape(-1)
    if not len(train_indices):
        raise ValueError(f"Seed {run.seed} has no training data before BO fit.")
    stop = int(train_indices.max().item()) + 1
    if not bool(train_mask[:stop].all()) or bool(train_mask[stop:].any()):
        raise ValueError(f"Seed {run.seed} evaluation archive is not ordered.")
    target_indices = torch.nonzero(
        bo_values == int(bo_iteration), as_tuple=False
    ).reshape(-1)
    if len(target_indices) != 1:
        raise ValueError(
            f"Seed {run.seed} requires exactly one archived evaluation at BO "
            f"iteration {bo_iteration}; found {len(target_indices)}."
        )
    target_index = int(target_indices[0].item())
    return {
        "x": run.archive["x"][:stop].clone(),
        "compliance": run.archive["physical_compliance"][:stop].clone(),
        "response": run.archive["gp_response_prestandardization"][:stop].clone(),
        "target_x": run.archive["x"][target_index : target_index + 1].clone(),
        "target_response": run.archive["gp_response_prestandardization"][
            target_index : target_index + 1
        ].clone(),
    }


def _prediction(model, target_x, target_response, response_sd):
    with torch.no_grad():
        posterior = model.posterior(target_x)
        mean = posterior.mean.reshape(-1)[0]
        latent_variance = posterior.variance.clamp_min(0).reshape(-1)[0]
        raw_noise = model.observation_noise_variance_raw_response()
        predictive_variance = latent_variance + raw_noise
        predictive_std = predictive_variance.sqrt()
        residual = target_response.reshape(-1)[0] - mean
        nlpd = 0.5 * (
            torch.log(2 * torch.pi * predictive_variance)
            + residual.square() / predictive_variance
        )
    return {
        "mean": float(mean.item()),
        "latent_std": float(latent_variance.sqrt().item()),
        "predictive_std": float(predictive_std.item()),
        "observed": float(target_response.reshape(-1)[0].item()),
        "z": float((residual / predictive_std).item()),
        "absolute_error_response_sd": float((residual.abs() / response_sd).item()),
        "nlpd": float(nlpd.item()),
    }


def _fit_model(train_x, train_y, nholes, raw_noise, previous_model):
    caught = []
    started = time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = Pred_Objective_Multi_Task(
                train_x,
                train_y,
                nholes,
                previous_model=previous_model,
                observation_noise_variance=raw_noise,
            )
        return model, caught, time.monotonic() - started, None
    except Exception as error:
        return (
            None,
            caught,
            time.monotonic() - started,
            {"type": type(error).__name__, "message": str(error)},
        )


def run_health_monitoring_study(
    numerical_study_directory,
    output_directory,
    *,
    selected_standardized_variance=None,
    windows=DEFAULT_WINDOWS,
):
    """Replay all source archives and persist production-style health evidence."""
    numerical_study_directory = Path(numerical_study_directory).expanduser().resolve()
    selected_variance, selection_source = _resolve_selected_variance(
        numerical_study_directory, selected_standardized_variance
    )
    runs = _load_source_runs(numerical_study_directory)
    windows = tuple(sorted({int(value) for value in windows}))
    if not windows or any(value < 2 for value in windows):
        raise ValueError("Every predictive window must be an integer of at least two.")
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    state_directory = output_directory / "replay_states"
    state_directory.mkdir(exist_ok=True)
    probe = representative_probe_points(runs).to(device=DEVICE, dtype=torch.double)
    study_config = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "numerical_study_directory": str(numerical_study_directory),
        "selected_standardized_noise_variance": selected_variance,
        "selection_source": selection_source,
        "source_runs": {str(run.seed): str(run.directory) for run in runs},
        "predictive_windows": list(windows),
        "replay_initialization": "production_warm_with_independent_cold_reference",
        "fixed_noise_conversion": (
            "raw variance per fit = selected standardized variance * "
            "sample variance of that fit's -log(compliance) responses"
        ),
        "representative_probe": "deduplicated feasible source-archive designs",
    }
    config_path = output_directory / "study_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != study_config:
            raise ValueError(
                "Existing health study configuration differs; choose a new "
                "output directory."
            )
    else:
        atomic_json_write(study_config, config_path)
    manifest_path = output_directory / "study_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": STUDY_SCHEMA_VERSION,
            "study_id": f"health-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "software": _software_manifest(),
            "study_source": _git_metadata(Path(__file__).resolve().parents[2]),
        }
        atomic_json_write(manifest, manifest_path)
    records_path = output_directory / "gp_health_replay.csv"
    existing_rows = _read_rows(records_path)
    rows = list(existing_rows)
    for run in runs:
        maximum_iteration = _maximum_bo_iteration(run)
        seed_rows = [row for row in rows if int(row["seed"]) == run.seed]
        recorded_iterations = {
            int(row["bo_iteration"]) for row in seed_rows
        }
        if recorded_iterations == set(range(1, maximum_iteration + 1)):
            continue
        rows = [row for row in rows if int(row["seed"]) != run.seed]
        _write_rows(rows, REPLAY_FIELDS, records_path)
        previous_model = None
        previous_iteration = None
        for bo_iteration in range(1, maximum_iteration + 1):
            data = _training_prefix_before(run, bo_iteration)
            train_x = data["x"].to(device=DEVICE, dtype=torch.double)
            train_y = data["compliance"].to(device=DEVICE, dtype=torch.double)
            train_response = data["response"].to(device=DEVICE, dtype=torch.double)
            target_x = data["target_x"].to(device=DEVICE, dtype=torch.double)
            target_response = data["target_response"].to(
                device=DEVICE, dtype=torch.double
            )
            response_variance = torch.var(
                train_response.reshape(-1), correction=1
            )
            if not torch.isfinite(response_variance) or response_variance <= 0:
                raise ValueError(
                    f"Seed {run.seed}, BO {bo_iteration} has zero response variance."
                )
            response_sd = response_variance.sqrt()
            raw_noise = standardized_to_raw_noise_variance(
                selected_variance, train_response
            )
            model, fit_warnings, duration, failure = _fit_model(
                train_x,
                train_y,
                run.nholes,
                raw_noise,
                previous_model,
            )
            cold_model = model if previous_model is None else None
            cold_warnings = fit_warnings if previous_model is None else []
            cold_failure = failure if previous_model is None else None
            if previous_model is not None:
                cold_model, cold_warnings, _, cold_failure = _fit_model(
                    train_x, train_y, run.nholes, raw_noise, None
                )
            row = {
                "seed": run.seed,
                "bo_iteration": bo_iteration,
                "source_run_directory": str(run.directory),
                "source_run_id": run.manifest.get("run_id"),
                "selected_standardized_noise_variance": selected_variance,
                "prescribed_raw_response_noise_variance": raw_noise,
                "training_response_sample_variance": float(response_variance.item()),
                "train_count": len(train_x),
                "probe_count": len(probe),
                "initialization_type": (
                    "cold" if previous_model is None else "warm"
                ),
                "warm_start_source_bo_iteration": previous_iteration,
                "status": "failed" if failure else "success",
                "fit_duration_seconds": duration,
                "warning_count": len(fit_warnings),
                "warnings": [
                    {
                        "category": item.category.__name__,
                        "message": str(item.message),
                    }
                    for item in fit_warnings
                ],
                "failure_type": None if failure is None else failure["type"],
                "failure_message": None if failure is None else failure["message"],
                "cold_reference_status": (
                    "failed" if cold_failure else "success"
                ),
                "cold_reference_warning_count": len(cold_warnings),
            }
            state_name = f"seed_{run.seed}_bo_{bo_iteration:04d}.pt"
            if model is not None:
                health = model.health_diagnostics(probe)
                prediction = _prediction(
                    model, target_x, target_response, response_sd
                )
                diagnostics = model.diagnostics()
                row.update(
                    {
                        "effective_standardized_noise_variance": diagnostics.noise,
                        "signal_variance": health.signal_variance,
                        "observation_noise_variance_standardized": (
                            health.observation_noise_variance
                        ),
                        "signal_to_noise_ratio": health.signal_to_noise_ratio,
                        "signal_fraction_nu": health.signal_fraction_nu,
                        "posterior_mean_spread_ratio": (
                            health.posterior_mean_spread_ratio
                        ),
                        "posterior_std_ratio": health.posterior_std_ratio,
                        "one_step_predictive_mean": prediction["mean"],
                        "one_step_latent_std": prediction["latent_std"],
                        "one_step_predictive_std": prediction["predictive_std"],
                        "one_step_observed_response": prediction["observed"],
                        "one_step_standardized_residual": prediction["z"],
                        "one_step_absolute_error_response_sd": prediction[
                            "absolute_error_response_sd"
                        ],
                        "one_step_negative_log_predictive_density": prediction[
                            "nlpd"
                        ],
                        "warm_lengthscales": diagnostics.lengthscales,
                        "warm_outputscales": diagnostics.outputscales,
                        "state_file": f"replay_states/{state_name}",
                    }
                )
            if cold_model is not None:
                cold_prediction = _prediction(
                    cold_model, target_x, target_response, response_sd
                )
                cold_diagnostics = cold_model.diagnostics()
                row.update(
                    {
                        "cold_reference_predictive_mean": cold_prediction["mean"],
                        "cold_reference_predictive_std": cold_prediction[
                            "predictive_std"
                        ],
                        "cold_lengthscales": cold_diagnostics.lengthscales,
                        "cold_outputscales": cold_diagnostics.outputscales,
                    }
                )
                if model is not None:
                    row["warm_cold_mean_disagreement_response_sd"] = abs(
                        prediction["mean"] - cold_prediction["mean"]
                    ) / float(response_sd.item())
            if model is not None:
                atomic_torch_save(
                    {
                        "schema_version": STUDY_SCHEMA_VERSION,
                        "seed": run.seed,
                        "bo_iteration": bo_iteration,
                        "source_archive": str(run.directory),
                        "selected_standardized_noise_variance": selected_variance,
                        "prescribed_raw_response_noise_variance": raw_noise,
                        "initialization_type": row["initialization_type"],
                        "warm_model_state": model.persistence_state(),
                        "cold_reference_state": (
                            None
                            if cold_model is None
                            else cold_model.persistence_state()
                        ),
                    },
                    state_directory / state_name,
                )
                previous_model = model
                previous_iteration = bo_iteration
            else:
                previous_model = None
                previous_iteration = None
            rows.append(row)
            rows.sort(key=lambda item: (int(item["seed"]), int(item["bo_iteration"])))
            _write_rows(rows, REPLAY_FIELDS, records_path)
    failed = sum(row.get("status") != "success" for row in rows)
    manifest.update(
        {
            "status": "complete" if not failed else "complete_with_failures",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "replay_count": len(rows),
            "failed_replay_count": failed,
            "records": "gp_health_replay.csv",
        }
    )
    atomic_json_write(manifest, manifest_path)
    return rows


def _rolling_rms_by_seed(rows, window):
    result = {}
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    for seed, seed_rows in grouped.items():
        seed_rows.sort(key=lambda item: int(item["bo_iteration"]))
        history = []
        for row in seed_rows:
            z = _number(row, "one_step_standardized_residual")
            history.append(z)
            key = (seed, int(row["bo_iteration"]))
            recent = history[-window:]
            if len(recent) == window and all(value is not None for value in recent):
                result[key] = math.sqrt(
                    statistics.fmean(value**2 for value in recent)
                )
            else:
                result[key] = None
    return result


def _rates(predictions, labels):
    positives = sum(labels)
    negatives = len(labels) - positives
    true_positive = sum(
        prediction and label for prediction, label in zip(predictions, labels)
    )
    false_positive = sum(
        prediction and not label for prediction, label in zip(predictions, labels)
    )
    return {
        "true_positive_rate": None if not positives else true_positive / positives,
        "false_positive_rate": None if not negatives else false_positive / negatives,
        "positive_count": positives,
        "negative_count": negatives,
    }


def _select_threshold(rows, field, direction, maximum_false_positive_rate):
    usable = [
        row
        for row in rows
        if row.get("adverse") is not None and row.get(field) is not None
    ]
    labels = [bool(row["adverse"]) for row in usable]
    if not usable or not any(labels) or all(labels):
        return None
    values = sorted({float(row[field]) for row in usable})
    candidates = values + [math.nextafter(values[-1], math.inf)]
    best = None
    for threshold in candidates:
        predictions = [
            float(row[field]) < threshold
            if direction == "lower"
            else float(row[field]) > threshold
            for row in usable
        ]
        rates = _rates(predictions, labels)
        if rates["false_positive_rate"] > maximum_false_positive_rate:
            continue
        score = rates["true_positive_rate"] - rates["false_positive_rate"]
        candidate = {
            "threshold": threshold,
            "score": score,
            **rates,
        }
        if best is None or (score, rates["true_positive_rate"]) > (
            best["score"],
            best["true_positive_rate"],
        ):
            best = candidate
    if best is None or best["score"] <= 0:
        return None
    return best


def _cross_seed_threshold(
    rows,
    field,
    direction,
    maximum_false_positive_rate,
):
    seeds = sorted({int(row["seed"]) for row in rows})
    folds = []
    thresholds = []
    for held_out_seed in seeds:
        training = [row for row in rows if int(row["seed"]) != held_out_seed]
        validation = [row for row in rows if int(row["seed"]) == held_out_seed]
        selected = _select_threshold(
            training, field, direction, maximum_false_positive_rate
        )
        fold = {
            "metric": field,
            "direction": direction,
            "held_out_seed": held_out_seed,
            "selected_threshold": None if selected is None else selected["threshold"],
            "training_score": None if selected is None else selected["score"],
            "training_true_positive_rate": (
                None if selected is None else selected["true_positive_rate"]
            ),
            "training_false_positive_rate": (
                None if selected is None else selected["false_positive_rate"]
            ),
        }
        if selected is not None:
            threshold = selected["threshold"]
            usable = [
                row
                for row in validation
                if row.get("adverse") is not None and row.get(field) is not None
            ]
            predictions = [
                float(row[field]) < threshold
                if direction == "lower"
                else float(row[field]) > threshold
                for row in usable
            ]
            rates = _rates(
                predictions, [bool(row["adverse"]) for row in usable]
            )
            fold.update(
                {
                    "validation_true_positive_rate": rates["true_positive_rate"],
                    "validation_false_positive_rate": rates["false_positive_rate"],
                    "validation_positive_count": rates["positive_count"],
                    "validation_negative_count": rates["negative_count"],
                }
            )
            if (
                rates["true_positive_rate"] is not None
                and rates["false_positive_rate"] is not None
            ):
                fold["validation_score"] = (
                    rates["true_positive_rate"] - rates["false_positive_rate"]
                )
                thresholds.append(threshold)
        folds.append(fold)
    validation_scores = [
        fold.get("validation_score")
        for fold in folds
        if fold.get("validation_score") is not None
    ]
    recommendation = None
    if thresholds and validation_scores and statistics.fmean(validation_scores) > 0:
        recommendation = statistics.median(thresholds)
    return recommendation, folds


def calibrate_health_thresholds(
    health_study_directory,
    *,
    adverse_z_threshold=2.576,
    maximum_false_positive_rate=0.10,
):
    """Cross-validate provisional health thresholds across source-run seeds."""
    if adverse_z_threshold <= 0:
        raise ValueError("adverse_z_threshold must be positive.")
    if not 0 < maximum_false_positive_rate < 1:
        raise ValueError("maximum_false_positive_rate must lie between zero and one.")
    health_study_directory = Path(health_study_directory)
    config = json.loads(
        (health_study_directory / "study_config.json").read_text(encoding="utf-8")
    )
    rows = _read_rows(health_study_directory / "gp_health_replay.csv")
    if not rows:
        raise FileNotFoundError("No health replay records were found.")
    windows = tuple(int(value) for value in config["predictive_windows"])
    derived = []
    rolling = {window: _rolling_rms_by_seed(rows, window) for window in windows}
    for row in rows:
        seed = int(row["seed"])
        iteration = int(row["bo_iteration"])
        item = dict(row)
        failed = row["status"] != "success"
        warning = (_number(row, "warning_count") or 0) > 0
        z = _number(row, "one_step_standardized_residual")
        item["adverse"] = failed or warning or (
            z is not None and abs(z) > float(adverse_z_threshold)
        )
        for window in windows:
            item[f"rolling_rms_z_window_{window}"] = rolling[window][
                (seed, iteration)
            ]
        derived.append(item)
    metric_specs = {
        "signal_variance": "lower",
        "posterior_mean_spread_ratio": "lower",
        "posterior_std_ratio": "lower",
    }
    recommendations = {}
    cross_validation = []
    for metric, direction in metric_specs.items():
        recommendation, folds = _cross_seed_threshold(
            derived, metric, direction, maximum_false_positive_rate
        )
        recommendations[metric] = recommendation
        cross_validation.extend(folds)
    window_results = []
    for window in windows:
        field = f"rolling_rms_z_window_{window}"
        recommendation, folds = _cross_seed_threshold(
            derived, field, "upper", maximum_false_positive_rate
        )
        valid_scores = [
            fold.get("validation_score")
            for fold in folds
            if fold.get("validation_score") is not None
        ]
        window_results.append(
            {
                "window": window,
                "threshold": recommendation,
                "mean_validation_score": (
                    None if not valid_scores else statistics.fmean(valid_scores)
                ),
            }
        )
        cross_validation.extend(folds)
    eligible_windows = [
        result
        for result in window_results
        if result["threshold"] is not None
        and result["mean_validation_score"] is not None
        and result["mean_validation_score"] > 0
    ]
    selected_window = (
        None
        if not eligible_windows
        else max(
            eligible_windows,
            key=lambda item: (item["mean_validation_score"], -item["window"]),
        )
    )
    fields = list(REPLAY_FIELDS) + ["adverse"] + [
        f"rolling_rms_z_window_{window}" for window in windows
    ]
    _write_rows(
        derived,
        fields,
        health_study_directory / "gp_health_threshold_study.csv",
    )
    _write_rows(
        cross_validation,
        tuple(
            dict.fromkeys(key for row in cross_validation for key in row)
        ),
        health_study_directory / "cross_seed_validation.csv",
    )
    _write_rows(
        window_results,
        ("window", "threshold", "mean_validation_score"),
        health_study_directory / "predictive_window_comparison.csv",
    )
    recommendation = {
        "status": (
            "provisional"
            if any(value is not None for value in recommendations.values())
            or selected_window is not None
            else "insufficient_evidence"
        ),
        "noise_mode": "prescribed",
        "selected_standardized_observation_noise_variance": config[
            "selected_standardized_noise_variance"
        ],
        "health_nu_min": None,
        "health_nu_min_not_applicable_reason": (
            "Current runtime uses signal variance, not nu, for prescribed noise."
        ),
        "health_signal_variance_min": recommendations["signal_variance"],
        "health_mean_spread_min": recommendations[
            "posterior_mean_spread_ratio"
        ],
        "health_posterior_std_min": recommendations["posterior_std_ratio"],
        "health_predictive_z_max": (
            None if selected_window is None else selected_window["threshold"]
        ),
        "health_predictive_window": (
            None if selected_window is None else selected_window["window"]
        ),
        "calibration": {
            "adverse_definition": (
                "fit failure OR fitting warning OR absolute one-step "
                f"standardized residual > {adverse_z_threshold}"
            ),
            "maximum_cross_validated_false_positive_rate": (
                maximum_false_positive_rate
            ),
            "validation": "leave-one-seed-out",
            "representative_probe": config["representative_probe"],
        },
        "automatic_runtime_default_change": False,
        "limitations": [
            (
                "Probe points are archived feasible designs, not the exact "
                "runtime restart pools."
            ),
            (
                "The selected variance is held fixed on the standardized scale "
                "by converting it separately at each fit."
            ),
            (
                "Recommendations require engineering review before becoming "
                "package defaults."
            ),
        ],
    }
    recommendation["optimization_config_overrides"] = {
        key: value
        for key, value in {
            "health_signal_variance_min": recommendation[
                "health_signal_variance_min"
            ],
            "health_mean_spread_min": recommendation["health_mean_spread_min"],
            "health_posterior_std_min": recommendation[
                "health_posterior_std_min"
            ],
            "health_predictive_z_max": recommendation["health_predictive_z_max"],
            "health_predictive_window": recommendation[
                "health_predictive_window"
            ],
        }.items()
        if value is not None
    }
    atomic_json_write(
        recommendation,
        health_study_directory / "recommended_health_config.json",
    )
    return recommendation


def summarize_health_monitoring_study(
    health_study_directory, *, make_plots=True
):
    """Write a compact per-seed replay summary."""
    health_study_directory = Path(health_study_directory)
    rows = _read_rows(health_study_directory / "gp_health_replay.csv")
    if not rows:
        raise FileNotFoundError("No health replay records were found.")
    summaries = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        group = [row for row in rows if int(row["seed"]) == seed]
        z_values = [
            abs(value)
            for row in group
            if (value := _number(row, "one_step_standardized_residual"))
            is not None
        ]
        disagreements = [
            value
            for row in group
            if (
                value := _number(
                    row, "warm_cold_mean_disagreement_response_sd"
                )
            )
            is not None
        ]
        summaries.append(
            {
                "seed": seed,
                "fit_count": len(group),
                "failed_fit_count": sum(row["status"] != "success" for row in group),
                "warning_fit_count": sum(
                    (_number(row, "warning_count") or 0) > 0 for row in group
                ),
                "median_absolute_one_step_z": (
                    None if not z_values else statistics.median(z_values)
                ),
                "maximum_absolute_one_step_z": (
                    None if not z_values else max(z_values)
                ),
                "median_warm_cold_mean_disagreement_response_sd": (
                    None if not disagreements else statistics.median(disagreements)
                ),
            }
        )
    _write_rows(
        summaries,
        tuple(dict.fromkeys(key for row in summaries for key in row)),
        health_study_directory / "summary_by_seed.csv",
    )
    if make_plots:
        try:
            import matplotlib.pyplot as plt

            metrics = (
                ("signal_variance", "Signal variance"),
                ("posterior_mean_spread_ratio", "Posterior mean spread ratio"),
                ("posterior_std_ratio", "Posterior SD ratio"),
                (
                    "warm_cold_mean_disagreement_response_sd",
                    "Warm/cold mean disagreement",
                ),
            )
            figure, axes = plt.subplots(2, 2, figsize=(10, 8))
            for axis, (field, label) in zip(axes.reshape(-1), metrics):
                points = [
                    (
                        _number(row, field),
                        _number(row, "one_step_standardized_residual"),
                    )
                    for row in rows
                ]
                points = [
                    (x_value, abs(z_value))
                    for x_value, z_value in points
                    if x_value is not None and z_value is not None
                ]
                if points:
                    axis.scatter(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        s=12,
                        alpha=0.65,
                    )
                axis.set_xlabel(label)
                axis.set_ylabel("Absolute one-step standardized residual")
            figure.tight_layout()
            figure.savefig(
                health_study_directory / "health_metrics_vs_prediction.png",
                dpi=180,
            )
            plt.close(figure)
        except (ImportError, ValueError):
            pass
    return summaries


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="replay archived BO fits")
    run.add_argument("numerical_study_directory")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--selected-standardized-variance", type=float)
    run.add_argument("--window", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    summarize = subparsers.add_parser("summarize", help="summarize replay fits")
    summarize.add_argument("health_study_directory")
    summarize.add_argument("--no-plots", action="store_true")
    calibrate = subparsers.add_parser("calibrate", help="recommend thresholds")
    calibrate.add_argument("health_study_directory")
    calibrate.add_argument("--adverse-z-threshold", type=float, default=2.576)
    calibrate.add_argument("--maximum-false-positive-rate", type=float, default=0.10)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        rows = run_health_monitoring_study(
            arguments.numerical_study_directory,
            arguments.output_dir,
            selected_standardized_variance=(
                arguments.selected_standardized_variance
            ),
            windows=arguments.window,
        )
        print(f"Recorded {len(rows)} sequential health replay fits")
    elif arguments.command == "summarize":
        rows = summarize_health_monitoring_study(
            arguments.health_study_directory,
            make_plots=not arguments.no_plots,
        )
        print(f"Summarized {len(rows)} seeds")
    else:
        result = calibrate_health_thresholds(
            arguments.health_study_directory,
            adverse_z_threshold=arguments.adverse_z_threshold,
            maximum_false_positive_rate=arguments.maximum_false_positive_rate,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
