"""Offline nested study of global/archive restart budgets.

The study reconstructs fixed GP fits selected by a completed numerical-noise
study.  It never calls the FEM objective.  For each seed and GP snapshot it
creates one shared source-specific candidate pool, orders it once, optimizes
each retained start once, and evaluates restart allocations as deterministic
prefixes of those source sequences.
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

import torch
from botorch.acquisition import LogExpectedImprovement

from .bayesian import Acquisition, Bounds
from .gaussian import build_objective_gp
from .numerical_noise_study import (
    load_archived_run,
    standardized_to_raw_noise_variance,
    validate_archived_runs,
)
from .physical import geometry_signature, rasterized_volume_feasibility
from .records import (
    _git_metadata,
    atomic_csv_write,
    atomic_json_write,
    atomic_torch_save,
)
from .runtime import DEVICE, TENSOR_KWARGS, make_generator
from .sampling import (
    AREA_BROAD,
    feasible_mask,
    rank_initial_condition_pool,
    sample_stratified_feasible,
    select_archive_local_conditions,
)


STUDY_SCHEMA_VERSION = 1
DEFAULT_FIT_AFTER = (0, 5, 10, 20)
TRIAL_FIELDS = (
    "start_id",
    "seed",
    "fit_after_bo_iteration",
    "source",
    "source_prefix_index",
    "start_log_ei",
    "optimized_log_ei",
    "status",
    "optimizer_runtime_seconds",
    "warning_count",
    "endpoint_signature_hex",
    "analytical_feasible_endpoint",
    "raster_feasible_endpoint",
    "duplicate_observed_endpoint",
    "failure_reason",
)
ALLOCATION_FIELDS = (
    "seed",
    "fit_after_bo_iteration",
    "requested_global_restarts",
    "requested_local_restarts",
    "requested_total_restarts",
    "requested_global_fraction",
    "actual_global_restart_attempts",
    "actual_local_restart_attempts",
    "actual_total_restart_attempts",
    "global_restart_shortfall",
    "local_restart_shortfall",
    "optimizer_failure_count",
    "infeasible_endpoint_count",
    "duplicate_observed_endpoint_count",
    "duplicate_within_allocation_count",
    "unique_feasible_endpoint_count",
    "unique_feasible_global_endpoint_count",
    "unique_feasible_local_endpoint_count",
    "best_feasible_log_ei",
    "winning_source",
    "optimizer_runtime_seconds",
    "global_start_pairwise_distance_min",
    "global_start_pairwise_distance_mean",
    "final_compliance",
    "final_compliance_not_applicable_reason",
)


def _csv_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_rows(rows, fields, path):
    atomic_csv_write(
        [{field: _csv_value(row.get(field)) for field in fields} for row in rows],
        fields,
        Path(path),
    )


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


def _selected_noise(numerical_study_directory):
    decision_path = Path(numerical_study_directory) / "decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(f"Missing numerical-noise decision: {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    value = decision.get("selected_standardized_noise_variance")
    if decision.get("status") != "provisional_selection" or value is None:
        raise ValueError(
            "decision.json must contain a provisional fixed-noise selection."
        )
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("The selected standardized noise variance is invalid.")
    return value, f"fixed_{value:.0e}", decision_path


def _training_snapshot(run, fit_after):
    bo_iteration = run.archive["bo_iteration"].reshape(-1)
    mask = (bo_iteration < 0) | (bo_iteration <= int(fit_after))
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if not len(indices):
        raise ValueError(f"Seed {run.seed} has no data at snapshot {fit_after}.")
    stop = int(indices.max().item()) + 1
    if not bool(mask[:stop].all()) or bool(mask[stop:].any()):
        raise ValueError(f"Seed {run.seed} evaluation archive is not ordered.")
    return {
        "x": run.archive["x"][:stop].clone(),
        "compliance": run.archive["physical_compliance"][:stop].clone(),
        "response": run.archive["gp_response_prestandardization"][:stop].clone(),
        "stop": stop,
    }


def _fit_record(numerical_directory, seed, fit_after, candidate, initialization_id):
    records = _read_csv(Path(numerical_directory) / "gp_fits_numerical_study.csv")
    matches = [
        row
        for row in records
        if int(row["seed"]) == int(seed)
        and int(row["fit_after_bo_iteration"]) == int(fit_after)
        and row["noise_candidate"] == candidate
        and int(row["initialization_id"]) == int(initialization_id)
        and row["status"] == "success"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one successful state for seed={seed}, after={fit_after}, "
            f"candidate={candidate}, initialization={initialization_id}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _restore_frozen_gp(
    run,
    snapshot,
    numerical_directory,
    standardized_noise,
    candidate,
    fit_after,
    initialization_id,
):
    record = _fit_record(
        numerical_directory, run.seed, fit_after, candidate, initialization_id
    )
    state_path = Path(numerical_directory) / record["fit_state_file"]
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("source_archive_sha256") != run.archive_sha256:
        raise ValueError(f"Seed {run.seed} frozen GP refers to a different archive.")
    if int(state.get("archive_stop_exclusive", -1)) != snapshot["stop"]:
        raise ValueError(f"Seed {run.seed} frozen GP training prefix is misaligned.")
    ellipse_feature_mode = run.configuration.get("ellipse_feature_mode", "raw")
    state_feature_mode = state.get("ellipse_feature_mode", "raw")
    if state_feature_mode != ellipse_feature_mode:
        raise ValueError(
            f"Seed {run.seed} frozen GP ellipse feature mode "
            f"{state_feature_mode!r} does not match {ellipse_feature_mode!r}."
        )
    train_x = snapshot["x"].to(device=DEVICE, dtype=torch.double)
    train_y = snapshot["compliance"].to(device=DEVICE, dtype=torch.double)
    raw_noise = standardized_to_raw_noise_variance(
        standardized_noise,
        snapshot["response"],
    )
    gp, train_response = build_objective_gp(
        train_x,
        train_y,
        run.nholes,
        observation_noise_variance=raw_noise,
        ellipse_feature_mode=ellipse_feature_mode,
        xsize=run.configuration.get("xsize"),
        ysize=run.configuration.get("ysize"),
    )
    gp.load_state_dict(state["model_state_dict"])
    gp.eval()
    gp.likelihood.eval()
    return gp, train_x, train_y, train_response, raw_noise, record


def _constraints(configuration):
    return Bounds(
        configuration["xsize"],
        configuration["ysize"],
        configuration["nholes"],
        configuration["bx"],
        configuration["by"],
        configuration["solid_max"],
        configuration["smin"],
        configuration["amin"],
        configuration["raster_tolerance"],
        tri_q_min=configuration.get("tri_q_min", 0.10),
    )


def _candidate_seed(seed, fit_after):
    return int(seed) * 1_000_003 + int(fit_after) * 10_007 + 91_739


def _ordered_starts(
    run,
    fit_after,
    gp,
    acquisition,
    train_x,
    train_y,
    constraints,
    max_global,
    max_local,
    raw_pool_multiplier,
    acquisition_weight,
    novelty_weight,
    minimum_distance,
):
    generator_seed = _candidate_seed(run.seed, fit_after)
    generator = make_generator(generator_seed)
    global_requested = max_global * raw_pool_multiplier
    if global_requested:
        global_pool = sample_stratified_feasible(
            global_requested,
            run.nholes,
            int(run.configuration["htype"]),
            constraints,
            float(run.configuration["amin"]),
            area_strategies=(AREA_BROAD,),
            allow_partial=True,
            generator=generator,
        )
    else:
        global_pool = torch.empty((0, run.dimension), **TENSOR_KWARGS)
    local_requested = max_local * raw_pool_multiplier
    local_pool, local_metadata = select_archive_local_conditions(
        local_requested,
        train_x,
        train_y,
        constraints,
        gp,
    )
    observed_signatures = {
        geometry_signature(point, constraints) for point in train_x
    }
    ranked = rank_initial_condition_pool(
        acquisition,
        global_pool,
        local_pool,
        constraints,
        observed_signatures,
        attempted_starts=[],
        minimum_distance=minimum_distance,
        observed_points=train_x,
        local_metadata=local_metadata,
        acquisition_weight=acquisition_weight,
        novelty_weight=novelty_weight,
        number_global=max_global,
        global_selection_strategy="greedy_dynamic_novelty",
        ellipse_feature_mode=run.configuration.get(
            "ellipse_feature_mode", "raw"
        ),
    )
    global_starts = [row for row in ranked if row["source"] == "global"][:max_global]
    local_starts = [row for row in ranked if row["source"] == "local"][:max_local]
    for source, starts in (("global", global_starts), ("local", local_starts)):
        for index, entry in enumerate(starts, start=1):
            entry["source"] = source
            entry["source_prefix_index"] = index
            entry["start_id"] = (
                f"seed{run.seed:03d}_after{fit_after:04d}_{source}_{index:04d}"
            )
    return {
        "generator_seed": generator_seed,
        "raw_global_pool": global_pool.detach().cpu(),
        "raw_local_pool": local_pool.detach().cpu(),
        "global_starts": global_starts,
        "local_starts": local_starts,
        "acquisition": acquisition,
    }


def _optimize_start(entry, acquisition, constraints, maxiter, timeout_seconds):
    study = Acquisition(
        acquisition,
        constraints,
        [entry],
        maxiter=maxiter,
        timeout_sec=timeout_seconds,
    )
    base = {
        "start_id": entry["start_id"],
        "seed": entry["seed"],
        "fit_after_bo_iteration": entry["fit_after_bo_iteration"],
        "source": entry["source"],
        "source_prefix_index": entry["source_prefix_index"],
        "start_log_ei": float(entry["raw_acquisition_value"].item()),
    }
    if study.failed_restarts:
        result = study.failed_restarts[0]
        base.update(
            {
                "status": "optimizer_failure",
                "optimizer_runtime_seconds": result.get("elapsed_seconds"),
                "failure_reason": result.get("reason"),
            }
        )
        return base
    result = study.successful_results[0]
    candidate = result["candidate"].detach().reshape(1, -1)
    analytical = bool(feasible_mask(candidate, constraints).all())
    raster, _ = rasterized_volume_feasibility(candidate, constraints)
    raster_feasible = bool(raster.all())
    signature = geometry_signature(candidate.reshape(-1), constraints)
    observed = signature in entry["observed_signatures"]
    status = "success_feasible"
    if not analytical or not raster_feasible:
        status = "infeasible_endpoint"
    elif observed:
        status = "duplicate_observed_endpoint"
    base.update(
        {
            "status": status,
            "candidate": candidate.detach().cpu(),
            "optimized_log_ei": float(result["acquisition_value"].item()),
            "optimizer_runtime_seconds": result.get("elapsed_seconds"),
            "warning_count": result.get("warning_count", 0),
            "endpoint_signature": signature,
            "analytical_feasible_endpoint": analytical,
            "raster_feasible_endpoint": raster_feasible,
            "duplicate_observed_endpoint": observed,
        }
    )
    return base


def _trial_csv_row(outcome):
    signature = outcome.get("endpoint_signature")
    return {
        **outcome,
        "endpoint_signature_hex": (
            None if signature is None else bytes(signature).hex()
        ),
    }


def _global_diversity(entries, constraints):
    if len(entries) < 2:
        return None, None
    lower, upper = constraints.box_bounds.detach().cpu()
    scale = (upper - lower).clamp_min(1e-12)
    points = torch.stack([entry["point"].detach().cpu() for entry in entries])
    distances = torch.pdist((points - lower) / scale)
    return float(distances.min().item()), float(distances.mean().item())


def _allocation_row(seed, fit_after, ng_rst, nl_rst, global_starts, local_starts, outcomes, constraints):
    selected_global = global_starts[:ng_rst]
    selected_local = local_starts[:nl_rst]
    selected = selected_global + selected_local
    selected_outcomes = [outcomes[entry["start_id"]] for entry in selected]
    unique = []
    signatures = set()
    duplicate_count = 0
    for outcome in selected_outcomes:
        if outcome.get("status") != "success_feasible":
            continue
        signature = outcome["endpoint_signature"]
        if signature in signatures:
            duplicate_count += 1
            continue
        signatures.add(signature)
        unique.append(outcome)
    winner = max(unique, key=lambda row: row["optimized_log_ei"], default=None)
    minimum, mean = _global_diversity(selected_global, constraints)
    total = ng_rst + nl_rst
    return {
        "seed": seed,
        "fit_after_bo_iteration": fit_after,
        "requested_global_restarts": ng_rst,
        "requested_local_restarts": nl_rst,
        "requested_total_restarts": total,
        "requested_global_fraction": None if not total else ng_rst / total,
        "actual_global_restart_attempts": len(selected_global),
        "actual_local_restart_attempts": len(selected_local),
        "actual_total_restart_attempts": len(selected),
        "global_restart_shortfall": ng_rst - len(selected_global),
        "local_restart_shortfall": nl_rst - len(selected_local),
        "optimizer_failure_count": sum(
            row.get("status") == "optimizer_failure" for row in selected_outcomes
        ),
        "infeasible_endpoint_count": sum(
            row.get("status") == "infeasible_endpoint" for row in selected_outcomes
        ),
        "duplicate_observed_endpoint_count": sum(
            row.get("status") == "duplicate_observed_endpoint"
            for row in selected_outcomes
        ),
        "duplicate_within_allocation_count": duplicate_count,
        "unique_feasible_endpoint_count": len(unique),
        "unique_feasible_global_endpoint_count": sum(
            row["source"] == "global" for row in unique
        ),
        "unique_feasible_local_endpoint_count": sum(
            row["source"] == "local" for row in unique
        ),
        "best_feasible_log_ei": (
            None if winner is None else winner["optimized_log_ei"]
        ),
        "winning_source": None if winner is None else winner["source"],
        "optimizer_runtime_seconds": sum(
            row.get("optimizer_runtime_seconds") or 0.0 for row in selected_outcomes
        ),
        "global_start_pairwise_distance_min": minimum,
        "global_start_pairwise_distance_mean": mean,
        "final_compliance": None,
        "final_compliance_not_applicable_reason": (
            "Frozen-acquisition study; no FEM evaluation is performed."
        ),
    }


def run_restart_budget_study(
    run_directories,
    numerical_study_directory,
    output_directory,
    *,
    fit_after=DEFAULT_FIT_AFTER,
    allocations=((5, 0), (3, 2), (0, 5), (10, 0), (7, 3), (5, 5), (0, 10)),
    initialization_id=0,
    raw_pool_multiplier=3,
    acquisition_weight=0.55,
    novelty_weight=0.45,
    minimum_restart_distance=0.02,
    optimizer_maxiter=250,
    optimizer_timeout_seconds=60.0,
    expected_seed_count=5,
):
    """Run or resume the archive-only nested restart study."""
    runs = [load_archived_run(seed, path) for seed, path in run_directories.items()]
    runs.sort(key=lambda run: run.seed)
    if len(runs) != int(expected_seed_count):
        raise ValueError(
            f"Expected {expected_seed_count} source seeds, received {len(runs)}."
        )
    validate_archived_runs(runs)
    fit_after = tuple(int(value) for value in fit_after)
    allocations = tuple((int(global_count), int(local_count)) for global_count, local_count in allocations)
    if not allocations or any(g < 0 or l < 0 or g + l <= 0 for g, l in allocations):
        raise ValueError("Every allocation must contain non-negative counts and a positive total.")
    if raw_pool_multiplier < 1:
        raise ValueError("raw_pool_multiplier must be positive.")
    numerical_study_directory = Path(numerical_study_directory).expanduser().resolve()
    standardized_noise, noise_candidate, decision_path = _selected_noise(
        numerical_study_directory
    )
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "run_directories": {str(run.seed): str(run.directory) for run in runs},
        "numerical_study_directory": str(numerical_study_directory),
        "decision_file": str(decision_path),
        "selected_standardized_noise_variance": standardized_noise,
        "noise_candidate": noise_candidate,
        "initialization_id": int(initialization_id),
        "fit_after_bo_iterations": list(fit_after),
        "allocations": [list(value) for value in allocations],
        "raw_pool_multiplier": int(raw_pool_multiplier),
        "acquisition_weight": float(acquisition_weight),
        "novelty_weight": float(novelty_weight),
        "minimum_restart_distance": float(minimum_restart_distance),
        "optimizer_maxiter": int(optimizer_maxiter),
        "optimizer_timeout_seconds": float(optimizer_timeout_seconds),
        "adaptive_restart_allocation": False,
        "local_start_strategy": "archive",
        "source_substitution": False,
        "global_score_scaling": "roundwise_remaining_eligible_pool",
        "physical_evaluations": False,
    }
    ellipse_feature_mode = runs[0].configuration.get(
        "ellipse_feature_mode", "raw"
    )
    if ellipse_feature_mode != "raw":
        config["ellipse_feature_mode"] = ellipse_feature_mode
    config_path = output_directory / "study_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing study configuration differs; use a new output directory.")
    else:
        atomic_json_write(config, config_path)
    manifest_path = output_directory / "study_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": STUDY_SCHEMA_VERSION,
            "study_id": f"restart-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "software": _software_manifest(),
            "study_source": _git_metadata(Path(__file__).resolve().parents[2]),
        }
        atomic_json_write(manifest, manifest_path)
    pool_path = output_directory / "candidate_pools.pt"
    optimization_path = output_directory / "restart_optimizations.pt"
    pool_cache = (
        torch.load(pool_path, map_location="cpu", weights_only=False)
        if pool_path.exists()
        else {"schema_version": STUDY_SCHEMA_VERSION, "snapshots": {}}
    )
    outcomes = (
        torch.load(optimization_path, map_location="cpu", weights_only=False).get("outcomes", {})
        if optimization_path.exists()
        else {}
    )
    max_global = max(global_count for global_count, _ in allocations)
    max_local = max(local_count for _, local_count in allocations)
    allocation_rows = []
    started = time.monotonic()
    for run in runs:
        constraints = _constraints(run.configuration)
        for snapshot_index in fit_after:
            key = f"seed{run.seed:03d}_after{snapshot_index:04d}"
            snapshot = _training_snapshot(run, snapshot_index)
            gp, train_x, train_y, train_response, raw_noise, fit_record = _restore_frozen_gp(
                run,
                snapshot,
                numerical_study_directory,
                standardized_noise,
                noise_candidate,
                snapshot_index,
                initialization_id,
            )
            acquisition = LogExpectedImprovement(model=gp, best_f=train_response.max())
            if key not in pool_cache["snapshots"]:
                pool = _ordered_starts(
                    run,
                    snapshot_index,
                    gp,
                    acquisition,
                    train_x,
                    train_y,
                    constraints,
                    max_global,
                    max_local,
                    raw_pool_multiplier,
                    acquisition_weight,
                    novelty_weight,
                    minimum_restart_distance,
                )
                pool_cache["snapshots"][key] = {
                    "seed": run.seed,
                    "fit_after_bo_iteration": snapshot_index,
                    "generator_seed": pool["generator_seed"],
                    "source_archive_sha256": run.archive_sha256,
                    "archive_stop_exclusive": snapshot["stop"],
                    "raw_response_noise_variance": raw_noise,
                    "fit_id": fit_record["fit_id"],
                    "raw_global_pool": pool["raw_global_pool"],
                    "raw_local_pool": pool["raw_local_pool"],
                    "global_starts": pool["global_starts"],
                    "local_starts": pool["local_starts"],
                }
                atomic_torch_save(pool_cache, pool_path)
            stored = pool_cache["snapshots"][key]
            global_starts = stored["global_starts"]
            local_starts = stored["local_starts"]
            observed_signatures = {
                geometry_signature(point, constraints) for point in train_x
            }
            for entry in global_starts + local_starts:
                start_id = entry["start_id"]
                if start_id in outcomes:
                    continue
                work = dict(entry)
                work["point"] = work["point"].to(
                    device=DEVICE, dtype=torch.double
                )
                work.update(
                    {
                        "seed": run.seed,
                        "fit_after_bo_iteration": snapshot_index,
                        "observed_signatures": observed_signatures,
                    }
                )
                outcomes[start_id] = _optimize_start(
                    work,
                    acquisition,
                    constraints,
                    int(optimizer_maxiter),
                    float(optimizer_timeout_seconds),
                )
                atomic_torch_save(
                    {"schema_version": STUDY_SCHEMA_VERSION, "outcomes": outcomes},
                    optimization_path,
                )
                _write_rows(
                    [_trial_csv_row(outcomes[item]) for item in sorted(outcomes)],
                    TRIAL_FIELDS,
                    output_directory / "restart_trials.csv",
                )
            for global_count, local_count in allocations:
                allocation_rows.append(
                    _allocation_row(
                        run.seed,
                        snapshot_index,
                        global_count,
                        local_count,
                        global_starts,
                        local_starts,
                        outcomes,
                        constraints,
                    )
                )
            _write_rows(
                allocation_rows,
                ALLOCATION_FIELDS,
                output_directory / "allocation_results.csv",
            )
    manifest.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "snapshot_count": len(runs) * len(fit_after),
            "optimized_start_count": len(outcomes),
            "allocation_result_count": len(allocation_rows),
        }
    )
    atomic_json_write(manifest, manifest_path)
    return allocation_rows


def summarize_restart_budget_study(output_directory, *, make_plots=True):
    """Aggregate paired seed/snapshot allocation outcomes."""
    output_directory = Path(output_directory)
    rows = _read_csv(output_directory / "allocation_results.csv")
    if not rows:
        raise FileNotFoundError("No allocation_results.csv was found.")
    blocks = {}
    for row in rows:
        key = (int(row["seed"]), int(row["fit_after_bo_iteration"]))
        value = _number(row, "best_feasible_log_ei")
        if value is not None:
            blocks[key] = max(value, blocks.get(key, -math.inf))
    groups = {}
    for row in rows:
        key = (
            int(row["requested_global_restarts"]),
            int(row["requested_local_restarts"]),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for (global_count, local_count), group in sorted(groups.items(), key=lambda item: (sum(item[0]), item[0])):
        values = [_number(row, "best_feasible_log_ei") for row in group]
        finite = [value for value in values if value is not None]
        regrets = []
        for row, value in zip(group, values):
            block = (int(row["seed"]), int(row["fit_after_bo_iteration"]))
            if value is not None and block in blocks:
                regrets.append(blocks[block] - value)
        summaries.append(
            {
                "ng_rst": global_count,
                "nl_rst": local_count,
                "total_restarts": global_count + local_count,
                "global_fraction": global_count / (global_count + local_count),
                "block_count": len(group),
                "metric_available_rate": len(finite) / len(group),
                "mean_best_feasible_log_ei": None if not finite else statistics.fmean(finite),
                "median_best_feasible_log_ei": None if not finite else statistics.median(finite),
                "mean_paired_log_ei_regret": None if not regrets else statistics.fmean(regrets),
                "se_paired_log_ei_regret": (
                    None
                    if len(regrets) < 2
                    else statistics.stdev(regrets) / math.sqrt(len(regrets))
                ),
                "mean_optimizer_runtime_seconds": statistics.fmean(
                    _number(row, "optimizer_runtime_seconds") or 0.0 for row in group
                ),
                "mean_optimizer_failure_count": statistics.fmean(
                    _number(row, "optimizer_failure_count") or 0.0 for row in group
                ),
                "mean_duplicate_endpoint_count": statistics.fmean(
                    (_number(row, "duplicate_within_allocation_count") or 0.0)
                    + (_number(row, "duplicate_observed_endpoint_count") or 0.0)
                    for row in group
                ),
                "total_source_shortfall": sum(
                    int(float(row["global_restart_shortfall"]))
                    + int(float(row["local_restart_shortfall"]))
                    for row in group
                ),
            }
        )
    fields = tuple(dict.fromkeys(key for row in summaries for key in row))
    _write_rows(summaries, fields, output_directory / "summary_by_allocation.csv")
    if make_plots:
        try:
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(1, 2, figsize=(11, 4))
            for total in sorted({row["total_restarts"] for row in summaries}):
                group = [row for row in summaries if row["total_restarts"] == total]
                axes[0].plot(
                    [row["global_fraction"] for row in group],
                    [row["mean_paired_log_ei_regret"] for row in group],
                    marker="o",
                    label=f"total={total}",
                )
            axes[0].set(xlabel="Global restart fraction", ylabel="Mean paired LogEI regret")
            axes[0].legend(fontsize=8)
            axes[1].scatter(
                [row["mean_optimizer_runtime_seconds"] for row in summaries],
                [row["mean_paired_log_ei_regret"] for row in summaries],
            )
            axes[1].set(xlabel="Mean optimizer runtime (s)", ylabel="Mean paired LogEI regret")
            figure.tight_layout()
            figure.savefig(output_directory / "restart_budget_summary.png", dpi=180)
            plt.close(figure)
        except (ImportError, ValueError):
            pass
    return summaries


def recommend_restart_allocation(output_directory, *, minimum_metric_rate=0.95):
    """Write a provisional one-standard-error nested-restart recommendation."""
    output_directory = Path(output_directory)
    summaries = summarize_restart_budget_study(output_directory, make_plots=False)
    eligible = [
        row
        for row in summaries
        if row["metric_available_rate"] >= minimum_metric_rate
        and row["mean_paired_log_ei_regret"] is not None
        and row["total_source_shortfall"] == 0
    ]
    if not eligible:
        recommendation = {
            "status": "insufficient_evidence",
            "reason": "No allocation met metric-availability and source-quota requirements.",
            "automatic_runtime_default_change": False,
        }
    else:
        best = min(eligible, key=lambda row: row["mean_paired_log_ei_regret"])
        best_upper = best["mean_paired_log_ei_regret"] + (
            best["se_paired_log_ei_regret"] or 0.0
        )
        one_se = [
            row
            for row in eligible
            if row["mean_paired_log_ei_regret"]
            - (row["se_paired_log_ei_regret"] or 0.0)
            <= best_upper
        ]
        selected = min(
            one_se,
            key=lambda row: (
                row["total_restarts"],
                row["mean_paired_log_ei_regret"],
                row["mean_optimizer_runtime_seconds"],
            ),
        )
        recommendation = {
            "status": "provisional",
            "ng_rst": selected["ng_rst"],
            "nl_rst": selected["nl_rst"],
            "total_restarts": selected["total_restarts"],
            "global_fraction": selected["global_fraction"],
            "selection_rule": (
                "Smallest restart total whose paired-regret one-standard-error "
                "interval overlaps that of the lowest mean feasible-LogEI regret; "
                "regret and runtime break ties."
            ),
            "local_start_strategy": "archive",
            "adaptive_restart_allocation": False,
            "automatic_runtime_default_change": False,
            "requires_sequential_bo_confirmation": True,
        }
    atomic_json_write(
        recommendation,
        output_directory / "recommended_restart_allocation.json",
    )
    return recommendation


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("study_directory")
    summarize.add_argument("--no-plots", action="store_true")
    recommend = subparsers.add_parser("recommend")
    recommend.add_argument("study_directory")
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    if arguments.command == "summarize":
        rows = summarize_restart_budget_study(
            arguments.study_directory, make_plots=not arguments.no_plots
        )
        print(f"Summarized {len(rows)} restart allocations")
    else:
        print(json.dumps(recommend_restart_allocation(arguments.study_directory), indent=2))


if __name__ == "__main__":
    main()
