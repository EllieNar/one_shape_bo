"""Offline screening and sequential confirmation of global restart weights.

The frozen stage changes only the normalized acquisition/novelty balance used
by the production greedy global selector.  It reuses archived GP fits and never
evaluates the FEM objective.  Sequential confirmation consumes completed BO
run archives; it does not launch expensive optimization runs implicitly.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from botorch.acquisition import LogExpectedImprovement

from .numerical_noise_study import load_archived_run, validate_archived_runs
from .physical import geometry_signature
from .records import atomic_csv_write, atomic_json_write, atomic_torch_save
from .restart_budget_study import (
    _constraints,
    _optimize_start,
    _restore_frozen_gp,
    _selected_noise,
    _software_manifest,
    _training_snapshot,
)
from .runtime import TENSOR_KWARGS, make_generator
from .sampling import (
    AREA_BROAD,
    rank_initial_condition_pool,
    sample_stratified_feasible,
)


STUDY_SCHEMA_VERSION = 1
DEFAULT_FIT_AFTER = (0, 5, 10, 20)
DEFAULT_ACQUISITION_FRACTIONS = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)
DEFAULT_GLOBAL_BUDGETS = (10,)

SELECTION_FIELDS = (
    "seed",
    "fit_after_bo_iteration",
    "global_restart_budget",
    "acquisition_fraction",
    "novelty_fraction",
    "selection_step",
    "start_id",
    "start_signature_hex",
    "raw_acquisition_value",
    "normalized_acquisition_score",
    "raw_novelty_distance",
    "normalized_novelty_score",
    "combined_rank_score",
    "selection_runtime_seconds",
)

TRIAL_FIELDS = (
    "start_id",
    "seed",
    "fit_after_bo_iteration",
    "global_restart_budget",
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

RESULT_FIELDS = (
    "seed",
    "fit_after_bo_iteration",
    "global_restart_budget",
    "acquisition_fraction",
    "novelty_fraction",
    "requested_global_restarts",
    "actual_global_restart_attempts",
    "global_restart_shortfall",
    "optimizer_failure_count",
    "infeasible_endpoint_count",
    "duplicate_observed_endpoint_count",
    "duplicate_within_weight_count",
    "unique_feasible_endpoint_count",
    "best_feasible_log_ei",
    "selection_runtime_seconds",
    "optimizer_runtime_seconds",
    "start_pairwise_distance_min",
    "start_pairwise_distance_mean",
    "final_compliance",
    "final_compliance_not_applicable_reason",
)

SEQUENTIAL_FIELDS = (
    "seed",
    "acquisition_fraction",
    "novelty_fraction",
    "run_directory",
    "run_id",
    "archive_sha256",
    "burn_in_count",
    "bo_evaluation_count",
    "best_burn_in_compliance",
    "best_compliance",
    "final_evaluation_compliance",
    "best_bo_iteration",
    "best_so_far_compliance_auc",
    "compliance_target",
    "first_bo_iteration_at_or_below_target",
    "total_ranking_runtime_seconds",
    "total_optimizer_runtime_seconds",
    "htype",
    "nholes",
    "ng_rst",
    "nl_rst",
    "adaptive_restart_allocation",
    "global_restart_selection_strategy",
)


def _csv_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _write_rows(rows, fields, path):
    atomic_csv_write(
        [{field: _csv_value(row.get(field)) for field in fields} for row in rows],
        fields,
        Path(path),
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


def _fraction_label(value):
    return f"a{int(round(float(value) * 10_000)):05d}"


def _complementary_fraction(value):
    return round(1.0 - float(value), 12)


def _point_digest(point):
    tensor = point.detach().cpu().to(dtype=torch.double).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _pool_seed(seed, fit_after, budget):
    return (
        int(seed) * 1_000_003
        + int(fit_after) * 10_007
        + int(budget) * 1_009
        + 314_159
    )


def _validate_fractions(values):
    fractions = tuple(float(value) for value in values)
    if not fractions or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in fractions
    ):
        raise ValueError("Acquisition fractions must lie in [0, 1].")
    if len(set(fractions)) != len(fractions):
        raise ValueError("Acquisition fractions must be unique.")
    return fractions


def _validate_budgets(values):
    budgets = tuple(int(value) for value in values)
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("Every global restart budget must be positive.")
    if len(set(budgets)) != len(budgets):
        raise ValueError("Global restart budgets must be unique.")
    return budgets


class _CachedAcquisition:
    """Return cached values for an unchanged candidate pool."""

    def __init__(self, points, values, fallback):
        self.fallback = fallback
        self.values = {
            _point_digest(point): value.detach().clone()
            for point, value in zip(points, values.reshape(-1))
        }

    def __call__(self, candidates):
        flat = candidates.reshape(-1, candidates.shape[-1])
        output = []
        missing = []
        for index, point in enumerate(flat):
            value = self.values.get(_point_digest(point))
            if value is None:
                missing.append((index, point))
                output.append(None)
            else:
                output.append(value.to(device=point.device, dtype=point.dtype))
        if missing:
            unknown = torch.stack([point for _, point in missing]).unsqueeze(1)
            evaluated = self.fallback(unknown).reshape(-1)
            for (index, _), value in zip(missing, evaluated):
                output[index] = value
        return torch.stack(output)


def _shared_global_pool(run, fit_after, budget, raw_pool_multiplier, constraints):
    generator_seed = _pool_seed(run.seed, fit_after, budget)
    generator = make_generator(generator_seed)
    pool = sample_stratified_feasible(
        int(budget) * int(raw_pool_multiplier),
        run.nholes,
        int(run.configuration["htype"]),
        constraints,
        float(run.configuration["amin"]),
        area_strategies=(AREA_BROAD,),
        allow_partial=True,
        generator=generator,
    )
    return pool, generator_seed


def _select_for_fraction(
    cached_acquisition,
    pool,
    train_x,
    constraints,
    budget,
    acquisition_fraction,
    minimum_distance,
):
    observed_signatures = {
        geometry_signature(point, constraints) for point in train_x
    }
    started = time.perf_counter()
    ranked = rank_initial_condition_pool(
        cached_acquisition,
        pool,
        torch.empty((0, pool.shape[-1]), **TENSOR_KWARGS),
        constraints,
        observed_signatures,
        attempted_starts=[],
        minimum_distance=minimum_distance,
        observed_points=train_x,
        acquisition_weight=acquisition_fraction,
        novelty_weight=_complementary_fraction(acquisition_fraction),
        number_global=budget,
        global_selection_strategy="greedy_dynamic_novelty",
    )
    elapsed = time.perf_counter() - started
    selected = sorted(
        (entry for entry in ranked if entry["source"] == "global"),
        key=lambda entry: entry["global_selection_step"],
    )[:budget]
    return selected, elapsed


def _start_id(seed, fit_after, budget, point):
    return (
        f"seed{int(seed):03d}_after{int(fit_after):04d}_g{int(budget):03d}_"
        f"{_point_digest(point)[:20]}"
    )


def _cpu_entry(entry, seed, fit_after, budget):
    result = {}
    for key, value in entry.items():
        result[key] = value.detach().cpu() if torch.is_tensor(value) else value
    result.update(
        {
            "start_id": _start_id(seed, fit_after, budget, entry["point"]),
            "seed": int(seed),
            "fit_after_bo_iteration": int(fit_after),
            "global_restart_budget": int(budget),
            "source": "global",
            "source_prefix_index": int(entry["global_selection_step"]),
        }
    )
    return result


def _selection_row(entry, fraction, elapsed):
    signature = entry.get("signature")
    return {
        "seed": entry["seed"],
        "fit_after_bo_iteration": entry["fit_after_bo_iteration"],
        "global_restart_budget": entry["global_restart_budget"],
        "acquisition_fraction": fraction,
        "novelty_fraction": _complementary_fraction(fraction),
        "selection_step": entry["global_selection_step"],
        "start_id": entry["start_id"],
        "start_signature_hex": (
            None if signature is None else bytes(signature).hex()
        ),
        "raw_acquisition_value": float(entry["raw_acquisition_value"].item()),
        "normalized_acquisition_score": float(
            entry["normalized_acquisition_score"].item()
        ),
        "raw_novelty_distance": float(entry["raw_novelty_distance"].item()),
        "normalized_novelty_score": float(
            entry["normalized_novelty_score"].item()
        ),
        "combined_rank_score": float(entry["rank_score"].item()),
        "selection_runtime_seconds": elapsed,
    }


def _trial_row(outcome):
    signature = outcome.get("endpoint_signature")
    return {
        **outcome,
        "endpoint_signature_hex": (
            None if signature is None else bytes(signature).hex()
        ),
    }


def _diversity(entries, constraints):
    if len(entries) < 2:
        return None, None
    lower, upper = constraints.box_bounds.detach().cpu()
    scale = (upper - lower).clamp_min(1e-12)
    points = torch.stack([entry["point"].detach().cpu() for entry in entries])
    distances = torch.pdist((points - lower) / scale)
    return float(distances.min().item()), float(distances.mean().item())


def _weight_result_row(
    seed,
    fit_after,
    budget,
    fraction,
    entries,
    outcomes,
    selection_runtime,
    constraints,
):
    selected_outcomes = [outcomes[entry["start_id"]] for entry in entries]
    unique = []
    endpoint_signatures = set()
    duplicate_count = 0
    for outcome in selected_outcomes:
        if outcome.get("status") != "success_feasible":
            continue
        signature = outcome["endpoint_signature"]
        if signature in endpoint_signatures:
            duplicate_count += 1
            continue
        endpoint_signatures.add(signature)
        unique.append(outcome)
    winner = max(unique, key=lambda row: row["optimized_log_ei"], default=None)
    minimum, mean = _diversity(entries, constraints)
    return {
        "seed": seed,
        "fit_after_bo_iteration": fit_after,
        "global_restart_budget": budget,
        "acquisition_fraction": fraction,
        "novelty_fraction": _complementary_fraction(fraction),
        "requested_global_restarts": budget,
        "actual_global_restart_attempts": len(entries),
        "global_restart_shortfall": budget - len(entries),
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
        "duplicate_within_weight_count": duplicate_count,
        "unique_feasible_endpoint_count": len(unique),
        "best_feasible_log_ei": (
            None if winner is None else winner["optimized_log_ei"]
        ),
        "selection_runtime_seconds": selection_runtime,
        "optimizer_runtime_seconds": sum(
            row.get("optimizer_runtime_seconds") or 0.0
            for row in selected_outcomes
        ),
        "start_pairwise_distance_min": minimum,
        "start_pairwise_distance_mean": mean,
        "final_compliance": None,
        "final_compliance_not_applicable_reason": (
            "Frozen-acquisition screening; no FEM evaluation is performed."
        ),
    }


def run_global_restart_weight_study(
    run_directories,
    numerical_study_directory,
    output_directory,
    *,
    fit_after=DEFAULT_FIT_AFTER,
    acquisition_fractions=DEFAULT_ACQUISITION_FRACTIONS,
    global_restart_budgets=DEFAULT_GLOBAL_BUDGETS,
    initialization_id=0,
    raw_pool_multiplier=3,
    minimum_restart_distance=0.02,
    optimizer_maxiter=250,
    optimizer_timeout_seconds=60.0,
    expected_seed_count=5,
    primary_global_restart_budget=10,
    incumbent_acquisition_fraction=0.55,
    reference_log_ei_tolerance=1e-8,
):
    """Run or resume frozen-GP screening without physical evaluations."""
    fractions = _validate_fractions(acquisition_fractions)
    budgets = _validate_budgets(global_restart_budgets)
    if int(primary_global_restart_budget) not in budgets:
        raise ValueError(
            "The primary global budget must be one of the studied budgets."
        )
    if float(incumbent_acquisition_fraction) not in fractions:
        raise ValueError("The incumbent acquisition fraction must be studied.")
    if int(raw_pool_multiplier) < 1:
        raise ValueError("raw_pool_multiplier must be positive.")
    if (
        not math.isfinite(reference_log_ei_tolerance)
        or reference_log_ei_tolerance < 0
    ):
        raise ValueError("reference_log_ei_tolerance must be non-negative.")
    runs = [load_archived_run(seed, path) for seed, path in run_directories.items()]
    runs.sort(key=lambda run: run.seed)
    if len(runs) != int(expected_seed_count):
        raise ValueError(
            f"Expected {expected_seed_count} source seeds, received {len(runs)}."
        )
    validate_archived_runs(runs)
    fit_after = tuple(int(value) for value in fit_after)
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
        "acquisition_fractions": list(fractions),
        "novelty_fraction_rule": "1 - acquisition_fraction",
        "global_restart_budgets": list(budgets),
        "primary_global_restart_budget": int(primary_global_restart_budget),
        "incumbent_acquisition_fraction": float(incumbent_acquisition_fraction),
        "reference_log_ei_tolerance": float(reference_log_ei_tolerance),
        "raw_pool_multiplier": int(raw_pool_multiplier),
        "minimum_restart_distance": float(minimum_restart_distance),
        "optimizer_maxiter": int(optimizer_maxiter),
        "optimizer_timeout_seconds": float(optimizer_timeout_seconds),
        "global_restart_selection_strategy": "greedy_dynamic_novelty",
        "global_score_scaling": "roundwise_remaining_eligible_pool",
        "local_restarts": 0,
        "adaptive_restart_allocation": False,
        "source_substitution": False,
        "physical_evaluations": False,
    }
    config_path = output_directory / "study_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(
                "Existing study configuration differs; use a new output directory."
            )
    else:
        atomic_json_write(config, config_path)
    manifest_path = output_directory / "study_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        from .records import _git_metadata

        manifest = {
            "schema_version": STUDY_SCHEMA_VERSION,
            "study_id": f"weight-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}",
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
        else {"schema_version": STUDY_SCHEMA_VERSION, "blocks": {}}
    )
    outcomes = (
        torch.load(optimization_path, map_location="cpu", weights_only=False).get(
            "outcomes", {}
        )
        if optimization_path.exists()
        else {}
    )
    result_rows = []
    selection_rows = []
    started = time.monotonic()
    for run in runs:
        constraints = _constraints(run.configuration)
        for snapshot_index in fit_after:
            snapshot = _training_snapshot(run, snapshot_index)
            gp, train_x, train_y, train_response, raw_noise, fit_record = (
                _restore_frozen_gp(
                    run,
                    snapshot,
                    numerical_study_directory,
                    standardized_noise,
                    noise_candidate,
                    snapshot_index,
                    initialization_id,
                )
            )
            del train_y
            acquisition = LogExpectedImprovement(
                model=gp, best_f=train_response.max()
            )
            observed_signatures = {
                geometry_signature(point, constraints) for point in train_x
            }
            for budget in budgets:
                block_key = (
                    f"seed{run.seed:03d}_after{snapshot_index:04d}_g{budget:03d}"
                )
                if block_key not in pool_cache["blocks"]:
                    pool, generator_seed = _shared_global_pool(
                        run,
                        snapshot_index,
                        budget,
                        raw_pool_multiplier,
                        constraints,
                    )
                    with torch.no_grad():
                        acquisition_values = (
                            acquisition(pool.unsqueeze(1)).reshape(-1)
                            if len(pool)
                            else torch.empty((0,), **TENSOR_KWARGS)
                        )
                    pool_cache["blocks"][block_key] = {
                        "seed": run.seed,
                        "fit_after_bo_iteration": snapshot_index,
                        "global_restart_budget": budget,
                        "generator_seed": generator_seed,
                        "source_archive_sha256": run.archive_sha256,
                        "archive_stop_exclusive": snapshot["stop"],
                        "raw_response_noise_variance": raw_noise,
                        "fit_id": fit_record["fit_id"],
                        "raw_global_pool": pool.detach().cpu(),
                        "raw_acquisition_values": acquisition_values.detach().cpu(),
                        "selections": {},
                    }
                    atomic_torch_save(pool_cache, pool_path)
                stored = pool_cache["blocks"][block_key]
                pool = stored["raw_global_pool"].to(**TENSOR_KWARGS)
                acquisition_values = stored["raw_acquisition_values"].to(
                    **TENSOR_KWARGS
                )
                cached = _CachedAcquisition(pool, acquisition_values, acquisition)
                for fraction in fractions:
                    label = _fraction_label(fraction)
                    if label not in stored["selections"]:
                        selected, elapsed = _select_for_fraction(
                            cached,
                            pool,
                            train_x,
                            constraints,
                            budget,
                            fraction,
                            minimum_restart_distance,
                        )
                        stored["selections"][label] = {
                            "acquisition_fraction": fraction,
                            "selection_runtime_seconds": elapsed,
                            "entries": [
                                _cpu_entry(
                                    entry,
                                    run.seed,
                                    snapshot_index,
                                    budget,
                                )
                                for entry in selected
                            ],
                        }
                        atomic_torch_save(pool_cache, pool_path)
                    selection = stored["selections"][label]
                    entries = selection["entries"]
                    selection_rows.extend(
                        _selection_row(
                            entry,
                            fraction,
                            selection["selection_runtime_seconds"],
                        )
                        for entry in entries
                    )
                    for entry in entries:
                        if entry["start_id"] in outcomes:
                            continue
                        work = dict(entry)
                        work["point"] = work["point"].to(**TENSOR_KWARGS)
                        work["observed_signatures"] = observed_signatures
                        outcomes[entry["start_id"]] = _optimize_start(
                            work,
                            acquisition,
                            constraints,
                            int(optimizer_maxiter),
                            float(optimizer_timeout_seconds),
                        )
                        outcomes[entry["start_id"]][
                            "global_restart_budget"
                        ] = budget
                        atomic_torch_save(
                            {
                                "schema_version": STUDY_SCHEMA_VERSION,
                                "outcomes": outcomes,
                            },
                            optimization_path,
                        )
                        _write_rows(
                            [_trial_row(outcomes[key]) for key in sorted(outcomes)],
                            TRIAL_FIELDS,
                            output_directory / "unique_restart_trials.csv",
                        )
                    result_rows.append(
                        _weight_result_row(
                            run.seed,
                            snapshot_index,
                            budget,
                            fraction,
                            entries,
                            outcomes,
                            selection["selection_runtime_seconds"],
                            constraints,
                        )
                    )
                _write_rows(
                    selection_rows,
                    SELECTION_FIELDS,
                    output_directory / "weight_selection_steps.csv",
                )
                _write_rows(
                    result_rows,
                    RESULT_FIELDS,
                    output_directory / "weight_results.csv",
                )
    manifest.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "seed_count": len(runs),
            "snapshot_count": len(runs) * len(fit_after),
            "pool_block_count": len(runs) * len(fit_after) * len(budgets),
            "weight_result_count": len(result_rows),
            "optimized_unique_start_count": len(outcomes),
        }
    )
    atomic_json_write(manifest, manifest_path)
    return result_rows


def _mean(values):
    finite = [value for value in values if value is not None]
    return None if not finite else statistics.fmean(finite)


def _standard_error(values):
    finite = [value for value in values if value is not None]
    if len(finite) < 2:
        return None
    return statistics.stdev(finite) / math.sqrt(len(finite))


def summarize_global_restart_weight_study(output_directory, *, make_plots=True):
    """Aggregate snapshots within seed before comparing weight fractions."""
    output_directory = Path(output_directory)
    rows = _read_rows(output_directory / "weight_results.csv")
    if not rows:
        raise FileNotFoundError("No weight_results.csv was found.")
    config = json.loads(
        (output_directory / "study_config.json").read_text(encoding="utf-8")
    )
    reference_tolerance = float(config.get("reference_log_ei_tolerance", 1e-8))
    block_best = {}
    for row in rows:
        block = (
            int(row["seed"]),
            int(row["fit_after_bo_iteration"]),
            int(row["global_restart_budget"]),
        )
        value = _number(row, "best_feasible_log_ei")
        if value is not None:
            block_best[block] = max(value, block_best.get(block, -math.inf))
    seed_groups = {}
    detailed = []
    for row in rows:
        block = (
            int(row["seed"]),
            int(row["fit_after_bo_iteration"]),
            int(row["global_restart_budget"]),
        )
        value = _number(row, "best_feasible_log_ei")
        regret = (
            None
            if value is None or block not in block_best
            else block_best[block] - value
        )
        item = dict(row)
        item["paired_log_ei_regret"] = regret
        detailed.append(item)
        key = (
            int(row["seed"]),
            int(row["global_restart_budget"]),
            float(row["acquisition_fraction"]),
        )
        seed_groups.setdefault(key, []).append((row, regret))
    seed_rows = []
    for (seed, budget, fraction), group in sorted(seed_groups.items()):
        regrets = [regret for _, regret in group if regret is not None]
        seed_rows.append(
            {
                "seed": seed,
                "global_restart_budget": budget,
                "acquisition_fraction": fraction,
                "novelty_fraction": _complementary_fraction(fraction),
                "snapshot_count": len(group),
                "metric_available_rate": len(regrets) / len(group),
                "mean_paired_log_ei_regret": _mean(regrets),
                "reference_attainment_rate": (
                    None
                    if not regrets
                    else sum(value <= reference_tolerance for value in regrets)
                    / len(regrets)
                ),
                "mean_best_feasible_log_ei": _mean(
                    [_number(row, "best_feasible_log_ei") for row, _ in group]
                ),
                "mean_selection_runtime_seconds": _mean(
                    [_number(row, "selection_runtime_seconds") for row, _ in group]
                ),
                "mean_optimizer_runtime_seconds": _mean(
                    [_number(row, "optimizer_runtime_seconds") for row, _ in group]
                ),
                "mean_start_pairwise_distance": _mean(
                    [_number(row, "start_pairwise_distance_mean") for row, _ in group]
                ),
                "total_shortfall": sum(
                    int(float(row["global_restart_shortfall"])) for row, _ in group
                ),
                "total_failures": sum(
                    int(float(row["optimizer_failure_count"])) for row, _ in group
                ),
                "total_duplicates": sum(
                    int(float(row["duplicate_within_weight_count"]))
                    + int(float(row["duplicate_observed_endpoint_count"]))
                    for row, _ in group
                ),
            }
        )
    summaries = []
    summary_groups = {}
    for row in seed_rows:
        key = (
            int(row["global_restart_budget"]),
            float(row["acquisition_fraction"]),
        )
        summary_groups.setdefault(key, []).append(row)
    for (budget, fraction), group in sorted(summary_groups.items()):
        regrets = [row["mean_paired_log_ei_regret"] for row in group]
        summaries.append(
            {
                "global_restart_budget": budget,
                "acquisition_fraction": fraction,
                "novelty_fraction": _complementary_fraction(fraction),
                "seed_count": len(group),
                "mean_seed_paired_log_ei_regret": _mean(regrets),
                "se_seed_paired_log_ei_regret": _standard_error(regrets),
                "mean_seed_reference_attainment_rate": _mean(
                    [row["reference_attainment_rate"] for row in group]
                ),
                "mean_best_feasible_log_ei": _mean(
                    [row["mean_best_feasible_log_ei"] for row in group]
                ),
                "mean_selection_runtime_seconds": _mean(
                    [row["mean_selection_runtime_seconds"] for row in group]
                ),
                "mean_optimizer_runtime_seconds": _mean(
                    [row["mean_optimizer_runtime_seconds"] for row in group]
                ),
                "mean_start_pairwise_distance": _mean(
                    [row["mean_start_pairwise_distance"] for row in group]
                ),
                "total_shortfall": sum(row["total_shortfall"] for row in group),
                "total_failures": sum(row["total_failures"] for row in group),
                "total_duplicates": sum(row["total_duplicates"] for row in group),
            }
        )
    _write_rows(
        detailed,
        RESULT_FIELDS + ("paired_log_ei_regret",),
        output_directory / "weight_results_with_regret.csv",
    )
    _write_rows(
        seed_rows,
        tuple(dict.fromkeys(key for row in seed_rows for key in row)),
        output_directory / "summary_by_seed_weight.csv",
    )
    _write_rows(
        summaries,
        tuple(dict.fromkeys(key for row in summaries for key in row)),
        output_directory / "summary_by_weight.csv",
    )
    incumbent = float(config["incumbent_acquisition_fraction"])
    incumbent_rows = {
        (int(row["seed"]), int(row["global_restart_budget"])): row
        for row in seed_rows
        if math.isclose(float(row["acquisition_fraction"]), incumbent)
    }
    pairwise = []
    for row in seed_rows:
        reference = incumbent_rows.get(
            (int(row["seed"]), int(row["global_restart_budget"]))
        )
        if reference is None:
            continue
        pairwise.append(
            {
                "seed": row["seed"],
                "global_restart_budget": row["global_restart_budget"],
                "acquisition_fraction": row["acquisition_fraction"],
                "novelty_fraction": row["novelty_fraction"],
                "incumbent_acquisition_fraction": incumbent,
                "candidate_minus_incumbent_mean_regret": (
                    row["mean_paired_log_ei_regret"]
                    - reference["mean_paired_log_ei_regret"]
                ),
            }
        )
    _write_rows(
        pairwise,
        tuple(dict.fromkeys(key for row in pairwise for key in row)),
        output_directory / "pairwise_vs_default.csv",
    )
    if make_plots:
        try:
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(1, 2, figsize=(11, 4))
            for budget in sorted({row["global_restart_budget"] for row in summaries}):
                group = [
                    row for row in summaries if row["global_restart_budget"] == budget
                ]
                axes[0].errorbar(
                    [row["acquisition_fraction"] for row in group],
                    [row["mean_seed_paired_log_ei_regret"] for row in group],
                    yerr=[
                        row["se_seed_paired_log_ei_regret"] or 0.0 for row in group
                    ],
                    marker="o",
                    label=f"global={budget}",
                )
                axes[1].plot(
                    [row["acquisition_fraction"] for row in group],
                    [row["mean_start_pairwise_distance"] for row in group],
                    marker="o",
                    label=f"global={budget}",
                )
            axes[0].set(
                xlabel="Acquisition fraction",
                ylabel="Mean seed-level paired LogEI regret",
            )
            axes[1].set(
                xlabel="Acquisition fraction",
                ylabel="Mean normalized start distance",
            )
            axes[0].legend(fontsize=8)
            axes[1].legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(output_directory / "weight_tradeoff.png", dpi=180)
            plt.close(figure)
        except (ImportError, ValueError):
            pass
    return summaries


def recommend_global_restart_weights(
    output_directory,
    *,
    maximum_challengers=2,
    minimum_seed_count=5,
):
    """Nominate frozen-study challengers without changing runtime defaults."""
    output_directory = Path(output_directory)
    summaries = summarize_global_restart_weight_study(
        output_directory, make_plots=False
    )
    config = json.loads(
        (output_directory / "study_config.json").read_text(encoding="utf-8")
    )
    budget = int(config["primary_global_restart_budget"])
    incumbent = float(config["incumbent_acquisition_fraction"])
    eligible = [
        row
        for row in summaries
        if int(row["global_restart_budget"]) == budget
        and int(row["seed_count"]) >= int(minimum_seed_count)
        and row["mean_seed_paired_log_ei_regret"] is not None
        and int(row["total_shortfall"]) == 0
    ]
    if not eligible:
        recommendation = {
            "status": "insufficient_evidence",
            "reason": "No primary-budget weight met seed-count and quota requirements.",
            "automatic_runtime_default_change": False,
        }
    else:
        best = min(eligible, key=lambda row: row["mean_seed_paired_log_ei_regret"])
        limit = best["mean_seed_paired_log_ei_regret"] + (
            best["se_seed_paired_log_ei_regret"] or 0.0
        )
        one_se = [
            row
            for row in eligible
            if row["mean_seed_paired_log_ei_regret"]
            - (row["se_seed_paired_log_ei_regret"] or 0.0)
            <= limit
        ]
        challengers = sorted(
            (
                row
                for row in one_se
                if not math.isclose(row["acquisition_fraction"], incumbent)
            ),
            key=lambda row: (
                row["mean_seed_paired_log_ei_regret"],
                -row["mean_seed_reference_attainment_rate"],
                abs(row["acquisition_fraction"] - incumbent),
            ),
        )[: int(maximum_challengers)]
        recommendation = {
            "status": "provisional_screening",
            "primary_global_restart_budget": budget,
            "incumbent": {
                "acquisition_weight": incumbent,
                "novelty_weight": _complementary_fraction(incumbent),
            },
            "best_frozen_screen": {
                "acquisition_weight": best["acquisition_fraction"],
                "novelty_weight": _complementary_fraction(
                    best["acquisition_fraction"]
                ),
            },
            "challengers": [
                {
                    "acquisition_weight": row["acquisition_fraction"],
                    "novelty_weight": _complementary_fraction(
                        row["acquisition_fraction"]
                    ),
                }
                for row in challengers
            ],
            "selection_rule": (
                "Seed-level paired feasible-LogEI regret with a one-standard-error "
                "screen; at most two challengers proceed to independent sequential BO."
            ),
            "automatic_runtime_default_change": False,
            "requires_independent_sequential_confirmation": True,
        }
    atomic_json_write(
        recommendation,
        output_directory / "recommended_global_weights.json",
    )
    return recommendation


def _config_without_pairing_fields(configuration):
    excluded = {"acquisition_weight", "novelty_weight", "output_dir"}
    return {key: value for key, value in configuration.items() if key not in excluded}


def _config_without_seed_or_pairing_fields(configuration):
    result = _config_without_pairing_fields(configuration)
    result.pop("seed", None)
    return result


def _runtime_totals(run_directory):
    rows = _read_rows(Path(run_directory) / "iteration_diagnostics.csv")
    return (
        sum(_number(row, "ranking_runtime_seconds") or 0.0 for row in rows),
        sum(_number(row, "optimizer_runtime_seconds") or 0.0 for row in rows),
    )


def _archive_summary(run, fraction, compliance_target=None):
    bo_iteration = run.archive["bo_iteration"].reshape(-1)
    compliance = run.archive["physical_compliance"].reshape(-1)
    feasible = run.archive["feasible"].reshape(-1).bool()
    valid = feasible & torch.isfinite(compliance)
    if not bool(valid.any()):
        raise ValueError(f"{run.directory} contains no feasible compliance result.")
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    best_position = valid_indices[torch.argmin(compliance[valid])]
    burn_mask = bo_iteration < 0
    valid_burn = valid & burn_mask
    if not bool(valid_burn.any()):
        raise ValueError(f"{run.directory} contains no feasible burn-in result.")
    best_burn_in = float(compliance[valid_burn].min().item())
    bo_mask = bo_iteration > 0
    bo_indices = torch.nonzero(bo_mask, as_tuple=False).reshape(-1)
    incumbent = best_burn_in
    best_curve = []
    first_target = (
        0
        if compliance_target is not None and incumbent <= compliance_target
        else None
    )
    for index in bo_indices:
        if bool(valid[index]):
            incumbent = min(incumbent, float(compliance[index].item()))
        best_curve.append(incumbent)
        iteration = int(bo_iteration[index].item())
        if (
            compliance_target is not None
            and first_target is None
            and incumbent <= compliance_target
        ):
            first_target = iteration
    ranking_runtime, optimizer_runtime = _runtime_totals(run.directory)
    manifest = run.manifest
    return {
        "seed": run.seed,
        "acquisition_fraction": fraction,
        "novelty_fraction": _complementary_fraction(fraction),
        "run_directory": str(run.directory),
        "run_id": manifest.get("run_id"),
        "archive_sha256": run.archive_sha256,
        "burn_in_count": int((bo_iteration < 0).sum().item()),
        "bo_evaluation_count": int(bo_mask.sum().item()),
        "best_burn_in_compliance": best_burn_in,
        "best_compliance": float(compliance[best_position].item()),
        "final_evaluation_compliance": (
            None if not len(bo_indices) else float(compliance[bo_indices[-1]].item())
        ),
        "best_bo_iteration": int(bo_iteration[best_position].item()),
        "best_so_far_compliance_auc": (
            None if not best_curve else statistics.fmean(best_curve)
        ),
        "compliance_target": compliance_target,
        "first_bo_iteration_at_or_below_target": first_target,
        "total_ranking_runtime_seconds": ranking_runtime,
        "total_optimizer_runtime_seconds": optimizer_runtime,
        "htype": int(run.configuration["htype"]),
        "nholes": int(run.nholes),
        "ng_rst": int(run.configuration["ng_rst"]),
        "nl_rst": int(run.configuration["nl_rst"]),
        "adaptive_restart_allocation": bool(
            run.configuration.get("adaptive_restart_allocation", True)
        ),
        "global_restart_selection_strategy": run.configuration.get(
            "global_restart_selection_strategy", "greedy_dynamic_novelty"
        ),
    }


def analyze_sequential_weight_confirmation(
    run_directories_by_fraction,
    output_directory,
    *,
    incumbent_acquisition_fraction=0.55,
    expected_seed_count=10,
    compliance_target=None,
):
    """Validate and summarize completed paired sequential BO runs."""
    fractions = _validate_fractions(run_directories_by_fraction)
    incumbent = float(incumbent_acquisition_fraction)
    if incumbent not in fractions:
        raise ValueError("Sequential runs must include the incumbent fraction.")
    if compliance_target is not None and (
        not math.isfinite(compliance_target) or compliance_target <= 0
    ):
        raise ValueError("compliance_target must be finite and positive, or None.")
    loaded = {}
    for fraction, directories in run_directories_by_fraction.items():
        if len(directories) != int(expected_seed_count):
            raise ValueError(
                f"Fraction {fraction} has {len(directories)} seeds; "
                f"expected {expected_seed_count}."
            )
        group = {
            int(seed): load_archived_run(seed, path)
            for seed, path in directories.items()
        }
        if len(group) != len(directories):
            raise ValueError(f"Fraction {fraction} contains duplicate seeds.")
        loaded[float(fraction)] = group
    seed_sets = {tuple(sorted(group)) for group in loaded.values()}
    if len(seed_sets) != 1:
        raise ValueError("Every fraction must contain the same seeds.")
    seeds = next(iter(seed_sets))
    rows = []
    reference_group = loaded[incumbent]
    reference_scope = None
    reference_source_hashes = None
    for seed in seeds:
        reference = reference_group[seed]
        reference_config = _config_without_pairing_fields(reference.configuration)
        scope = _config_without_seed_or_pairing_fields(reference.configuration)
        if reference_scope is None:
            reference_scope = scope
        elif scope != reference_scope:
            raise ValueError("Incumbent runs differ beyond seed and output directory.")
        sources = reference.manifest.get("git", {}).get("source_sha256", {})
        if not sources:
            raise ValueError(f"Seed {seed} does not record source provenance.")
        if reference_source_hashes is None:
            reference_source_hashes = sources
        elif sources != reference_source_hashes:
            raise ValueError("Incumbent seeds used different source revisions.")
        burn_mask = reference.archive["bo_iteration"].reshape(-1) < 0
        reference_burn_x = reference.archive["x"][burn_mask]
        reference_burn_y = reference.archive["physical_compliance"][burn_mask]
        for fraction in fractions:
            run = loaded[fraction][seed]
            configured_fraction = float(run.configuration["acquisition_weight"]) / (
                float(run.configuration["acquisition_weight"])
                + float(run.configuration["novelty_weight"])
            )
            if not math.isclose(configured_fraction, fraction, abs_tol=1e-12):
                raise ValueError(
                    f"{run.directory} weight ratio does not match fraction {fraction}."
                )
            if _config_without_pairing_fields(run.configuration) != reference_config:
                raise ValueError(
                    f"Seed {seed} configurations differ beyond the two weights."
                )
            reference_sources = reference.manifest.get("git", {}).get(
                "source_sha256", {}
            )
            candidate_sources = run.manifest.get("git", {}).get(
                "source_sha256", {}
            )
            if not candidate_sources:
                raise ValueError(f"{run.directory} does not record source provenance.")
            if reference_sources != candidate_sources:
                raise ValueError(f"Seed {seed} runs used different source revisions.")
            if bool(run.configuration.get("adaptive_restart_allocation", True)):
                raise ValueError(
                    "Sequential weight confirmation requires "
                    "adaptive_restart_allocation=False."
                )
            if run.configuration.get("global_restart_selection_strategy") != (
                "greedy_dynamic_novelty"
            ):
                raise ValueError(
                    "Sequential weight confirmation requires the greedy "
                    "global selector."
                )
            candidate_burn_mask = run.archive["bo_iteration"].reshape(-1) < 0
            if not torch.equal(run.archive["x"][candidate_burn_mask], reference_burn_x):
                raise ValueError(f"Seed {seed} burn-in designs are not paired.")
            if not torch.equal(
                run.archive["physical_compliance"][candidate_burn_mask],
                reference_burn_y,
            ):
                raise ValueError(f"Seed {seed} burn-in responses are not paired.")
            if int((run.archive["bo_iteration"].reshape(-1) > 0).sum()) != int(
                (reference.archive["bo_iteration"].reshape(-1) > 0).sum()
            ):
                raise ValueError(f"Seed {seed} BO evaluation counts differ.")
            rows.append(_archive_summary(run, fraction, compliance_target))
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, SEQUENTIAL_FIELDS, output_directory / "sequential_results.csv")
    incumbent_rows = {
        int(row["seed"]): row
        for row in rows
        if math.isclose(row["acquisition_fraction"], incumbent)
    }
    paired = []
    for row in rows:
        reference = incumbent_rows[int(row["seed"])]
        difference = row["best_compliance"] - reference["best_compliance"]
        auc_difference = (
            None
            if row["best_so_far_compliance_auc"] is None
            or reference["best_so_far_compliance_auc"] is None
            else row["best_so_far_compliance_auc"]
            - reference["best_so_far_compliance_auc"]
        )
        paired.append(
            {
                **row,
                "incumbent_acquisition_fraction": incumbent,
                "incumbent_best_compliance": reference["best_compliance"],
                "candidate_minus_incumbent_compliance": difference,
                "candidate_minus_incumbent_relative_compliance": (
                    difference / reference["best_compliance"]
                ),
                "candidate_minus_incumbent_compliance_auc": auc_difference,
            }
        )
    fields = tuple(dict.fromkeys(key for row in paired for key in row))
    _write_rows(
        paired, fields, output_directory / "sequential_pairwise_results.csv"
    )
    summary = []
    for fraction in fractions:
        group = [row for row in paired if row["acquisition_fraction"] == fraction]
        relative = [
            row["candidate_minus_incumbent_relative_compliance"] for row in group
        ]
        summary.append(
            {
                "acquisition_fraction": fraction,
                "novelty_fraction": _complementary_fraction(fraction),
                "seed_count": len(group),
                "mean_best_compliance": statistics.fmean(
                    row["best_compliance"] for row in group
                ),
                "median_best_compliance": statistics.median(
                    row["best_compliance"] for row in group
                ),
                "worst_seed_best_compliance": max(
                    row["best_compliance"] for row in group
                ),
                "mean_relative_difference_vs_incumbent": statistics.fmean(relative),
                "median_relative_difference_vs_incumbent": statistics.median(relative),
                "win_rate_vs_incumbent": sum(value < 0 for value in relative)
                / len(relative),
                "mean_best_so_far_compliance_auc": _mean(
                    [row["best_so_far_compliance_auc"] for row in group]
                ),
                "mean_total_ranking_runtime_seconds": _mean(
                    [row["total_ranking_runtime_seconds"] for row in group]
                ),
                "mean_total_optimizer_runtime_seconds": _mean(
                    [row["total_optimizer_runtime_seconds"] for row in group]
                ),
                "target_attainment_rate": (
                    None
                    if compliance_target is None
                    else sum(
                        row["first_bo_iteration_at_or_below_target"] is not None
                        for row in group
                    )
                    / len(group)
                ),
                "median_first_bo_iteration_at_or_below_target": (
                    None
                    if compliance_target is None
                    else (
                        statistics.median(
                            row["first_bo_iteration_at_or_below_target"]
                            for row in group
                            if row["first_bo_iteration_at_or_below_target"]
                            is not None
                        )
                        if any(
                            row["first_bo_iteration_at_or_below_target"]
                            is not None
                            for row in group
                        )
                        else None
                    )
                ),
            }
        )
    _write_rows(
        summary,
        tuple(dict.fromkeys(key for row in summary for key in row)),
        output_directory / "sequential_summary_by_weight.csv",
    )
    atomic_json_write(
        {
            "schema_version": STUDY_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "incumbent_acquisition_fraction": incumbent,
            "seed_count": len(seeds),
            "fractions": list(fractions),
            "compliance_target": compliance_target,
            "pairing": (
                "exact burn-in vectors and compliance; all non-weight config equal"
            ),
            "status": "complete",
        },
        output_directory / "sequential_study_manifest.json",
    )
    return summary


def _bootstrap_mean_interval(values, *, samples=10_000, seed=20_261_019):
    if len(values) < 2:
        return None, None
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(int(samples))
    )
    lower = means[max(0, int(0.025 * len(means)))]
    upper = means[min(len(means) - 1, int(0.975 * len(means)))]
    return lower, upper


def decide_confirmed_global_restart_weights(
    sequential_study_directory,
    *,
    practical_relative_improvement,
    incumbent_acquisition_fraction=0.55,
    minimum_seed_count=10,
    bootstrap_samples=10_000,
):
    """Select only a challenger with independently confirmed useful improvement."""
    if practical_relative_improvement is None:
        raise ValueError("A predeclared practical improvement is required.")
    margin = float(practical_relative_improvement)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("practical_relative_improvement must be non-negative.")
    directory = Path(sequential_study_directory)
    rows = _read_rows(directory / "sequential_pairwise_results.csv")
    if not rows:
        raise FileNotFoundError("No sequential_pairwise_results.csv was found.")
    incumbent = float(incumbent_acquisition_fraction)
    assessments = []
    fractions = sorted(
        {
            float(row["acquisition_fraction"])
            for row in rows
            if not math.isclose(float(row["acquisition_fraction"]), incumbent)
        }
    )
    for fraction in fractions:
        values = [
            float(row["candidate_minus_incumbent_relative_compliance"])
            for row in rows
            if math.isclose(float(row["acquisition_fraction"]), fraction)
        ]
        lower, upper = _bootstrap_mean_interval(
            values,
            samples=bootstrap_samples,
            seed=20_261_019 + int(round(fraction * 10_000)),
        )
        confirmed = (
            len(values) >= int(minimum_seed_count)
            and upper is not None
            and upper < -margin
        )
        assessments.append(
            {
                "acquisition_weight": fraction,
                "novelty_weight": _complementary_fraction(fraction),
                "seed_count": len(values),
                "mean_relative_compliance_difference": statistics.fmean(values),
                "bootstrap_95_percent_interval": [lower, upper],
                "confirmed_practical_improvement": confirmed,
            }
        )
    confirmed = [row for row in assessments if row["confirmed_practical_improvement"]]
    if confirmed:
        selected = min(
            confirmed, key=lambda row: row["mean_relative_compliance_difference"]
        )
        status = "confirmed_challenger"
        recommendation = {
            "acquisition_weight": selected["acquisition_weight"],
            "novelty_weight": selected["novelty_weight"],
        }
    else:
        status = "insufficient_evidence"
        recommendation = {
            "acquisition_weight": incumbent,
            "novelty_weight": _complementary_fraction(incumbent),
        }
    decision = {
        "status": status,
        "recommended_weights": recommendation,
        "incumbent_retained_unless_challenger_confirmed": True,
        "practical_relative_improvement_required": margin,
        "minimum_seed_count": int(minimum_seed_count),
        "bootstrap_samples": int(bootstrap_samples),
        "candidate_assessments": assessments,
        "automatic_runtime_default_change": False,
    }
    atomic_json_write(decision, directory / "confirmed_weight_decision.json")
    return decision


def _parse_seed_paths(values):
    result = {}
    for value in values:
        seed, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"Expected SEED=PATH, received {value!r}.")
        result[int(seed)] = Path(path)
    return result


def _parse_fraction_seed_paths(values):
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or "," not in key:
            raise ValueError(
                f"Expected ACQUISITION_FRACTION,SEED=PATH, received {value!r}."
            )
        fraction, seed = key.split(",", maxsplit=1)
        result.setdefault(float(fraction), {})[int(seed)] = Path(path)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run", action="append", required=True)
    run.add_argument("--numerical-study-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--fit-after", nargs="+", type=int, default=DEFAULT_FIT_AFTER)
    run.add_argument(
        "--acquisition-fraction",
        nargs="+",
        type=float,
        default=DEFAULT_ACQUISITION_FRACTIONS,
    )
    run.add_argument(
        "--global-budget", nargs="+", type=int, default=DEFAULT_GLOBAL_BUDGETS
    )
    run.add_argument("--primary-global-budget", type=int, default=10)
    run.add_argument("--raw-pool-multiplier", type=int, default=3)
    run.add_argument("--expected-seed-count", type=int, default=5)
    run.add_argument("--reference-log-ei-tolerance", type=float, default=1e-8)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("study_directory")
    summarize.add_argument("--no-plots", action="store_true")
    recommend = subparsers.add_parser("recommend")
    recommend.add_argument("study_directory")
    analyze = subparsers.add_parser("analyze-sequential")
    analyze.add_argument("--run", action="append", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--incumbent-acquisition-fraction", type=float, default=0.55)
    analyze.add_argument("--expected-seed-count", type=int, default=10)
    analyze.add_argument("--compliance-target", type=float)
    decide = subparsers.add_parser("decide-sequential")
    decide.add_argument("study_directory")
    decide.add_argument("--practical-relative-improvement", type=float, required=True)
    decide.add_argument("--minimum-seed-count", type=int, default=10)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        rows = run_global_restart_weight_study(
            _parse_seed_paths(arguments.run),
            arguments.numerical_study_dir,
            arguments.output_dir,
            fit_after=arguments.fit_after,
            acquisition_fractions=arguments.acquisition_fraction,
            global_restart_budgets=arguments.global_budget,
            primary_global_restart_budget=arguments.primary_global_budget,
            raw_pool_multiplier=arguments.raw_pool_multiplier,
            expected_seed_count=arguments.expected_seed_count,
            compliance_target=arguments.compliance_target,
            reference_log_ei_tolerance=arguments.reference_log_ei_tolerance,
        )
        print(f"Recorded {len(rows)} frozen weight results")
    elif arguments.command == "summarize":
        rows = summarize_global_restart_weight_study(
            arguments.study_directory, make_plots=not arguments.no_plots
        )
        print(f"Summarized {len(rows)} weight/budget combinations")
    elif arguments.command == "recommend":
        print(
            json.dumps(
                recommend_global_restart_weights(arguments.study_directory),
                indent=2,
            )
        )
    elif arguments.command == "analyze-sequential":
        rows = analyze_sequential_weight_confirmation(
            _parse_fraction_seed_paths(arguments.run),
            arguments.output_dir,
            incumbent_acquisition_fraction=(
                arguments.incumbent_acquisition_fraction
            ),
            expected_seed_count=arguments.expected_seed_count,
        )
        print(f"Summarized {len(rows)} sequential weight groups")
    else:
        print(
            json.dumps(
                decide_confirmed_global_restart_weights(
                    arguments.study_directory,
                    practical_relative_improvement=(
                        arguments.practical_relative_improvement
                    ),
                    minimum_seed_count=arguments.minimum_seed_count,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
