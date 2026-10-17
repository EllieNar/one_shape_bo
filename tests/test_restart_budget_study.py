import json

import torch

from one_shape_bo import restart_budget_study as study


class Constraints:
    box_bounds = torch.stack(
        (torch.zeros(7, dtype=torch.double), torch.ones(7, dtype=torch.double))
    )


def _entry(source, index, value):
    return {
        "start_id": f"{source}-{index}",
        "source": source,
        "source_prefix_index": index,
        "point": torch.full((7,), value, dtype=torch.double),
    }


def _outcome(entry, value, signature):
    return {
        "start_id": entry["start_id"],
        "source": entry["source"],
        "status": "success_feasible",
        "optimized_log_ei": value,
        "optimizer_runtime_seconds": 1.0,
        "endpoint_signature": signature,
    }


def test_selected_noise_must_come_from_provisional_decision(tmp_path):
    decision = {
        "status": "provisional_selection",
        "selected_standardized_noise_variance": 1e-5,
    }
    (tmp_path / "decision.json").write_text(json.dumps(decision))

    value, candidate, path = study._selected_noise(tmp_path)

    assert value == 1e-5
    assert candidate == "fixed_1e-05"
    assert path == tmp_path / "decision.json"


def test_allocations_are_nested_source_prefixes_and_report_actual_counts():
    global_starts = [_entry("global", index, index / 10) for index in range(1, 4)]
    local_starts = [_entry("local", index, 0.5 + index / 10) for index in range(1, 3)]
    entries = global_starts + local_starts
    outcomes = {
        entry["start_id"]: _outcome(
            entry,
            float(position),
            b"same" if position in (1, 2) else bytes([position]),
        )
        for position, entry in enumerate(entries, start=1)
    }

    small = study._allocation_row(
        0, 5, 2, 1, global_starts, local_starts, outcomes, Constraints()
    )
    large = study._allocation_row(
        0, 5, 3, 2, global_starts, local_starts, outcomes, Constraints()
    )

    assert small["actual_global_restart_attempts"] == 2
    assert small["actual_local_restart_attempts"] == 1
    assert small["duplicate_within_allocation_count"] == 1
    assert large["actual_total_restart_attempts"] == 5
    assert large["best_feasible_log_ei"] == 5.0
    assert large["winning_source"] == "local"


def test_recommendation_prefers_smallest_one_se_budget(tmp_path):
    rows = []
    for seed, best_small, best_large in ((0, 4.9, 5.0), (1, 5.0, 5.0)):
        rows.extend(
            [
                {
                    **{field: None for field in study.ALLOCATION_FIELDS},
                    "seed": seed,
                    "fit_after_bo_iteration": 0,
                    "requested_global_restarts": 3,
                    "requested_local_restarts": 2,
                    "requested_total_restarts": 5,
                    "requested_global_fraction": 0.6,
                    "actual_global_restart_attempts": 3,
                    "actual_local_restart_attempts": 2,
                    "actual_total_restart_attempts": 5,
                    "global_restart_shortfall": 0,
                    "local_restart_shortfall": 0,
                    "optimizer_failure_count": 0,
                    "duplicate_within_allocation_count": 0,
                    "duplicate_observed_endpoint_count": 0,
                    "best_feasible_log_ei": best_small,
                    "optimizer_runtime_seconds": 5.0,
                },
                {
                    **{field: None for field in study.ALLOCATION_FIELDS},
                    "seed": seed,
                    "fit_after_bo_iteration": 0,
                    "requested_global_restarts": 7,
                    "requested_local_restarts": 3,
                    "requested_total_restarts": 10,
                    "requested_global_fraction": 0.7,
                    "actual_global_restart_attempts": 7,
                    "actual_local_restart_attempts": 3,
                    "actual_total_restart_attempts": 10,
                    "global_restart_shortfall": 0,
                    "local_restart_shortfall": 0,
                    "optimizer_failure_count": 0,
                    "duplicate_within_allocation_count": 0,
                    "duplicate_observed_endpoint_count": 0,
                    "best_feasible_log_ei": best_large,
                    "optimizer_runtime_seconds": 10.0,
                },
            ]
        )
    study._write_rows(
        rows, study.ALLOCATION_FIELDS, tmp_path / "allocation_results.csv"
    )

    recommendation = study.recommend_restart_allocation(tmp_path)

    assert recommendation["status"] == "provisional"
    assert recommendation["ng_rst"] == 3
    assert recommendation["nl_rst"] == 2
    assert recommendation["adaptive_restart_allocation"] is False
    assert recommendation["local_start_strategy"] == "archive"
