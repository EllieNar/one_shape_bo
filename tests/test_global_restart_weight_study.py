import json

import torch

from one_shape_bo import global_restart_weight_study as study
from one_shape_bo.records import atomic_json_write, atomic_torch_save


class Constraints:
    box_bounds = torch.stack(
        (torch.zeros(7, dtype=torch.double), torch.ones(7, dtype=torch.double))
    )


def test_weight_selection_uses_global_pool_only_and_complementary_weights(monkeypatch):
    pool = torch.rand(
        (4, 7), generator=torch.Generator().manual_seed(4), dtype=torch.double
    )
    observed = pool[:1].clone()
    captured = {}

    def fake_rank(acquisition, global_conditions, local_conditions, *args, **kwargs):
        captured.update(kwargs)
        captured["global"] = global_conditions
        captured["local"] = local_conditions
        return [
            {
                "point": global_conditions[index],
                "source": "global",
                "global_selection_step": index + 1,
            }
            for index in range(len(global_conditions))
        ]

    monkeypatch.setattr(study, "rank_initial_condition_pool", fake_rank)
    monkeypatch.setattr(study, "geometry_signature", lambda point, constraints: b"x")

    selected, _ = study._select_for_fraction(
        object(), pool, observed, Constraints(), 2, 0.65, 0.02
    )

    assert len(selected) == 2
    assert captured["local"].shape == (0, 7)
    assert torch.equal(captured["global"], pool)
    assert captured["acquisition_weight"] == 0.65
    assert captured["novelty_weight"] == 0.35
    assert captured["number_global"] == 2
    assert captured["global_selection_strategy"] == "greedy_dynamic_novelty"


def _frozen_rows():
    rows = []
    for seed in range(5):
        for snapshot in (0, 5):
            for fraction, value in ((0.55, 4.0), (0.7, 5.0)):
                row = {field: None for field in study.RESULT_FIELDS}
                row.update(
                    {
                        "seed": seed,
                        "fit_after_bo_iteration": snapshot,
                        "global_restart_budget": 10,
                        "acquisition_fraction": fraction,
                        "novelty_fraction": 1.0 - fraction,
                        "requested_global_restarts": 10,
                        "actual_global_restart_attempts": 10,
                        "global_restart_shortfall": 0,
                        "optimizer_failure_count": 0,
                        "infeasible_endpoint_count": 0,
                        "duplicate_observed_endpoint_count": 0,
                        "duplicate_within_weight_count": 0,
                        "unique_feasible_endpoint_count": 10,
                        "best_feasible_log_ei": value,
                        "selection_runtime_seconds": 0.1,
                        "optimizer_runtime_seconds": 2.0,
                        "start_pairwise_distance_mean": 0.4,
                    }
                )
                rows.append(row)
    return rows


def test_summary_clusters_snapshots_by_seed_and_recommendation_is_screening(tmp_path):
    atomic_json_write(
        {
            "primary_global_restart_budget": 10,
            "incumbent_acquisition_fraction": 0.55,
        },
        tmp_path / "study_config.json",
    )
    study._write_rows(
        _frozen_rows(), study.RESULT_FIELDS, tmp_path / "weight_results.csv"
    )

    summary = study.summarize_global_restart_weight_study(
        tmp_path, make_plots=False
    )
    recommendation = study.recommend_global_restart_weights(tmp_path)

    candidate = next(row for row in summary if row["acquisition_fraction"] == 0.7)
    incumbent = next(row for row in summary if row["acquisition_fraction"] == 0.55)
    assert candidate["seed_count"] == 5
    assert candidate["mean_seed_paired_log_ei_regret"] == 0.0
    assert incumbent["mean_seed_paired_log_ei_regret"] == 1.0
    assert recommendation["status"] == "provisional_screening"
    assert recommendation["challengers"] == [
        {"acquisition_weight": 0.7, "novelty_weight": 0.3}
    ]
    assert recommendation["automatic_runtime_default_change"] is False
    assert recommendation["requires_independent_sequential_confirmation"] is True


def _sequential_rows(candidate_differences):
    rows = []
    for seed, difference in enumerate(candidate_differences):
        for fraction, relative in ((0.55, 0.0), (0.7, difference)):
            rows.append(
                {
                    "seed": seed,
                    "acquisition_fraction": fraction,
                    "candidate_minus_incumbent_relative_compliance": relative,
                }
            )
    return rows


def test_confirmed_decision_requires_practical_improvement(tmp_path):
    rows = _sequential_rows([-0.05] * 10)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    study._write_rows(rows, fields, tmp_path / "sequential_pairwise_results.csv")

    decision = study.decide_confirmed_global_restart_weights(
        tmp_path,
        practical_relative_improvement=0.01,
        bootstrap_samples=500,
    )

    assert decision["status"] == "confirmed_challenger"
    assert decision["recommended_weights"]["acquisition_weight"] == 0.7
    assert decision["automatic_runtime_default_change"] is False


def test_inconclusive_sequential_result_retains_incumbent(tmp_path):
    rows = _sequential_rows([-0.03, 0.03] * 5)
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    study._write_rows(rows, fields, tmp_path / "sequential_pairwise_results.csv")

    decision = study.decide_confirmed_global_restart_weights(
        tmp_path,
        practical_relative_improvement=0.01,
        bootstrap_samples=500,
    )

    assert decision["status"] == "insufficient_evidence"
    assert decision["recommended_weights"] == {
        "acquisition_weight": 0.55,
        "novelty_weight": 0.45,
    }


def _write_sequential_run(directory, seed, fraction, bo_compliance):
    directory.mkdir(parents=True)
    atomic_json_write(
        {
            "seed": seed,
            "nholes": 1,
            "htype": 0,
            "ng_rst": 2,
            "nl_rst": 0,
            "adaptive_restart_allocation": False,
            "global_restart_selection_strategy": "greedy_dynamic_novelty",
            "acquisition_weight": fraction,
            "novelty_weight": 1.0 - fraction,
            "output_dir": str(directory),
        },
        directory / "run_config.json",
    )
    atomic_json_write(
        {
            "run_id": f"seed-{seed}-a-{fraction}",
            "git": {"source_sha256": {"src/one_shape_bo/sampling.py": "same"}},
        },
        directory / "run_manifest.json",
    )
    compliance = torch.tensor(
        [[10.0 + seed], [bo_compliance], [bo_compliance + 1.0]],
        dtype=torch.double,
    )
    atomic_torch_save(
        {
            "x": torch.arange(21, dtype=torch.double).reshape(3, 7) / 21,
            "physical_compliance": compliance,
            "gp_response_prestandardization": -torch.log(compliance),
            "bo_iteration": torch.tensor([-1, 1, 2]),
            "feasible": torch.ones(3, dtype=torch.bool),
        },
        directory / "evaluation_archive.pt",
    )


def test_sequential_analysis_requires_and_accepts_exact_paired_burn_in(tmp_path):
    run_map = {0.55: {}, 0.7: {}}
    for seed in (0, 1):
        for fraction in run_map:
            directory = tmp_path / f"a{fraction}" / f"seed{seed}"
            _write_sequential_run(
                directory,
                seed,
                fraction,
                bo_compliance=8.0 - (0.5 if fraction == 0.7 else 0.0),
            )
            run_map[fraction][seed] = directory

    summary = study.analyze_sequential_weight_confirmation(
        run_map,
        tmp_path / "analysis",
        expected_seed_count=2,
    )

    challenger = next(row for row in summary if row["acquisition_fraction"] == 0.7)
    assert challenger["seed_count"] == 2
    assert challenger["mean_relative_difference_vs_incumbent"] < 0
    assert (tmp_path / "analysis" / "sequential_pairwise_results.csv").exists()
