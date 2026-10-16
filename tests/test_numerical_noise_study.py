import inspect

import torch

from one_shape_bo.gaussian import build_objective_gp
from one_shape_bo.numerical_noise_study import (
    ArchivedRun,
    FIT_FIELDS,
    apply_controlled_initialization,
    construct_cold_study_model,
    decide_provisional_fixed_variance,
    snapshot_from_archive,
    standardized_to_raw_noise_variance,
)
from one_shape_bo.records import atomic_csv_write


def make_archived_run():
    x = torch.arange(35, dtype=torch.double).reshape(5, 7) / 35
    compliance = torch.tensor([[10.0], [12.0], [9.0], [8.0], [7.0]])
    return ArchivedRun(
        seed=0,
        directory=None,
        archive={
            "x": x,
            "physical_compliance": compliance,
            "gp_response_prestandardization": -torch.log(compliance),
            "bo_iteration": torch.tensor([-1, -1, 1, 2, 3]),
            "feasible": torch.ones(5, dtype=torch.bool),
        },
        configuration={"seed": 0, "nholes": 1, "htype": 0},
        manifest={},
        archive_sha256="archive",
    )


def test_snapshot_uses_burn_in_and_requested_bo_prefix_then_future_holdout():
    snapshot = snapshot_from_archive(make_archived_run(), fit_after=1, holdout_size=2)

    assert snapshot["archive_stop_exclusive"] == 3
    assert len(snapshot["train_x"]) == 3
    assert snapshot["holdout_start"] == 2
    assert snapshot["holdout_stop"] == 3
    assert torch.equal(snapshot["holdout_x"], make_archived_run().archive["x"][3:])


def test_standardized_noise_conversion_uses_training_response_sample_variance():
    score = torch.tensor([[-2.0], [-1.0], [1.0]], dtype=torch.double)

    raw = standardized_to_raw_noise_variance(1e-5, score)

    assert raw == 1e-5 * torch.var(score, correction=1).item()


def test_controlled_initializations_are_reproducible_distinct_and_paired():
    generator = torch.Generator().manual_seed(4)
    x = torch.rand((6, 7), generator=generator, dtype=torch.double)
    x[:, 0] = 0.0
    y = torch.linspace(100.0, 150.0, 6, dtype=torch.double).unsqueeze(-1)

    learned_a, _ = build_objective_gp(x, y, nholes=1)
    learned_b, _ = build_objective_gp(x, y, nholes=1)
    fixed, _ = build_objective_gp(
        x, y, nholes=1, observation_noise_variance=1e-4
    )
    different, _ = build_objective_gp(x, y, nholes=1)
    first = apply_controlled_initialization(
        learned_a, seed=76, fit_after=10, initialization_id=1
    )
    repeat = apply_controlled_initialization(
        learned_b, seed=76, fit_after=10, initialization_id=1
    )
    paired = apply_controlled_initialization(
        fixed, seed=76, fit_after=10, initialization_id=1
    )
    other = apply_controlled_initialization(
        different, seed=76, fit_after=10, initialization_id=2
    )

    assert first["initial_full_hyperparameter_sha256"] == repeat[
        "initial_full_hyperparameter_sha256"
    ]
    assert first["initial_kernel_mean_sha256"] == paired[
        "initial_kernel_mean_sha256"
    ]
    assert first["initial_kernel_mean_sha256"] != other[
        "initial_kernel_mean_sha256"
    ]


def _decision_rows():
    rows = []
    candidates = {
        "learned": (1.0, 1.0),
        "fixed_1e-06": (1.1, 1.2),
        "fixed_1e-05": (0.9, 0.8),
        "fixed_1e-04": (1.2, 1.3),
    }
    for candidate, (rmse, nlpd) in candidates.items():
        for seed in (0, 1):
            for fit_after in (0, 5):
                for initialization in range(3):
                    row = {field: "N/A" for field in FIT_FIELDS}
                    row.update(
                        {
                            "fit_id": (
                                f"{candidate}-{seed}-{fit_after}-{initialization}"
                            ),
                            "seed": seed,
                            "fit_after_bo_iteration": fit_after,
                            "noise_candidate": candidate,
                            "status": "success",
                            "warning_count": 0,
                            "heldout_rmse": rmse,
                            "heldout_mae": rmse,
                            "heldout_negative_log_predictive_density": nlpd,
                            "heldout_rms_standardized_residual": 1.0,
                            "heldout_95_percent_coverage": 0.95,
                            "probe_mean_spread_ratio": 0.5,
                            "probe_latent_sd_ratio_q90": 0.5,
                            "effective_standardized_noise_variance": (
                                "N/A"
                                if candidate == "learned"
                                else float(candidate.removeprefix("fixed_"))
                            ),
                        }
                    )
                    rows.append(row)
    return rows


def test_decision_selects_eligible_best_predictive_fixed_candidate(tmp_path):
    rows = _decision_rows()
    atomic_csv_write(
        rows, FIT_FIELDS, tmp_path / "gp_fits_numerical_study.csv"
    )

    decision = decide_provisional_fixed_variance(tmp_path)

    assert decision["status"] == "provisional_selection"
    assert decision["selected_standardized_noise_variance"] == 1e-5
    assert decision["learned_noise_remains_control"] is True
    assert (tmp_path / "decision.json").exists()


def test_fit_csv_schema_explicitly_records_cold_start_provenance():
    assert "initialization_type" in FIT_FIELDS
    assert "warm_start_used" in FIT_FIELDS
    assert "warm_start_source_fit_id" in FIT_FIELDS


def test_study_model_constructor_cannot_accept_a_warm_start_source():
    parameters = inspect.signature(construct_cold_study_model).parameters

    assert "previous_model" not in parameters
    assert "warm_start_policy" not in parameters
