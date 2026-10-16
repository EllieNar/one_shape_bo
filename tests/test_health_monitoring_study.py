import json

import torch

from one_shape_bo.health_monitoring_study import (
    REPLAY_FIELDS,
    _resolve_selected_variance,
    _training_prefix_before,
    calibrate_health_thresholds,
)
from one_shape_bo.numerical_noise_study import ArchivedRun
from one_shape_bo.records import atomic_csv_write, atomic_json_write


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


def test_training_prefix_targets_exact_next_archived_evaluation():
    data = _training_prefix_before(make_archived_run(), bo_iteration=2)

    assert len(data["x"]) == 3
    assert torch.equal(data["target_x"], make_archived_run().archive["x"][3:4])
    assert torch.equal(
        data["target_response"],
        make_archived_run().archive["gp_response_prestandardization"][3:4],
    )


def test_selected_variance_is_read_from_numerical_study_decision(tmp_path):
    atomic_json_write(
        {"status": "complete"},
        tmp_path / "study_manifest.json",
    )
    atomic_json_write(
        {
            "status": "provisional_selection",
            "selected_standardized_noise_variance": 1e-6,
        },
        tmp_path / "decision.json",
    )

    value, source = _resolve_selected_variance(tmp_path, None)

    assert value == 1e-6
    assert source.endswith("decision.json")


def _synthetic_replay_rows():
    rows = []
    for seed in range(5):
        for iteration in range(1, 9):
            adverse = iteration >= 6
            row = {field: "N/A" for field in REPLAY_FIELDS}
            row.update(
                {
                    "seed": seed,
                    "bo_iteration": iteration,
                    "status": "success",
                    "warning_count": 0,
                    "signal_variance": 0.01 if adverse else 1.0,
                    "posterior_mean_spread_ratio": 0.02 if adverse else 0.5,
                    "posterior_std_ratio": 0.01 if adverse else 0.4,
                    "one_step_standardized_residual": 4.0 if adverse else 0.2,
                }
            )
            rows.append(row)
    return rows


def test_calibration_recommends_supported_thresholds_across_seeds(tmp_path):
    atomic_json_write(
        {
            "selected_standardized_noise_variance": 1e-6,
            "predictive_windows": [2, 3],
            "representative_probe": "test probe",
        },
        tmp_path / "study_config.json",
    )
    atomic_csv_write(
        _synthetic_replay_rows(),
        REPLAY_FIELDS,
        tmp_path / "gp_health_replay.csv",
    )

    result = calibrate_health_thresholds(tmp_path)

    assert result["status"] == "provisional"
    assert result["health_nu_min"] is None
    assert result["health_signal_variance_min"] is not None
    assert result["health_mean_spread_min"] is not None
    assert result["health_posterior_std_min"] is not None
    assert result["health_predictive_z_max"] is not None
    assert result["health_predictive_window"] in (2, 3)
    assert "health_nu_min" not in result["optimization_config_overrides"]
    saved = json.loads((tmp_path / "recommended_health_config.json").read_text())
    assert saved["automatic_runtime_default_change"] is False
