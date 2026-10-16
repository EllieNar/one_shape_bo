import csv
import json
import random
import warnings

import numpy as np
import pytest
import torch

from one_shape_bo.records import (
    RunRecorder,
    atomic_json_write,
    capture_rng_state,
    load_evaluation_archive,
    load_fit_state,
    reconstruct_training_set,
    restore_rng_state,
)


def make_recorder(tmp_path):
    return RunRecorder(
        tmp_path,
        {"seed": 7, "output_dir": str(tmp_path)},
        {"burn_in": 2, "iterations": 2, "dimension": 2},
        tmp_path,
    )


def append_burn_in(recorder):
    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.double)
    compliance = torch.tensor([[10.0], [20.0]], dtype=torch.double)
    response = -torch.log(compliance)
    recorder.append_evaluations(
        x,
        compliance,
        response,
        torch.tensor([0.50, 0.49], dtype=torch.double),
        [b"aa", b"bb"],
        torch.tensor([True, True]),
        torch.tensor([True, True]),
        phase="burn-in",
        sources="burn-in",
        burn_in_indices=[1, 2],
    )
    recorder.record_burn_in_summary()
    return x, compliance, response


def test_archive_round_trip_alignment_and_training_reconstruction(tmp_path):
    recorder = make_recorder(tmp_path)
    burn_x, burn_y, burn_score = append_burn_in(recorder)
    rng = capture_rng_state(torch.Generator().manual_seed(7))
    first_fit_index = recorder.record_fit(
        bo_iteration=1,
        train_x=burn_x,
        train_compliance=burn_y,
        train_gp_response=burn_score,
        rng_before=rng,
        rng_after=rng,
        warnings_recorded=[],
        model_state={"weight": torch.tensor([1.25], dtype=torch.double)},
        diagnostics={
            "lengthscales": {"kernel": [0.3]},
            "outputscales": {"kernel": 1.2},
            "noise": 0.01,
        },
        transform_parameters={"mean": [-2.5], "std": [0.5]},
        initialization_metadata={
            "initialization_type": "cold",
            "warm_start_source_fit_index": None,
            "warm_start_state_reference": None,
            "inherited_lengthscales": None,
            "inherited_outputscales": None,
            "inherited_likelihood_noise_variance": None,
            "inherited_mean_module_state": None,
        },
    )
    recorder.update_fit_health(
        first_fit_index,
        {
            "noise_mode": "learned",
            "prescribed_observation_noise_variance": None,
            "signal_variance": 0.001,
            "observation_noise_variance": 1.0,
            "signal_to_noise_ratio": 0.001,
            "signal_fraction_nu": 0.001 / 1.001,
            "representative_point_count": 4,
            "posterior_mean_spread_ratio": 0.4,
            "posterior_std_ratio": 0.2,
            "withheld_predictive_mean": -2.0,
            "withheld_predictive_std": 0.1,
            "withheld_observed_response": -2.1,
            "withheld_standardized_residual": -1.0,
            "predictive_window_count": 1,
            "predictive_rms_z": 1.0,
            "health_warning_triggered": True,
            "health_warning_reasons": ["signal_fraction_nu_below_threshold"],
            "health_action": "cold_fit_next_iteration",
            "next_fit_forced_cold": True,
        },
    )
    bo_x = torch.tensor([[0.5, 0.6]], dtype=torch.double)
    bo_y = torch.tensor([[8.0]], dtype=torch.double)
    bo_score = -torch.log(bo_y)
    recorder.append_evaluations(
        bo_x,
        bo_y,
        bo_score,
        torch.tensor([0.48], dtype=torch.double),
        [b"cc"],
        torch.tensor([True]),
        torch.tensor([True]),
        phase="bo",
        sources=["global"],
        bo_iterations=[1],
    )
    recorder.record_fit(
        bo_iteration=2,
        train_x=torch.cat((burn_x, bo_x)),
        train_compliance=torch.cat((burn_y, bo_y)),
        train_gp_response=torch.cat((burn_score, bo_score)),
        rng_before=rng,
        rng_after=rng,
        warnings_recorded=[],
        model_state={"weight": torch.tensor([1.5], dtype=torch.double)},
        diagnostics={
            "lengthscales": {"kernel": [0.4]},
            "outputscales": {"kernel": 1.3},
            "noise": 0.02,
        },
        transform_parameters={"mean": [-2.4], "std": [0.6]},
        initialization_metadata={
            "initialization_type": "warm",
            "warm_start_source_fit_index": 1,
            "warm_start_state_reference": "fit_states/fit_0001.pt:model_state",
            "inherited_lengthscales": {"kernel": [0.3]},
            "inherited_outputscales": {"kernel": 1.2},
            "inherited_likelihood_noise_variance": 0.01,
            "inherited_mean_module_state": {"raw_constant": [0.25]},
        },
    )

    archive = load_evaluation_archive(tmp_path)
    before = reconstruct_training_set(tmp_path, 1, "before")
    after = reconstruct_training_set(tmp_path, 1, "after")
    fit = load_fit_state(tmp_path, 1)

    assert torch.equal(archive["x"][:2], burn_x)
    assert torch.equal(archive["physical_compliance"][:2], burn_y)
    assert torch.equal(archive["gp_response_prestandardization"][:2], burn_score)
    assert archive["burn_in_index"].tolist() == [1, 2, -1]
    assert archive["bo_iteration"].tolist() == [-1, -1, 1]
    assert torch.equal(before["x"], burn_x)
    assert torch.equal(after["x"], torch.cat((burn_x, bo_x)))
    assert torch.equal(after["physical_compliance"], torch.cat((burn_y, bo_y)))
    assert before["archive_stop_exclusive"] == 2
    assert after["archive_stop_exclusive"] == 3
    assert fit["training_archive_reference"]["stop_exclusive"] == 2
    assert torch.equal(
        fit["model_state"]["weight"], torch.tensor([1.25], dtype=torch.double)
    )
    assert fit["health"]["signal_variance"] == 0.001

    with (tmp_path / "evaluations.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["evaluation_number"] for row in rows] == ["1", "2", "3"]
    assert rows[0]["burn_in_index"] == "1"
    assert rows[0]["bo_iteration"] == "N/A"
    assert rows[2]["burn_in_index"] == "N/A"
    assert rows[2]["bo_iteration"] == "1"
    assert json.loads(rows[0]["encoded_design_vector"]) == burn_x[0].tolist()
    assert len(rows[0]["design_vector_sha256"]) == 64
    with (tmp_path / "gp_fits.csv").open(newline="") as stream:
        fit_rows = list(csv.DictReader(stream))
    assert fit_rows[0]["initialization_type"] == "cold"
    assert fit_rows[0]["warm_start_source_fit_index"] == "N/A"
    assert fit_rows[0]["health_warning_triggered"] == "True"
    assert json.loads(fit_rows[0]["health_warning_reasons"]) == [
        "signal_fraction_nu_below_threshold"
    ]
    assert json.loads(fit_rows[0]["post_fit_lengthscales"]) == {"kernel": [0.3]}
    assert fit_rows[1]["initialization_type"] == "warm"
    assert fit_rows[1]["warm_start_source_fit_index"] == "1"
    assert fit_rows[1]["warm_start_state_reference"] == (
        "fit_states/fit_0001.pt:model_state"
    )
    assert json.loads(fit_rows[1]["inherited_lengthscales"]) == {
        "kernel": [0.3]
    }
    summary = json.loads((tmp_path / "burn_in_summary.json").read_text())
    assert summary["best"]["burn_in_index"] == 1
    assert summary["worst"]["burn_in_index"] == 2
    assert summary["raw_compliance_sample_variance"] == 50.0


def test_rng_state_round_trip_reproduces_all_streams():
    generator = torch.Generator().manual_seed(19)
    original = capture_rng_state(generator)
    try:
        state = capture_rng_state(generator)
        expected = (
            torch.rand(3, generator=generator),
            torch.rand(3),
            np.random.random(3),
            [random.random() for _ in range(3)],
        )
        restore_rng_state(state, generator)
        actual = (
            torch.rand(3, generator=generator),
            torch.rand(3),
            np.random.random(3),
            [random.random() for _ in range(3)],
        )
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        assert np.array_equal(actual[2], expected[2])
        assert actual[3] == expected[3]
    finally:
        restore_rng_state(original, generator)


def test_fit_failure_and_warning_are_persisted_with_not_applicable_values(tmp_path):
    recorder = make_recorder(tmp_path)
    x, y, score = append_burn_in(recorder)
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("unstable fit", RuntimeWarning)
    rng = capture_rng_state(torch.Generator().manual_seed(7))
    recorder.record_fit(
        bo_iteration=1,
        train_x=x,
        train_compliance=y,
        train_gp_response=score,
        rng_before=rng,
        rng_after=rng,
        warnings_recorded=caught,
        failure={"type": "RuntimeError", "message": "fit failed"},
    )

    records = json.loads((tmp_path / "gp_fits.json").read_text())
    assert records[0]["status"] == "failed"
    assert records[0]["warning_count"] == 1
    assert records[0]["likelihood_observation_noise_variance"] is None
    assert records[0]["not_applicable"]["standardization_mean"] == "GP fit failed"


def test_evaluation_phase_requires_matching_indices(tmp_path):
    recorder = make_recorder(tmp_path)
    values = torch.tensor([[0.1, 0.2]], dtype=torch.double)
    with pytest.raises(ValueError, match="Burn-in records require"):
        recorder.append_evaluations(
            values,
            torch.tensor([[10.0]]),
            torch.tensor([[-2.0]]),
            torch.tensor([0.5]),
            [b"aa"],
            torch.tensor([True]),
            torch.tensor([True]),
            phase="burn-in",
            sources="burn-in",
        )


def test_atomic_json_failure_preserves_previous_target(monkeypatch, tmp_path):
    target = tmp_path / "record.json"
    atomic_json_write({"version": 1}, target)

    def fail_replace(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr("one_shape_bo.records.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        atomic_json_write({"version": 2}, target)

    assert json.loads(target.read_text()) == {"version": 1}
