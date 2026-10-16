import csv

import pytest
import torch

from one_shape_bo.plot_evaluation import (
    _print_result,
    plot_best_burn_in_geometry,
    plot_best_geometry,
    plot_evaluated_burn_in_geometry,
    plot_evaluated_geometry,
)
from one_shape_bo.records import RunRecorder


def _record_run(path):
    config = {"xsize": 12, "ysize": 6, "nholes": 1, "output_dir": str(path)}
    recorder = RunRecorder(
        path,
        config,
        {"burn_in": 2, "iterations": 2, "dimension": 7},
        path,
    )
    burn_x = torch.tensor(
        [
            [0.0, 0.3, 0.3, 0.1, 0.0, 0.0, 0.1],
            [0.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.1],
        ],
        dtype=torch.double,
    )
    burn_y = torch.tensor([[10.0], [5.0]], dtype=torch.double)
    recorder.append_evaluations(
        burn_x,
        burn_y,
        -torch.log(burn_y),
        torch.tensor([0.9, 0.9], dtype=torch.double),
        [b"aa", b"bb"],
        torch.tensor([True, True]),
        torch.tensor([True, True]),
        phase="burn-in",
        sources="burn-in",
        burn_in_indices=[1, 2],
    )
    for iteration, compliance in ((1, 7.0), (2, 4.0)):
        design = burn_x[0:1] + iteration * 0.01
        value = torch.tensor([[compliance]], dtype=torch.double)
        recorder.append_evaluations(
            design,
            value,
            -torch.log(value),
            torch.tensor([0.9], dtype=torch.double),
            [bytes([iteration, iteration])],
            torch.tensor([True]),
            torch.tensor([True]),
            phase="bo",
            sources="global",
            bo_iterations=[iteration],
        )


def test_plot_bo_iteration_uses_best_record_available_at_that_step(tmp_path):
    _record_run(tmp_path)
    result = plot_best_geometry(tmp_path, eval_number=1)

    assert result["record_number"] == 2
    assert result["physical_compliance"] == 5.0
    assert result["title"] == "Best after BO evaluation 1"
    assert result["output_path"].name == "Best_Geometry_eval_0001.png"
    assert result["output_path"].is_file()


def test_plot_burn_in_index_and_reject_missing_step(tmp_path):
    _record_run(tmp_path)
    output = tmp_path / "chosen.png"
    result = plot_best_burn_in_geometry(
        tmp_path, eval_number=1, output_path=output
    )

    assert result["record_number"] == 1
    assert result["title"] == "Best after burn-in evaluation 1"
    assert output.is_file()
    with pytest.raises(KeyError, match="No BO evaluation was recorded"):
        plot_best_geometry(tmp_path, eval_number=3)


def test_plot_requires_positive_evaluation_number(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        plot_best_geometry(tmp_path, eval_number=0)


def test_plot_exact_evaluation_returns_and_prints_csv_design(tmp_path, capsys):
    _record_run(tmp_path)
    result = plot_evaluated_geometry(tmp_path, eval_number=1)

    assert result["record_number"] == 3
    assert result["phase"] == "bo"
    assert result["bo_iteration"] == 1
    assert result["physical_compliance"] == 7.0
    assert result["encoded_design_vector"] == pytest.approx(
        [0.01, 0.31, 0.31, 0.11, 0.01, 0.01, 0.11]
    )
    assert result["output_path"].name == "Geometry_eval_0001.png"
    assert result["output_path"].is_file()

    _print_result(result)
    printed = capsys.readouterr().out
    assert "encoded design vector:" in printed
    assert "BO iteration: 1" in printed


def test_plot_exact_burn_in_evaluation_uses_burn_in_index(tmp_path):
    _record_run(tmp_path)
    result = plot_evaluated_burn_in_geometry(tmp_path, eval_number=2)

    assert result["phase"] == "burn-in"
    assert result["burn_in_index"] == 2
    assert result["physical_compliance"] == 5.0
    assert result["encoded_design_vector"] == pytest.approx(
        [0.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.1]
    )
    assert result["output_path"].name == "Burn_In_Geometry_eval_0002.png"
    assert result["output_path"].is_file()


def test_exact_evaluation_rejects_csv_vector_that_fails_checksum(tmp_path):
    _record_run(tmp_path)
    csv_path = tmp_path / "evaluations.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fields = list(rows[0])
    rows[0]["encoded_design_vector"] = "[0,0,0,0,0,0,0]"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="recorded checksum"):
        plot_evaluated_burn_in_geometry(tmp_path, eval_number=1)
