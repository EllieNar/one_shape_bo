"""Plot best-so-far or exact geometries from a recorded optimization run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .physical import generate_geometry
from .records import _tensor_sha256, load_evaluation_archive


def _load_run_records(run_directory):
    run_directory = Path(run_directory)
    csv_path = run_directory / "evaluations.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation index not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    archive = load_evaluation_archive(run_directory)
    if len(rows) != len(archive["x"]):
        raise ValueError("evaluations.csv and evaluation_archive.pt are misaligned.")
    if any(
        int(row["evaluation_index"]) != expected
        for expected, row in enumerate(rows)
    ):
        raise ValueError("evaluations.csv does not follow archive row order.")

    config_path = run_directory / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return run_directory, rows, archive, config


def _render_geometry(
    design,
    row,
    config,
    output_path,
    title,
    *,
    show,
    include_design_vector=False,
):
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError("output_path must have a .png suffix.")
    generate_geometry(
        int(config["xsize"]),
        int(config["ysize"]),
        design,
        int(config["nholes"]),
        plot=True,
        namesave=output_path.stem,
        title=title,
        fol_save=output_path.parent,
        show=show,
    )
    result = {
        "output_path": output_path,
        "title": title,
        "record_number": int(row["evaluation_number"]),
        "phase": row["phase"],
        "burn_in_index": (
            None if row["burn_in_index"] == "N/A" else int(row["burn_in_index"])
        ),
        "bo_iteration": (
            None if row["bo_iteration"] == "N/A" else int(row["bo_iteration"])
        ),
        "physical_compliance": float(row["physical_compliance"]),
        "raster_solid_fraction": float(row["raster_solid_fraction"]),
    }
    if include_design_vector:
        result["encoded_design_vector"] = design.detach().cpu().tolist()
    return result


def _require_eval_number(eval_number):
    if eval_number < 1:
        raise ValueError("eval_number must be a positive integer.")


def _matching_row(rows, phase, field, eval_number):
    matching = [
        row
        for row in rows
        if row["phase"] == phase and int(row[field]) == eval_number
    ]
    if not matching:
        label = "BO" if phase == "bo" else "burn-in"
        raise KeyError(f"No {label} evaluation was recorded for number {eval_number}.")
    return matching[0]


def _verified_csv_design(row, archive):
    archive_index = int(row["evaluation_index"])
    design = torch.as_tensor(
        json.loads(row["encoded_design_vector"]), dtype=archive["x"].dtype
    )
    if _tensor_sha256(design) != row["design_vector_sha256"]:
        raise ValueError("The CSV design vector does not match its recorded checksum.")
    if not torch.equal(design, archive["x"][archive_index]):
        raise ValueError("The CSV design vector does not match the lossless archive row.")
    return design


def plot_best_geometry(
    run_directory,
    *,
    eval_number: int,
    output_path=None,
    show: bool = False,
):
    """Save the best geometry available through one BO evaluation."""
    _require_eval_number(eval_number)
    run_directory, rows, archive, config = _load_run_records(run_directory)
    _matching_row(rows, "bo", "bo_iteration", eval_number)
    eligible = [
        row
        for row in rows
        if row["phase"] == "burn-in"
        or (row["phase"] == "bo" and int(row["bo_iteration"]) <= eval_number)
    ]
    best_row = min(eligible, key=lambda row: float(row["physical_compliance"]))
    output_path = (
        Path(output_path)
        if output_path is not None
        else run_directory / f"Best_Geometry_eval_{eval_number:04d}.png"
    )
    return _render_geometry(
        archive["x"][int(best_row["evaluation_index"])],
        best_row,
        config,
        output_path,
        f"Best after BO evaluation {eval_number}",
        show=show,
    )


def plot_evaluated_geometry(
    run_directory,
    *,
    eval_number: int,
    output_path=None,
    show: bool = False,
):
    """Save and report the exact design at one BO evaluation."""
    _require_eval_number(eval_number)
    run_directory, rows, archive, config = _load_run_records(run_directory)
    row = _matching_row(rows, "bo", "bo_iteration", eval_number)
    design = _verified_csv_design(row, archive)
    output_path = (
        Path(output_path)
        if output_path is not None
        else run_directory / f"Geometry_eval_{eval_number:04d}.png"
    )
    return _render_geometry(
        design,
        row,
        config,
        output_path,
        f"Geometry at BO evaluation {eval_number}",
        show=show,
        include_design_vector=True,
    )


def plot_best_burn_in_geometry(
    run_directory,
    *,
    eval_number: int,
    output_path=None,
    show: bool = False,
):
    """Save the best geometry available through one burn-in evaluation."""
    _require_eval_number(eval_number)
    run_directory, rows, archive, config = _load_run_records(run_directory)
    _matching_row(rows, "burn-in", "burn_in_index", eval_number)
    eligible = [
        row
        for row in rows
        if row["phase"] == "burn-in"
        and int(row["burn_in_index"]) <= eval_number
    ]
    best_row = min(eligible, key=lambda row: float(row["physical_compliance"]))
    output_path = (
        Path(output_path)
        if output_path is not None
        else run_directory / f"Best_Burn_In_Geometry_eval_{eval_number:04d}.png"
    )
    return _render_geometry(
        archive["x"][int(best_row["evaluation_index"])],
        best_row,
        config,
        output_path,
        f"Best after burn-in evaluation {eval_number}",
        show=show,
    )


def plot_evaluated_burn_in_geometry(
    run_directory,
    *,
    eval_number: int,
    output_path=None,
    show: bool = False,
):
    """Save and report the exact design at one burn-in evaluation."""
    _require_eval_number(eval_number)
    run_directory, rows, archive, config = _load_run_records(run_directory)
    row = _matching_row(rows, "burn-in", "burn_in_index", eval_number)
    design = _verified_csv_design(row, archive)

    output_path = (
        Path(output_path)
        if output_path is not None
        else run_directory / f"Burn_In_Geometry_eval_{eval_number:04d}.png"
    )
    return _render_geometry(
        design,
        row,
        config,
        output_path,
        f"Geometry at burn-in evaluation {eval_number}",
        show=show,
        include_design_vector=True,
    )


def _print_result(result):
    print(f"title: {result['title']}")
    print(f"archive record: {result['record_number']}")
    print(f"phase: {result['phase']}")
    print(f"burn-in index: {result['burn_in_index']}")
    print(f"BO iteration: {result['bo_iteration']}")
    if "encoded_design_vector" in result:
        print(f"encoded design vector: {result['encoded_design_vector']}")
    print(f"compliance: {result['physical_compliance']:.6f}")
    print(f"raster solid fraction: {result['raster_solid_fraction']:.6f}")
    print(f"saved: {result['output_path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--best-eval", "--best-after-bo", "--bo-iteration", type=int
    )
    selector.add_argument(
        "--eval", dest="evaluated", type=int
    )
    selector.add_argument(
        "--best-burn-in", "--best-after-burn-in", "--burn-in-index", type=int
    )
    selector.add_argument("--burn-in", dest="evaluated_burn_in", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()
    if arguments.best_eval is not None:
        result = plot_best_geometry(
            arguments.run_directory,
            eval_number=arguments.best_eval,
            output_path=arguments.output,
            show=arguments.show,
        )
    elif arguments.evaluated is not None:
        result = plot_evaluated_geometry(
            arguments.run_directory,
            eval_number=arguments.evaluated,
            output_path=arguments.output,
            show=arguments.show,
        )
    elif arguments.best_burn_in is not None:
        result = plot_best_burn_in_geometry(
            arguments.run_directory,
            eval_number=arguments.best_burn_in,
            output_path=arguments.output,
            show=arguments.show,
        )
    else:
        result = plot_evaluated_burn_in_geometry(
            arguments.run_directory,
            eval_number=arguments.evaluated_burn_in,
            output_path=arguments.output,
            show=arguments.show,
        )
    _print_result(result)


if __name__ == "__main__":
    main()
