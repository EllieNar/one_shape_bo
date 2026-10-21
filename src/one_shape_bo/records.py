"""Lossless, replayable run records for Bayesian optimisation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import uuid

import numpy as np
import torch


SCHEMA_VERSION = 4
ARCHIVE_NAME = "evaluation_archive.pt"


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return value


def _json_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    return Path(temporary)


def atomic_torch_save(payload, path: Path):
    """Write a torch payload without exposing a partially-written target."""
    path = Path(path)
    temporary = _atomic_path(path)
    try:
        with temporary.open("wb") as stream:
            torch.save(_cpu_copy(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_write(payload, path: Path):
    path = Path(path)
    temporary = _atomic_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv_write(rows, fields, path: Path):
    path = Path(path)
    temporary = _atomic_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng_state(generator: torch.Generator | None = None):
    """Capture RNG state without advancing any random stream."""
    return {
        "package_generator": None if generator is None else generator.get_state().cpu(),
        "torch_cpu": torch.random.get_rng_state().cpu(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state, generator: torch.Generator | None = None):
    """Restore a state produced by :func:`capture_rng_state`."""
    if generator is not None and state.get("package_generator") is not None:
        generator.set_state(state["package_generator"])
    torch.random.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _sha256_bytes(content: bytes):
    return hashlib.sha256(content).hexdigest()


def _tensor_sha256(tensor):
    contiguous = torch.as_tensor(tensor).detach().cpu().contiguous()
    metadata_bytes = f"{contiguous.dtype}:{tuple(contiguous.shape)}".encode()
    return _sha256_bytes(metadata_bytes + contiguous.numpy().tobytes())


def _git_metadata(repository: Path):
    def run(*arguments):
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    source_hashes = {}
    source_root = repository / "src" / "one_shape_bo"
    if source_root.exists():
        for path in sorted(source_root.glob("*.py")):
            source_hashes[str(path.relative_to(repository))] = _sha256_bytes(
                path.read_bytes()
            )
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "tracked_status": status,
        "source_sha256": source_hashes,
    }


def _software_metadata():
    packages = {}
    for name in ("numpy", "scipy", "torch", "botorch", "gpytorch", "matplotlib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def sample_variance(values):
    """Return the ordinary n-1 sample variance, or ``None`` when undefined."""
    values = torch.as_tensor(values).detach().reshape(-1).double().cpu()
    if values.numel() < 2:
        return None
    return float(torch.var(values, correction=1).item())


class RunRecorder:
    """Maintain an ordered evaluation archive and per-fit replay snapshots."""

    evaluation_fields = (
        "evaluation_index",
        "evaluation_number",
        "phase",
        "burn_in_index",
        "bo_iteration",
        "candidate_source",
        "physical_compliance",
        "gp_response_prestandardization",
        "raw_compliance_sample_variance_to_date",
        "gp_response_sample_variance_to_date",
        "analytical_feasible",
        "raster_feasible",
        "feasible",
        "raster_solid_fraction",
        "encoded_design_vector",
        "design_vector_sha256",
        "geometry_signature_sha256",
        "tensor_reference",
    )

    fit_fields = (
        "fit_index",
        "bo_iteration",
        "status",
        "train_count",
        "archive_stop_exclusive",
        "raw_compliance_sample_variance",
        "gp_response_sample_variance",
        "standardization_mean",
        "standardization_std",
        "noise_mode",
        "prescribed_observation_noise_variance",
        "prescribed_observation_noise_variance_standardized",
        "prescribed_observation_noise_variance_raw_response",
        "signal_variance",
        "observation_noise_variance",
        "signal_to_noise_ratio",
        "signal_fraction_nu",
        "representative_point_count",
        "posterior_mean_spread_ratio",
        "posterior_std_ratio",
        "withheld_predictive_mean",
        "withheld_predictive_std",
        "withheld_observed_response",
        "withheld_standardized_residual",
        "predictive_window_count",
        "predictive_rms_z",
        "health_warning_triggered",
        "health_warning_reasons",
        "health_action",
        "next_fit_forced_cold",
        "initialization_type",
        "warm_start_source_fit_index",
        "warm_start_state_reference",
        "inherited_lengthscales",
        "inherited_outputscales",
        "inherited_likelihood_noise_variance",
        "inherited_mean_module_state",
        "post_fit_lengthscales",
        "post_fit_outputscales",
        "likelihood_observation_noise_variance",
        "kernel_outputscale_mean",
        "warning_count",
        "failure_type",
        "failure_message",
        "state_file",
        "configuration_sha256",
    )

    def __init__(
        self,
        output_dir: Path,
        configuration: dict,
        resolved_configuration: dict,
        repository: Path,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        protected = (
            "run_config.json",
            "run_manifest.json",
            ARCHIVE_NAME,
            "evaluations.csv",
            "burn_in_sampling.json",
            "burn_in_summary.json",
            "gp_fits.json",
            "gp_fits.csv",
            "initial_rng_state.pt",
            "iteration_events.json",
            "restart_attempts.csv",
            "fit_states",
        )
        collisions = [name for name in protected if (self.output_dir / name).exists()]
        if collisions:
            raise FileExistsError(
                "Run record files already exist in output directory: "
                + ", ".join(collisions)
            )
        self.fit_dir = self.output_dir / "fit_states"
        self.fit_dir.mkdir(exist_ok=True)
        self.rows = []
        self.fit_records = []
        self.iteration_events = []
        self.configuration = _json_safe(configuration)
        configuration_bytes = json.dumps(
            self.configuration, sort_keys=True, separators=(",", ":")
        ).encode()
        self.configuration_sha256 = _sha256_bytes(configuration_bytes)
        self.run_id = (
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "initializing",
            "initial_seed": self.configuration.get("seed"),
            "configuration_file": "run_config.json",
            "configuration_sha256": self.configuration_sha256,
            "resolved_configuration": _json_safe(resolved_configuration),
            "evaluation_archive": ARCHIVE_NAME,
            "software": _software_metadata(),
            "git": _git_metadata(Path(repository)),
        }
        atomic_json_write(self.configuration, self.output_dir / "run_config.json")
        atomic_json_write(self.manifest, self.output_dir / "run_manifest.json")

    @property
    def evaluation_count(self):
        return len(self.rows)

    def record_initial_rng(self, rng_state):
        atomic_torch_save(rng_state, self.output_dir / "initial_rng_state.pt")
        self.manifest["initial_rng_state"] = "initial_rng_state.pt"
        self.manifest["status"] = "running"
        atomic_json_write(self.manifest, self.output_dir / "run_manifest.json")

    def append_evaluations(
        self,
        designs,
        compliances,
        gp_responses,
        raster_fractions,
        signatures,
        analytical_feasible,
        raster_feasible,
        *,
        phase: str,
        sources,
        burn_in_indices=None,
        bo_iterations=None,
    ):
        designs = torch.as_tensor(designs).detach().cpu()
        compliances = torch.as_tensor(compliances).detach().cpu().reshape(-1)
        gp_responses = torch.as_tensor(gp_responses).detach().cpu().reshape(-1)
        raster_fractions = torch.as_tensor(raster_fractions).detach().cpu().reshape(-1)
        analytical_feasible = (
            torch.as_tensor(analytical_feasible).detach().cpu().bool().reshape(-1)
        )
        raster_feasible = (
            torch.as_tensor(raster_feasible).detach().cpu().bool().reshape(-1)
        )
        count = designs.shape[0]
        if phase not in ("burn-in", "bo"):
            raise ValueError("phase must be 'burn-in' or 'bo'.")
        if isinstance(sources, str):
            sources = [sources] * count
        burn_in_indices = burn_in_indices or [None] * count
        bo_iterations = bo_iterations or [None] * count
        aligned = (
            compliances,
            gp_responses,
            raster_fractions,
            analytical_feasible,
            raster_feasible,
            signatures,
            sources,
            burn_in_indices,
            bo_iterations,
        )
        if any(len(values) != count for values in aligned):
            raise ValueError("Evaluation record fields must have identical row counts.")
        if phase == "burn-in" and (
            any(index is None for index in burn_in_indices)
            or any(index is not None for index in bo_iterations)
        ):
            raise ValueError(
                "Burn-in records require burn-in indices and no BO iterations."
            )
        if phase == "bo" and (
            any(index is not None for index in burn_in_indices)
            or any(index is None for index in bo_iterations)
        ):
            raise ValueError("BO records require BO iterations and no burn-in indices.")
        for offset in range(count):
            signature = bytes(signatures[offset])
            self.rows.append(
                {
                    "x": designs[offset].clone(),
                    "compliance": compliances[offset].clone(),
                    "gp_response": gp_responses[offset].clone(),
                    "raster_fraction": raster_fractions[offset].clone(),
                    "signature": torch.tensor(list(signature), dtype=torch.uint8),
                    "analytical_feasible": bool(analytical_feasible[offset]),
                    "raster_feasible": bool(raster_feasible[offset]),
                    "feasible": bool(
                        analytical_feasible[offset] and raster_feasible[offset]
                    ),
                    "phase": phase,
                    "source": str(sources[offset]),
                    "burn_in_index": burn_in_indices[offset],
                    "bo_iteration": bo_iterations[offset],
                }
            )
        self._flush_evaluations()

    def _archive_payload(self):
        if not self.rows:
            return {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "x": torch.empty((0, 0), dtype=torch.double),
            }
        signature_lengths = {row["signature"].numel() for row in self.rows}
        if len(signature_lengths) != 1:
            raise ValueError("Geometry signatures must have a fixed length within a run.")
        source_names = ("burn-in", "global", "local")
        source_code = {name: index for index, name in enumerate(source_names)}
        phase_names = ("burn-in", "bo")
        phase_code = {name: index for index, name in enumerate(phase_names)}
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "source_names": source_names,
            "phase_names": phase_names,
            "x": torch.stack([row["x"] for row in self.rows]),
            "physical_compliance": torch.stack(
                [row["compliance"] for row in self.rows]
            ).reshape(-1, 1),
            "gp_response_prestandardization": torch.stack(
                [row["gp_response"] for row in self.rows]
            ).reshape(-1, 1),
            "raster_solid_fraction": torch.stack(
                [row["raster_fraction"] for row in self.rows]
            ),
            "geometry_signature": torch.stack(
                [row["signature"] for row in self.rows]
            ),
            "analytical_feasible": torch.tensor(
                [row["analytical_feasible"] for row in self.rows], dtype=torch.bool
            ),
            "raster_feasible": torch.tensor(
                [row["raster_feasible"] for row in self.rows], dtype=torch.bool
            ),
            "feasible": torch.tensor(
                [row["feasible"] for row in self.rows], dtype=torch.bool
            ),
            "source_code": torch.tensor(
                [source_code.get(row["source"], -1) for row in self.rows],
                dtype=torch.int8,
            ),
            "phase_code": torch.tensor(
                [phase_code[row["phase"]] for row in self.rows], dtype=torch.int8
            ),
            "burn_in_index": torch.tensor(
                [
                    -1 if row["burn_in_index"] is None else row["burn_in_index"]
                    for row in self.rows
                ],
                dtype=torch.int64,
            ),
            "bo_iteration": torch.tensor(
                [
                    -1 if row["bo_iteration"] is None else row["bo_iteration"]
                    for row in self.rows
                ],
                dtype=torch.int64,
            ),
        }

    def _flush_evaluations(self):
        payload = self._archive_payload()
        atomic_torch_save(payload, self.output_dir / ARCHIVE_NAME)
        csv_rows = []
        for index, row in enumerate(self.rows):
            compliance_to_date = payload["physical_compliance"][: index + 1]
            response_to_date = payload["gp_response_prestandardization"][: index + 1]
            compliance_variance = sample_variance(compliance_to_date)
            response_variance = sample_variance(response_to_date)
            csv_rows.append(
                {
                    "evaluation_index": index,
                    "evaluation_number": index + 1,
                    "phase": row["phase"],
                    "burn_in_index": (
                        "N/A"
                        if row["burn_in_index"] is None
                        else row["burn_in_index"]
                    ),
                    "bo_iteration": (
                        "N/A" if row["bo_iteration"] is None else row["bo_iteration"]
                    ),
                    "candidate_source": row["source"],
                    "physical_compliance": float(row["compliance"]),
                    "gp_response_prestandardization": float(row["gp_response"]),
                    "raw_compliance_sample_variance_to_date": (
                        "N/A"
                        if compliance_variance is None
                        else compliance_variance
                    ),
                    "gp_response_sample_variance_to_date": (
                        "N/A"
                        if response_variance is None
                        else response_variance
                    ),
                    "analytical_feasible": row["analytical_feasible"],
                    "raster_feasible": row["raster_feasible"],
                    "feasible": row["feasible"],
                    "raster_solid_fraction": float(row["raster_fraction"]),
                    "encoded_design_vector": json.dumps(
                        row["x"].tolist(), separators=(",", ":")
                    ),
                    "design_vector_sha256": _tensor_sha256(row["x"]),
                    "geometry_signature_sha256": _sha256_bytes(
                        bytes(row["signature"].tolist())
                    ),
                    "tensor_reference": f"{ARCHIVE_NAME}:x[{index}]",
                }
            )
        atomic_csv_write(
            csv_rows, self.evaluation_fields, self.output_dir / "evaluations.csv"
        )
        self.manifest["evaluation_count"] = len(self.rows)
        atomic_json_write(self.manifest, self.output_dir / "run_manifest.json")

    def record_burn_in_sampling(self, diagnostics):
        """Persist cheap sampling diagnostics before physical evaluation."""
        atomic_json_write(
            diagnostics, self.output_dir / "burn_in_sampling.json"
        )
        self.manifest["burn_in_sampling"] = "burn_in_sampling.json"
        atomic_json_write(self.manifest, self.output_dir / "run_manifest.json")

    def record_burn_in_summary(self):
        indices = [
            index for index, row in enumerate(self.rows) if row["phase"] == "burn-in"
        ]
        if not indices:
            return
        compliance = torch.stack([self.rows[index]["compliance"] for index in indices])
        responses = torch.stack([self.rows[index]["gp_response"] for index in indices])
        best_local = int(torch.argmin(compliance).item())
        worst_local = int(torch.argmax(compliance).item())

        def endpoint(local_index):
            archive_index = indices[local_index]
            row = self.rows[archive_index]
            return {
                "archive_index": archive_index,
                "burn_in_index": row["burn_in_index"],
                "encoded_design_vector": row["x"],
                "physical_compliance": float(row["compliance"]),
                "gp_response_prestandardization": float(row["gp_response"]),
            }

        summary = {
            "count": len(indices),
            "archive_range": [min(indices), max(indices) + 1],
            "best": endpoint(best_local),
            "worst": endpoint(worst_local),
            "raw_compliance_sample_variance": sample_variance(compliance),
            "gp_response_sample_variance": sample_variance(responses),
            "variance_definition": "n-1 sample variance",
            "actual_designs_reference": f"{ARCHIVE_NAME}:x[0:{len(indices)}]",
        }
        atomic_json_write(summary, self.output_dir / "burn_in_summary.json")

    def record_fit(
        self,
        *,
        bo_iteration: int,
        train_x,
        train_compliance,
        train_gp_response,
        rng_before,
        rng_after,
        warnings_recorded,
        model_state=None,
        diagnostics=None,
        transform_parameters=None,
        initialization_metadata=None,
        noise_mode=None,
        prescribed_observation_noise_variance=None,
        prescribed_observation_noise_variance_standardized=None,
        prescribed_observation_noise_variance_raw_response=None,
        failure=None,
    ):
        train_x = torch.as_tensor(train_x).detach().cpu()
        train_compliance = torch.as_tensor(train_compliance).detach().cpu()
        train_gp_response = torch.as_tensor(train_gp_response).detach().cpu()
        train_count = len(train_x)
        if train_count != self.evaluation_count:
            raise ValueError(
                "GP training rows must align with the recorded evaluation archive."
            )
        fit_index = len(self.fit_records) + 1
        state_name = f"fit_{fit_index:04d}.pt"
        status = "failed" if failure is not None else "success"
        warning_rows = [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
                "filename": warning.filename,
                "lineno": warning.lineno,
            }
            for warning in warnings_recorded
        ]
        initialization_metadata = initialization_metadata or {
            "initialization_type": "unknown",
            "warm_start_source_fit_index": None,
            "warm_start_state_reference": None,
            "inherited_lengthscales": None,
            "inherited_outputscales": None,
            "inherited_likelihood_noise_variance": None,
            "inherited_mean_module_state": None,
        }
        state = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "fit_index": fit_index,
            "bo_iteration": bo_iteration,
            "status": status,
            "training_archive_reference": {
                "file": f"../{ARCHIVE_NAME}",
                "start_inclusive": 0,
                "stop_exclusive": train_count,
            },
            "train_x_sha256": _tensor_sha256(train_x),
            "train_compliance_sha256": _tensor_sha256(train_compliance),
            "train_gp_response_sha256": _tensor_sha256(train_gp_response),
            "configuration_reference": "../run_config.json",
            "configuration_sha256": self.configuration_sha256,
            "objective_transform": {
                "name": "negative_log_compliance",
                "formula": "-log(clamp_min(compliance, 1e-12))",
                "clamp_min": 1e-12,
            },
            "transform_parameters": transform_parameters,
            "initialization": initialization_metadata,
            "noise_mode": noise_mode,
            "prescribed_observation_noise_variance": (
                prescribed_observation_noise_variance
            ),
            "prescribed_observation_noise_variance_standardized": (
                prescribed_observation_noise_variance_standardized
            ),
            "prescribed_observation_noise_variance_raw_response": (
                prescribed_observation_noise_variance_raw_response
            ),
            "health": None,
            "model_state": model_state,
            "diagnostics": diagnostics,
            "rng_before_fit": rng_before,
            "rng_after_fit": rng_after,
            "warnings": warning_rows,
            "failure": failure,
        }
        atomic_torch_save(state, self.fit_dir / state_name)
        outputscales = []
        if diagnostics is not None:
            outputscales = list(diagnostics.get("outputscales", {}).values())
        record = {
            "fit_index": fit_index,
            "bo_iteration": bo_iteration,
            "status": status,
            "train_count": train_count,
            "archive_stop_exclusive": train_count,
            "raw_compliance_sample_variance": sample_variance(train_compliance),
            "gp_response_sample_variance": sample_variance(train_gp_response),
            "standardization_mean": (
                None if transform_parameters is None else transform_parameters.get("mean")
            ),
            "standardization_std": (
                None if transform_parameters is None else transform_parameters.get("std")
            ),
            "noise_mode": noise_mode,
            "prescribed_observation_noise_variance": (
                prescribed_observation_noise_variance
            ),
            "prescribed_observation_noise_variance_standardized": (
                prescribed_observation_noise_variance_standardized
            ),
            "prescribed_observation_noise_variance_raw_response": (
                prescribed_observation_noise_variance_raw_response
            ),
            "signal_variance": None,
            "observation_noise_variance": None,
            "signal_to_noise_ratio": None,
            "signal_fraction_nu": None,
            "representative_point_count": None,
            "posterior_mean_spread_ratio": None,
            "posterior_std_ratio": None,
            "withheld_predictive_mean": None,
            "withheld_predictive_std": None,
            "withheld_observed_response": None,
            "withheld_standardized_residual": None,
            "predictive_window_count": None,
            "predictive_rms_z": None,
            "health_warning_triggered": None,
            "health_warning_reasons": None,
            "health_action": None,
            "next_fit_forced_cold": None,
            "initialization_type": initialization_metadata.get(
                "initialization_type"
            ),
            "warm_start_source_fit_index": initialization_metadata.get(
                "warm_start_source_fit_index"
            ),
            "warm_start_state_reference": initialization_metadata.get(
                "warm_start_state_reference"
            ),
            "inherited_lengthscales": initialization_metadata.get(
                "inherited_lengthscales"
            ),
            "inherited_outputscales": initialization_metadata.get(
                "inherited_outputscales"
            ),
            "inherited_likelihood_noise_variance": initialization_metadata.get(
                "inherited_likelihood_noise_variance"
            ),
            "inherited_mean_module_state": initialization_metadata.get(
                "inherited_mean_module_state"
            ),
            "post_fit_lengthscales": (
                None if diagnostics is None else diagnostics.get("lengthscales")
            ),
            "post_fit_outputscales": (
                None if diagnostics is None else diagnostics.get("outputscales")
            ),
            "likelihood_observation_noise_variance": (
                None if diagnostics is None else diagnostics.get("noise")
            ),
            "kernel_outputscale_mean": (
                None if not outputscales else float(np.mean(outputscales))
            ),
            "warning_count": len(warning_rows),
            "warnings": warning_rows,
            "failure_type": None if failure is None else failure.get("type"),
            "failure_message": None if failure is None else failure.get("message"),
            "not_applicable": (
                {}
                if failure is None
                else {
                    "standardization_mean": "GP fit failed",
                    "standardization_std": "GP fit failed",
                    "likelihood_observation_noise_variance": "GP fit failed",
                    "kernel_outputscale_mean": "GP fit failed",
                }
            ),
            "state_file": f"fit_states/{state_name}",
            "configuration_sha256": self.configuration_sha256,
        }
        self.fit_records.append(record)
        self._flush_fit_records()
        return fit_index

    def _flush_fit_records(self):
        atomic_json_write(self.fit_records, self.output_dir / "gp_fits.json")
        csv_rows = []
        for item in self.fit_records:
            csv_rows.append(
                {
                    field: (
                        "N/A"
                        if item.get(field) is None
                        else json.dumps(item[field])
                        if isinstance(item[field], (dict, list))
                        else item[field]
                    )
                    for field in self.fit_fields
                }
            )
        atomic_csv_write(csv_rows, self.fit_fields, self.output_dir / "gp_fits.csv")

    def update_fit_health(self, fit_index: int, health):
        """Attach health results to an existing fit record and snapshot."""
        if fit_index < 1 or fit_index > len(self.fit_records):
            raise IndexError(f"Unknown fit index {fit_index}.")
        health = _json_safe(health)
        record = self.fit_records[fit_index - 1]
        for field in self.fit_fields:
            if field in health:
                record[field] = health[field]
        state_path = self.fit_dir / f"fit_{fit_index:04d}.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        state["health"] = health
        atomic_torch_save(state, state_path)
        self._flush_fit_records()

    def record_iteration_event(self, event):
        self.iteration_events.append(_json_safe(event))
        atomic_json_write(
            self.iteration_events, self.output_dir / "iteration_events.json"
        )

    def finalize(self, status: str, **details):
        self.manifest["status"] = status
        self.manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.manifest["evaluation_count"] = self.evaluation_count
        self.manifest["fit_count"] = len(self.fit_records)
        self.manifest.update(_json_safe(details))
        atomic_json_write(self.manifest, self.output_dir / "run_manifest.json")


def load_evaluation_archive(run_directory):
    """Load a recorded archive. Run directories are trusted local artifacts."""
    return torch.load(
        Path(run_directory) / ARCHIVE_NAME, map_location="cpu", weights_only=False
    )


def load_fit_state(run_directory, fit_index: int):
    return torch.load(
        Path(run_directory) / "fit_states" / f"fit_{fit_index:04d}.pt",
        map_location="cpu",
        weights_only=False,
    )


def reconstruct_training_set(run_directory, bo_iteration: int, position="before"):
    """Reconstruct exact GP data before or after a recorded BO iteration."""
    if position not in ("before", "after"):
        raise ValueError("position must be 'before' or 'after'.")
    run_directory = Path(run_directory)
    archive = load_evaluation_archive(run_directory)
    fit_records = json.loads((run_directory / "gp_fits.json").read_text())
    matching = [
        record
        for record in fit_records
        if int(record["bo_iteration"]) == int(bo_iteration)
    ]
    if not matching:
        raise KeyError(f"No GP fit was recorded for BO iteration {bo_iteration}.")
    stop = int(matching[0]["archive_stop_exclusive"])
    if position == "after":
        bo_indices = torch.nonzero(
            archive["bo_iteration"] == int(bo_iteration), as_tuple=False
        ).reshape(-1)
        if not len(bo_indices):
            raise KeyError(f"No evaluation was recorded for BO iteration {bo_iteration}.")
        stop = max(stop, int(bo_indices.max().item()) + 1)
    return {
        "x": archive["x"][:stop].clone(),
        "physical_compliance": archive["physical_compliance"][:stop].clone(),
        "gp_response_prestandardization": archive[
            "gp_response_prestandardization"
        ][:stop].clone(),
        "archive_stop_exclusive": stop,
    }
