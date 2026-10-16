"""Offline, cold-started numerical study of GP observation-noise choices.

The study consumes completed BO archives and never evaluates the physical
model.  Every fit is constructed afresh from the production GP definition;
neither the production warm-start policy nor a preceding study fit is used.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import statistics
import time
import warnings

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from .gaussian import build_objective_gp, fit_objective_gp
from .records import (
    _git_metadata,
    atomic_csv_write,
    atomic_json_write,
    atomic_torch_save,
    load_evaluation_archive,
)
from .runtime import DEVICE


DEFAULT_SEEDS = (0, 1, 2, 3, 76)
DEFAULT_STANDARDIZED_VARIANCES = (1e-6, 1e-5, 1e-4)
DEFAULT_FIT_AFTER = (0, 5, 10, 20)
STUDY_SCHEMA_VERSION = 1
FIT_FIELDS = (
    "fit_id",
    "seed",
    "run_directory",
    "run_id",
    "source_archive_sha256",
    "source_configuration_sha256",
    "source_gaussian_sha256",
    "fit_after_bo_iteration",
    "train_count",
    "archive_stop_exclusive",
    "holdout_start_bo_iteration",
    "holdout_stop_bo_iteration",
    "holdout_count",
    "probe_count",
    "noise_mode",
    "noise_candidate",
    "requested_standardized_noise_variance",
    "requested_raw_response_noise_variance",
    "effective_standardized_noise_variance",
    "effective_raw_response_noise_variance",
    "fixed_noise_floor_applied",
    "training_response_sample_variance",
    "initialization_id",
    "initialization_seed",
    "initialization_type",
    "warm_start_used",
    "warm_start_source_fit_id",
    "initial_kernel_mean_sha256",
    "initial_full_hyperparameter_sha256",
    "initial_hyperparameters",
    "status",
    "duration_seconds",
    "warning_count",
    "warnings",
    "failure_type",
    "failure_message",
    "final_marginal_log_likelihood",
    "fitted_lengthscales",
    "fitted_outputscales",
    "fitted_likelihood_noise_variance_standardized",
    "probe_mean_min",
    "probe_mean_max",
    "probe_mean_range",
    "probe_mean_sd",
    "probe_mean_spread_ratio",
    "probe_latent_sd_min",
    "probe_latent_sd_max",
    "probe_latent_sd_median",
    "probe_latent_sd_q90",
    "probe_latent_sd_ratio_q90",
    "signal_variance",
    "signal_to_noise_ratio",
    "signal_fraction_nu",
    "heldout_rmse",
    "heldout_mae",
    "heldout_max_absolute_error",
    "heldout_negative_log_predictive_density",
    "heldout_rms_standardized_residual",
    "heldout_95_percent_coverage",
    "fit_state_file",
)


@dataclass(frozen=True)
class ArchivedRun:
    seed: int
    directory: Path
    archive: dict
    configuration: dict
    manifest: dict
    archive_sha256: str

    @property
    def dimension(self):
        return int(self.archive["x"].shape[-1])

    @property
    def nholes(self):
        if self.dimension % 7:
            raise ValueError(
                f"Archive dimension {self.dimension} is not divisible by seven."
            )
        return self.dimension // 7


def _sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensors(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_value(value), separators=(",", ":"), sort_keys=True)
    return value


def _write_fit_rows(rows, output_directory):
    csv_rows = [
        {field: _csv_value(row.get(field)) for field in FIT_FIELDS} for row in rows
    ]
    atomic_csv_write(
        csv_rows,
        FIT_FIELDS,
        Path(output_directory) / "gp_fits_numerical_study.csv",
    )


def _read_fit_rows(output_directory):
    path = Path(output_directory) / "gp_fits_numerical_study.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_archived_run(seed, directory):
    """Load and validate one completed persistent BO archive."""
    directory = Path(directory).expanduser().resolve()
    config_path = directory / "run_config.json"
    manifest_path = directory / "run_manifest.json"
    archive_path = directory / "evaluation_archive.pt"
    missing = [
        str(path.name)
        for path in (config_path, manifest_path, archive_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"{directory} is missing: {', '.join(missing)}")
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configured_seed = configuration.get("seed")
    if configured_seed is not None and int(configured_seed) != int(seed):
        raise ValueError(
            f"Seed mapping {seed} disagrees with {directory}'s seed "
            f"{configured_seed}."
        )
    archive = load_evaluation_archive(directory)
    required = {
        "x",
        "physical_compliance",
        "gp_response_prestandardization",
        "bo_iteration",
        "feasible",
    }
    if missing_keys := required.difference(archive):
        raise ValueError(f"{directory} archive lacks {sorted(missing_keys)}.")
    count = len(archive["x"])
    if any(len(archive[key]) != count for key in required):
        raise ValueError(f"Misaligned archive tensors in {directory}.")
    return ArchivedRun(
        seed=int(seed),
        directory=directory,
        archive=archive,
        configuration=configuration,
        manifest=manifest,
        archive_sha256=_sha256_file(archive_path),
    )


def validate_archived_runs(runs, *, allow_incompatible_sources=False):
    """Require comparable dimensions, model configuration and source revisions."""
    if not runs:
        raise ValueError("At least one archived run is required.")
    if len({run.seed for run in runs}) != len(runs):
        raise ValueError("Each archived run must have a distinct seed.")
    dimensions = {run.dimension for run in runs}
    if len(dimensions) != 1:
        raise ValueError(f"Archive dimensions differ: {sorted(dimensions)}")
    modelling_keys = ("nholes", "htype")
    for key in modelling_keys:
        values = {run.configuration.get(key) for run in runs}
        if len(values) != 1:
            raise ValueError(f"Archived run setting {key!r} differs: {values}")
    non_control_runs = [
        run.seed
        for run in runs
        if run.configuration.get("observation_noise_variance") is not None
    ]
    if non_control_runs:
        raise ValueError(
            "The shared trajectories must be learned-noise controls; prescribed "
            f"noise was configured for seeds {non_control_runs}."
        )
    source_hashes = {
        run.manifest.get("git", {})
        .get("source_sha256", {})
        .get("src/one_shape_bo/gaussian.py")
        for run in runs
    }
    source_hashes.discard(None)
    if len(source_hashes) > 1 and not allow_incompatible_sources:
        raise ValueError(
            "Archived runs record different gaussian.py source hashes; use "
            "--allow-incompatible-sources only after auditing that difference."
        )


def snapshot_from_archive(run, fit_after, holdout_size):
    """Return one training prefix and its subsequent archived BO holdout."""
    fit_after = int(fit_after)
    if fit_after < 0:
        raise ValueError("fit_after must be non-negative.")
    bo_iteration = run.archive["bo_iteration"].reshape(-1)
    train_mask = (bo_iteration < 0) | (bo_iteration <= fit_after)
    stop = int(torch.nonzero(train_mask, as_tuple=False).max().item()) + 1
    if not bool(train_mask[:stop].all()) or bool(train_mask[stop:].any()):
        raise ValueError("Evaluation archive is not ordered by BO iteration.")
    holdout_mask = bo_iteration > fit_after
    holdout_indices = torch.nonzero(holdout_mask, as_tuple=False).reshape(-1)
    holdout_indices = holdout_indices[: int(holdout_size)]
    if len(holdout_indices) < int(holdout_size):
        raise ValueError(
            f"Seed {run.seed} snapshot {fit_after} has only "
            f"{len(holdout_indices)} later BO evaluations; {holdout_size} required."
        )
    train_x = run.archive["x"][:stop].clone()
    train_y = run.archive["physical_compliance"][:stop].clone()
    train_score = run.archive["gp_response_prestandardization"][:stop].clone()
    expected_score = -torch.log(train_y.clamp_min(1e-12))
    if not torch.equal(train_score, expected_score):
        if not torch.allclose(train_score, expected_score, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Seed {run.seed} stored GP responses are inconsistent.")
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_score": train_score,
        "archive_stop_exclusive": stop,
        "holdout_x": run.archive["x"][holdout_indices].clone(),
        "holdout_score": run.archive["gp_response_prestandardization"][
            holdout_indices
        ].clone(),
        "holdout_start": int(bo_iteration[holdout_indices[0]].item()),
        "holdout_stop": int(bo_iteration[holdout_indices[-1]].item()),
    }


def representative_probe_points(runs):
    """Build one deterministic feasible-design probe tensor shared by all fits."""
    feasible = []
    for run in sorted(runs, key=lambda item: item.seed):
        mask = run.archive["feasible"].reshape(-1).bool()
        feasible.append(run.archive["x"][mask].detach().cpu())
    if not feasible or not sum(len(points) for points in feasible):
        raise ValueError("No feasible archived designs are available as probes.")
    return torch.unique(torch.cat(feasible), dim=0, sorted=True)


def standardized_to_raw_noise_variance(standardized_variance, train_score):
    """Convert desired standardized variance to pre-transform response units."""
    sample_variance = torch.var(
        torch.as_tensor(train_score).reshape(-1).double(), correction=1
    )
    if not torch.isfinite(sample_variance) or sample_variance <= 0:
        raise ValueError("Training GP responses must have positive sample variance.")
    return float(standardized_variance) * float(sample_variance.item())


def _parameter_items(gp, *, kernel_mean_only=False):
    prefixes = ("mean_module.", "covar_module.")
    for name, parameter in gp.named_parameters():
        if name.startswith(prefixes) or (
            not kernel_mean_only and name.startswith("likelihood.")
        ):
            yield name, parameter


def _parameter_summary(gp):
    return {
        name: parameter.detach().cpu().reshape(-1).tolist()
        for name, parameter in _parameter_items(gp)
    }


def _stable_initialization_seed(seed, fit_after, initialization_id):
    content = f"{int(seed)}:{int(fit_after)}:{int(initialization_id)}".encode()
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "little") % (2**31)


def apply_controlled_initialization(
    gp,
    *,
    seed,
    fit_after,
    initialization_id,
    perturbation_scale=0.25,
):
    """Apply one reproducible cold initialization to an unfitted fresh GP."""
    initialization_id = int(initialization_id)
    if initialization_id < 0:
        raise ValueError("initialization_id must be non-negative.")
    initialization_seed = _stable_initialization_seed(
        seed, fit_after, initialization_id
    )
    if initialization_id:
        with torch.no_grad():
            for name, parameter in _parameter_items(gp):
                name_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{initialization_seed}:{name}".encode()
                    ).digest()[:8],
                    "little",
                ) % (2**31)
                generator = torch.Generator(device="cpu").manual_seed(name_seed)
                perturbation = torch.randn(
                    parameter.shape, generator=generator, dtype=torch.double
                ).to(device=parameter.device, dtype=parameter.dtype)
                parameter.add_(float(perturbation_scale) * perturbation)
    kernel_mean_items = list(_parameter_items(gp, kernel_mean_only=True))
    full_items = list(_parameter_items(gp))
    return {
        "initialization_seed": initialization_seed,
        "initial_hyperparameters": _parameter_summary(gp),
        "initial_kernel_mean_sha256": _sha256_tensors(kernel_mean_items),
        "initial_full_hyperparameter_sha256": _sha256_tensors(full_items),
    }


def _fitted_hyperparameters(gp):
    lengthscales, outputscales = {}, {}
    for name, module in gp.covar_module.named_modules():
        label = name or "covar_module"
        if hasattr(module, "lengthscale") and module.lengthscale is not None:
            lengthscales[label] = (
                module.lengthscale.detach().cpu().reshape(-1).tolist()
            )
        if hasattr(module, "outputscale"):
            outputscales[label] = float(
                module.outputscale.detach().cpu().reshape(-1)[0].item()
            )
    return lengthscales, outputscales


def _effective_noise(gp):
    standardized = float(gp.likelihood.noise.detach().reshape(-1).median().item())
    response_std = float(
        gp.outcome_transform.stdvs.detach().reshape(-1)[0].cpu().item()
    )
    return standardized, standardized * response_std**2


def _predictive_metrics(gp, probe_x, holdout_x, holdout_score, response_sd):
    gp.eval()
    gp.likelihood.eval()
    with torch.no_grad():
        probe = gp.posterior(probe_x)
        heldout = gp.posterior(holdout_x)
        probe_mean = probe.mean.reshape(-1)
        probe_sd = probe.variance.clamp_min(0).sqrt().reshape(-1)
        heldout_mean = heldout.mean.reshape(-1)
        heldout_latent_var = heldout.variance.clamp_min(0).reshape(-1)
        observed = holdout_score.reshape(-1)
        standardized_noise, raw_noise = _effective_noise(gp)
        predictive_var = heldout_latent_var + raw_noise
        predictive_sd = predictive_var.sqrt()
        residual = observed - heldout_mean
        nlpd = 0.5 * (
            torch.log(2 * torch.pi * predictive_var)
            + residual.square() / predictive_var
        )
        kernel_diagonal = gp.covar_module(probe_x).diagonal(dim1=-2, dim2=-1)
        signal = float(kernel_diagonal.reshape(-1).median().item())
    return {
        "effective_standardized_noise_variance": standardized_noise,
        "effective_raw_response_noise_variance": raw_noise,
        "probe_mean_min": float(probe_mean.min().item()),
        "probe_mean_max": float(probe_mean.max().item()),
        "probe_mean_range": float((probe_mean.max() - probe_mean.min()).item()),
        "probe_mean_sd": float(probe_mean.std(correction=1).item()),
        "probe_mean_spread_ratio": float(
            (probe_mean.std(correction=1) / response_sd).item()
        ),
        "probe_latent_sd_min": float(probe_sd.min().item()),
        "probe_latent_sd_max": float(probe_sd.max().item()),
        "probe_latent_sd_median": float(probe_sd.median().item()),
        "probe_latent_sd_q90": float(torch.quantile(probe_sd, 0.9).item()),
        "probe_latent_sd_ratio_q90": float(
            (torch.quantile(probe_sd, 0.9) / response_sd).item()
        ),
        "signal_variance": signal,
        "signal_to_noise_ratio": signal / max(standardized_noise, 1e-300),
        "signal_fraction_nu": signal / max(signal + standardized_noise, 1e-300),
        "heldout_rmse": float(residual.square().mean().sqrt().item()),
        "heldout_mae": float(residual.abs().mean().item()),
        "heldout_max_absolute_error": float(residual.abs().max().item()),
        "heldout_negative_log_predictive_density": float(nlpd.mean().item()),
        "heldout_rms_standardized_residual": float(
            (residual / predictive_sd).square().mean().sqrt().item()
        ),
        "heldout_95_percent_coverage": float(
            (residual.abs() <= 1.96 * predictive_sd).double().mean().item()
        ),
    }


def construct_cold_study_model(
    *,
    train_x,
    train_y,
    nholes,
    observation_noise_variance,
    seed,
    fit_after,
    initialization_id,
    perturbation_scale=0.25,
):
    """Construct and initialize one independent, unfitted study GP.

    No previous-model argument exists by design, making warm-starting impossible
    through this interface.
    """
    gp, train_score = build_objective_gp(
        train_x,
        train_y,
        nholes,
        observation_noise_variance=observation_noise_variance,
    )
    initialization = apply_controlled_initialization(
        gp,
        seed=seed,
        fit_after=fit_after,
        initialization_id=initialization_id,
        perturbation_scale=perturbation_scale,
    )
    return gp, train_score, initialization


def fit_cold_study_model(**arguments):
    """Construct, initialize and fit one independent study GP once."""
    gp, train_score, initialization = construct_cold_study_model(**arguments)
    fit_objective_gp(gp, max_attempts=1)
    return gp, train_score, initialization


def _source_gaussian_hash(run):
    return (
        run.manifest.get("git", {})
        .get("source_sha256", {})
        .get("src/one_shape_bo/gaussian.py")
    )


def _fit_identifier(seed, fit_after, noise_candidate, initialization_id):
    candidate = str(noise_candidate).replace("-", "m").replace(".", "p")
    return (
        f"seed{int(seed):03d}_after{int(fit_after):04d}_"
        f"{candidate}_init{int(initialization_id)}"
    )


def _final_mll(gp):
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    gp.train()
    gp.likelihood.train()
    with torch.no_grad():
        output = gp(*gp.train_inputs)
        value = mll(output, gp.train_targets)
    return float(value.detach().reshape(-1).mean().cpu().item())


def _software_manifest():
    packages = {}
    for package in ("torch", "botorch", "gpytorch", "numpy", "scipy"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {"packages": packages, "device": str(DEVICE)}


def run_numerical_noise_study(
    run_directories,
    output_directory,
    *,
    standardized_variances=DEFAULT_STANDARDIZED_VARIANCES,
    fit_after=DEFAULT_FIT_AFTER,
    holdout_size=10,
    initializations=3,
    perturbation_scale=0.25,
    expected_seed_count=5,
    allow_partial=False,
    allow_incompatible_sources=False,
):
    """Execute or resume the full archive-only numerical-noise study."""
    runs = [load_archived_run(seed, path) for seed, path in run_directories.items()]
    runs.sort(key=lambda item: item.seed)
    if len(runs) != int(expected_seed_count):
        raise ValueError(
            f"Expected {expected_seed_count} seed archives, received {len(runs)}."
        )
    validate_archived_runs(
        runs, allow_incompatible_sources=allow_incompatible_sources
    )
    standardized_variances = tuple(float(value) for value in standardized_variances)
    if not standardized_variances or any(
        value <= 0 for value in standardized_variances
    ):
        raise ValueError("Standardized variances must be positive.")
    fit_after = tuple(int(value) for value in fit_after)
    if int(initializations) < 1:
        raise ValueError("initializations must be positive.")
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    state_directory = output_directory / "fit_states"
    state_directory.mkdir(exist_ok=True)
    probe_x = representative_probe_points(runs)
    study_config = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "run_directories": {str(run.seed): str(run.directory) for run in runs},
        "seeds": [run.seed for run in runs],
        "standardized_variances": list(standardized_variances),
        "learned_noise_control": True,
        "fit_after_bo_iterations": list(fit_after),
        "holdout_size": int(holdout_size),
        "initializations": int(initializations),
        "perturbation_scale": float(perturbation_scale),
        "cold_start_every_fit": True,
        "fit_max_attempts": 1,
        "allow_partial": bool(allow_partial),
        "allow_incompatible_sources": bool(allow_incompatible_sources),
    }
    config_path = output_directory / "study_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != study_config:
            raise ValueError(
                "Existing study_config.json differs; choose a new output directory."
            )
    else:
        atomic_json_write(study_config, config_path)
    manifest_path = output_directory / "study_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": STUDY_SCHEMA_VERSION,
            "study_id": (
                f"noise-{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "software": _software_manifest(),
            "study_source": _git_metadata(Path(__file__).resolve().parents[2]),
            "source_archives": {
                str(run.seed): {
                    "directory": str(run.directory),
                    "run_id": run.manifest.get("run_id"),
                    "archive_sha256": run.archive_sha256,
                    "configuration_sha256": run.manifest.get(
                        "configuration_sha256"
                    ),
                    "gaussian_source_sha256": _source_gaussian_hash(run),
                }
                for run in runs
            },
        }
        atomic_json_write(manifest, manifest_path)

    existing_rows = _read_fit_rows(output_directory)
    completed_ids = {
        row["fit_id"] for row in existing_rows if row.get("status") == "success"
    }
    rows = list(existing_rows)
    failures = []
    for run in runs:
        for snapshot_index in fit_after:
            try:
                snapshot = snapshot_from_archive(run, snapshot_index, holdout_size)
            except ValueError as error:
                if not allow_partial:
                    raise
                failures.append(
                    {
                        "seed": run.seed,
                        "fit_after": snapshot_index,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                continue
            train_x = snapshot["train_x"].to(device=DEVICE, dtype=torch.double)
            train_y = snapshot["train_y"].to(device=DEVICE, dtype=torch.double)
            train_score = snapshot["train_score"].to(
                device=DEVICE, dtype=torch.double
            )
            response_variance = float(
                torch.var(train_score.reshape(-1), correction=1).item()
            )
            response_sd = torch.tensor(
                math.sqrt(response_variance), device=DEVICE, dtype=torch.double
            )
            candidates = [("learned", None, None)] + [
                (
                    f"fixed_{value:.0e}",
                    value,
                    standardized_to_raw_noise_variance(value, train_score),
                )
                for value in standardized_variances
            ]
            for candidate, requested_standardized, requested_raw in candidates:
                for initialization_id in range(int(initializations)):
                    fit_id = _fit_identifier(
                        run.seed, snapshot_index, candidate, initialization_id
                    )
                    if fit_id in completed_ids:
                        continue
                    rows = [row for row in rows if row.get("fit_id") != fit_id]
                    state_name = f"{fit_id}.pt"
                    row = {
                        "fit_id": fit_id,
                        "seed": run.seed,
                        "run_directory": str(run.directory),
                        "run_id": run.manifest.get("run_id"),
                        "source_archive_sha256": run.archive_sha256,
                        "source_configuration_sha256": run.manifest.get(
                            "configuration_sha256"
                        ),
                        "source_gaussian_sha256": _source_gaussian_hash(run),
                        "fit_after_bo_iteration": snapshot_index,
                        "train_count": len(train_x),
                        "archive_stop_exclusive": snapshot[
                            "archive_stop_exclusive"
                        ],
                        "holdout_start_bo_iteration": snapshot["holdout_start"],
                        "holdout_stop_bo_iteration": snapshot["holdout_stop"],
                        "holdout_count": len(snapshot["holdout_x"]),
                        "probe_count": len(probe_x),
                        "noise_mode": (
                            "learned" if requested_raw is None else "prescribed"
                        ),
                        "noise_candidate": candidate,
                        "requested_standardized_noise_variance": (
                            requested_standardized
                        ),
                        "requested_raw_response_noise_variance": requested_raw,
                        "training_response_sample_variance": response_variance,
                        "initialization_id": initialization_id,
                        "initialization_type": "controlled_cold",
                        "warm_start_used": False,
                        "warm_start_source_fit_id": None,
                        "status": "running",
                        "fit_state_file": None,
                    }
                    started = time.monotonic()
                    caught = []
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            gp, _, initialization = construct_cold_study_model(
                                train_x=train_x,
                                train_y=train_y,
                                nholes=run.nholes,
                                observation_noise_variance=requested_raw,
                                seed=run.seed,
                                fit_after=snapshot_index,
                                initialization_id=initialization_id,
                                perturbation_scale=perturbation_scale,
                            )
                        row.update(initialization)
                        fit_objective_gp(gp, max_attempts=1)
                        lengthscales, outputscales = _fitted_hyperparameters(gp)
                        row.update(
                            _predictive_metrics(
                                gp,
                                probe_x.to(device=DEVICE, dtype=torch.double),
                                snapshot["holdout_x"].to(
                                    device=DEVICE, dtype=torch.double
                                ),
                                snapshot["holdout_score"].to(
                                    device=DEVICE, dtype=torch.double
                                ),
                                response_sd,
                            )
                        )
                        effective = row[
                            "effective_standardized_noise_variance"
                        ]
                        row["fixed_noise_floor_applied"] = (
                            None
                            if requested_standardized is None
                            else effective
                            > requested_standardized * (1 + 1e-8)
                        )
                        row.update(
                            {
                                "status": "success",
                                "final_marginal_log_likelihood": _final_mll(gp),
                                "fitted_lengthscales": lengthscales,
                                "fitted_outputscales": outputscales,
                                "fitted_likelihood_noise_variance_standardized": (
                                    effective
                                ),
                            }
                        )
                        atomic_torch_save(
                            {
                                "schema_version": STUDY_SCHEMA_VERSION,
                                "fit_id": fit_id,
                                "initialization_type": "controlled_cold",
                                "warm_start_used": False,
                                "source_archive": str(run.directory),
                                "source_archive_sha256": run.archive_sha256,
                                "archive_stop_exclusive": snapshot[
                                    "archive_stop_exclusive"
                                ],
                                "initialization": initialization,
                                "model_state_dict": gp.state_dict(),
                            },
                            state_directory / state_name,
                        )
                        row["fit_state_file"] = f"fit_states/{state_name}"
                    except Exception as error:  # persist the remaining study
                        row.update(
                            {
                                "status": "failed",
                                "failure_type": type(error).__name__,
                                "failure_message": str(error),
                            }
                        )
                    row["duration_seconds"] = time.monotonic() - started
                    row["warning_count"] = len(caught)
                    row["warnings"] = [
                        {
                            "category": warning.category.__name__,
                            "message": str(warning.message),
                        }
                        for warning in caught
                    ]
                    rows.append(row)
                    rows.sort(key=lambda item: item["fit_id"])
                    _write_fit_rows(rows, output_directory)
    failed_fit_count = sum(row.get("status") != "success" for row in rows)
    manifest["status"] = (
        "complete"
        if not failed_fit_count and not failures
        else "complete_with_failures"
    )
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["fit_count"] = len(rows)
    manifest["successful_fit_count"] = sum(
        row.get("status") == "success" for row in rows
    )
    manifest["failed_fit_count"] = failed_fit_count
    manifest["skipped_snapshots"] = failures
    atomic_json_write(manifest, manifest_path)
    return rows


def _number(row, field):
    value = row.get(field)
    if value in (None, "", "N/A"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _standard_error(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def _group_summary(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        groups.setdefault(key, []).append(row)
    summaries = []
    metrics = (
        "heldout_rmse",
        "heldout_mae",
        "heldout_negative_log_predictive_density",
        "heldout_rms_standardized_residual",
        "heldout_95_percent_coverage",
        "probe_mean_spread_ratio",
        "probe_latent_sd_ratio_q90",
        "effective_standardized_noise_variance",
    )
    for key, group in sorted(groups.items()):
        successful = [row for row in group if row["status"] == "success"]
        summary = dict(zip(keys, key))
        summary.update(
            {
                "fit_count": len(group),
                "success_count": len(successful),
                "success_rate": len(successful) / len(group),
                "warning_fit_rate": sum(
                    (_number(row, "warning_count") or 0) > 0 for row in group
                )
                / len(group),
            }
        )
        for metric in metrics:
            values = [_number(row, metric) for row in successful]
            summary[f"median_{metric}"] = _median(values)
            summary[f"mean_{metric}"] = _mean(values)
            summary[f"se_{metric}"] = _standard_error(values)
            finite = [value for value in values if value is not None]
            summary[f"range_{metric}"] = (
                None if not finite else max(finite) - min(finite)
            )
        summaries.append(summary)
    return summaries


def _write_flexible_csv(rows, path):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    atomic_csv_write(
        [{field: _csv_value(row.get(field)) for field in fields} for row in rows],
        fields,
        path,
    )


def summarize_numerical_noise_study(output_directory, *, make_plots=True):
    """Aggregate fit outcomes without confusing fit, seed and point variance."""
    output_directory = Path(output_directory)
    rows = _read_fit_rows(output_directory)
    if not rows:
        raise FileNotFoundError("No gp_fits_numerical_study.csv was found.")
    by_seed_snapshot = _group_summary(
        rows, ("noise_candidate", "seed", "fit_after_bo_iteration")
    )
    by_seed = _group_summary(rows, ("noise_candidate", "seed"))
    by_candidate = _group_summary(rows, ("noise_candidate",))
    _write_flexible_csv(
        by_seed_snapshot, output_directory / "summary_by_seed_snapshot.csv"
    )
    _write_flexible_csv(by_seed, output_directory / "summary_by_seed.csv")
    _write_flexible_csv(
        by_candidate, output_directory / "summary_by_candidate.csv"
    )
    if make_plots:
        try:
            import matplotlib.pyplot as plt

            labels = [row["noise_candidate"] for row in by_candidate]
            rmse = [row["median_heldout_rmse"] for row in by_candidate]
            nlpd = [
                row["median_heldout_negative_log_predictive_density"]
                for row in by_candidate
            ]
            success = [row["success_rate"] for row in by_candidate]
            figure, axes = plt.subplots(1, 3, figsize=(12, 4))
            for axis, values, title in zip(
                axes,
                (rmse, nlpd, success),
                ("Held-out RMSE", "Held-out NLPD", "Fit success rate"),
            ):
                axis.bar(labels, [math.nan if x is None else x for x in values])
                axis.set_title(title)
                axis.tick_params(axis="x", rotation=30)
            figure.tight_layout()
            figure.savefig(
                output_directory / "numerical_noise_summary.png", dpi=180
            )
            plt.close(figure)
        except (ImportError, ValueError):
            pass
    return {
        "by_seed_snapshot": by_seed_snapshot,
        "by_seed": by_seed,
        "by_candidate": by_candidate,
    }


def decide_provisional_fixed_variance(
    output_directory,
    *,
    minimum_success_rate=0.95,
    maximum_warning_fit_rate=0.25,
    maximum_rmse_ratio_to_control=1.25,
    minimum_mean_spread_ratio_to_control=0.25,
    minimum_latent_sd_ratio_to_control=0.05,
):
    """Select a provisional fixed variance, or explicitly decline to select.

    Eligibility requires numerical stability, held-out accuracy and useful
    posterior variation relative to the learned-noise control. Among eligible
    candidates, the one-standard-error set is formed using held-out NLPD; fit
    stability breaks ties before the larger (more regularising) variance.
    """
    output_directory = Path(output_directory)
    rows = _read_fit_rows(output_directory)
    summaries = summarize_numerical_noise_study(
        output_directory, make_plots=False
    )["by_seed_snapshot"]
    control_groups = {
        (row["seed"], row["fit_after_bo_iteration"]): row
        for row in summaries
        if row["noise_candidate"] == "learned"
    }
    fixed_candidates = sorted(
        {
            row["noise_candidate"]
            for row in rows
            if row["noise_candidate"].startswith("fixed_")
        },
        key=lambda label: float(label.removeprefix("fixed_")),
    )
    assessments = []
    for candidate in fixed_candidates:
        candidate_rows = [row for row in rows if row["noise_candidate"] == candidate]
        group_rows = [
            row for row in summaries if row["noise_candidate"] == candidate
        ]
        reasons = []
        success_rate = sum(row["status"] == "success" for row in candidate_rows) / len(
            candidate_rows
        )
        warning_rate = sum(
            (_number(row, "warning_count") or 0) > 0 for row in candidate_rows
        ) / len(candidate_rows)
        if success_rate < minimum_success_rate:
            reasons.append("fit_success_rate_below_threshold")
        if warning_rate > maximum_warning_fit_rate:
            reasons.append("warning_fit_rate_above_threshold")
        paired_nlpd = []
        paired_rmse = []
        initialization_spreads = []
        missing_controls = 0
        for group in group_rows:
            key = (group["seed"], group["fit_after_bo_iteration"])
            control = control_groups.get(key)
            if control is None:
                missing_controls += 1
                continue
            rmse = group["median_heldout_rmse"]
            control_rmse = control["median_heldout_rmse"]
            mean_spread = group["median_probe_mean_spread_ratio"]
            control_mean_spread = control["median_probe_mean_spread_ratio"]
            latent_sd = group["median_probe_latent_sd_ratio_q90"]
            control_latent_sd = control["median_probe_latent_sd_ratio_q90"]
            if None in (rmse, control_rmse) or rmse > (
                maximum_rmse_ratio_to_control * max(control_rmse, 1e-300)
            ):
                reasons.append(f"heldout_rmse_inadequate_at_{key[0]}_{key[1]}")
            if (
                None in (mean_spread, control_mean_spread)
                or mean_spread
                < minimum_mean_spread_ratio_to_control * control_mean_spread
            ):
                reasons.append(f"posterior_mean_collapsed_at_{key[0]}_{key[1]}")
            if (
                None in (latent_sd, control_latent_sd)
                or latent_sd < minimum_latent_sd_ratio_to_control * control_latent_sd
            ):
                reasons.append(f"posterior_sd_collapsed_at_{key[0]}_{key[1]}")
            paired_nlpd.append(
                group["median_heldout_negative_log_predictive_density"]
            )
            paired_rmse.append(rmse)
            initialization_spreads.append(group["range_heldout_rmse"])
        if missing_controls:
            reasons.append("missing_learned_control_groups")
        nlpd_values = [value for value in paired_nlpd if value is not None]
        if not nlpd_values:
            reasons.append("no_finite_heldout_nlpd")
        assessments.append(
            {
                "noise_candidate": candidate,
                "requested_standardized_noise_variance": float(
                    candidate.removeprefix("fixed_")
                ),
                "eligible": not reasons,
                "ineligibility_reasons": sorted(set(reasons)),
                "success_rate": success_rate,
                "warning_fit_rate": warning_rate,
                "paired_group_count": len(nlpd_values),
                "mean_heldout_nlpd": _mean(nlpd_values),
                "se_heldout_nlpd": _standard_error(nlpd_values),
                "median_heldout_rmse": _median(paired_rmse),
                "median_initialization_rmse_range": _median(
                    initialization_spreads
                ),
            }
        )
    eligible = [item for item in assessments if item["eligible"]]
    if not eligible:
        decision = {
            "status": "no_selection",
            "selected_standardized_noise_variance": None,
            "rationale": (
                "No fixed variance satisfied fit stability, paired held-out "
                "accuracy, and posterior-variation requirements."
            ),
            "candidate_assessments": assessments,
        }
    else:
        best = min(eligible, key=lambda item: item["mean_heldout_nlpd"])
        allowance = best["se_heldout_nlpd"] or 0.0
        one_se_set = [
            item
            for item in eligible
            if item["mean_heldout_nlpd"]
            <= best["mean_heldout_nlpd"] + allowance
        ]
        selected = min(
            one_se_set,
            key=lambda item: (
                item["warning_fit_rate"],
                item["median_initialization_rmse_range"]
                if item["median_initialization_rmse_range"] is not None
                else math.inf,
                -item["requested_standardized_noise_variance"],
            ),
        )
        decision = {
            "status": "provisional_selection",
            "selected_standardized_noise_variance": selected[
                "requested_standardized_noise_variance"
            ],
            "selected_candidate": selected["noise_candidate"],
            "one_standard_error_candidates": [
                item["noise_candidate"] for item in one_se_set
            ],
            "rationale": (
                "Eligible candidates were compared by paired held-out NLPD. "
                "Within one standard error of the best, warning rate and "
                "initialization sensitivity were preferred; larger variance "
                "was used only as the final regularising tie-break."
            ),
            "candidate_assessments": assessments,
        }
    decision["selection_is_provisional"] = True
    decision["learned_noise_remains_control"] = True
    decision["eligibility_thresholds"] = {
        "minimum_success_rate": minimum_success_rate,
        "maximum_warning_fit_rate": maximum_warning_fit_rate,
        "maximum_rmse_ratio_to_control": maximum_rmse_ratio_to_control,
        "minimum_mean_spread_ratio_to_control": (
            minimum_mean_spread_ratio_to_control
        ),
        "minimum_latent_sd_ratio_to_control": minimum_latent_sd_ratio_to_control,
    }
    atomic_json_write(decision, output_directory / "decision.json")
    return decision


def _parse_runs(values):
    result = {}
    for value in values:
        try:
            seed_text, path = value.split("=", 1)
            seed = int(seed_text)
        except (ValueError, TypeError) as error:
            raise argparse.ArgumentTypeError(
                f"Run {value!r} must have the form SEED=PATH."
            ) from error
        if seed in result:
            raise argparse.ArgumentTypeError(f"Seed {seed} was supplied twice.")
        result[seed] = path
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="fit all cold-start study models")
    run.add_argument("--run", action="append", required=True, metavar="SEED=PATH")
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--standardized-variance",
        nargs="+",
        type=float,
        default=list(DEFAULT_STANDARDIZED_VARIANCES),
    )
    run.add_argument(
        "--fit-after", nargs="+", type=int, default=list(DEFAULT_FIT_AFTER)
    )
    run.add_argument("--holdout-size", type=int, default=10)
    run.add_argument("--initializations", type=int, default=3)
    run.add_argument("--perturbation-scale", type=float, default=0.25)
    run.add_argument("--expected-seed-count", type=int, default=5)
    run.add_argument("--allow-partial", action="store_true")
    run.add_argument("--allow-incompatible-sources", action="store_true")
    summarize = subparsers.add_parser("summarize", help="aggregate saved fits")
    summarize.add_argument("output_dir")
    summarize.add_argument("--no-plots", action="store_true")
    decide = subparsers.add_parser("decide", help="cast a provisional decision")
    decide.add_argument("output_dir")
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        rows = run_numerical_noise_study(
            _parse_runs(arguments.run),
            arguments.output_dir,
            standardized_variances=arguments.standardized_variance,
            fit_after=arguments.fit_after,
            holdout_size=arguments.holdout_size,
            initializations=arguments.initializations,
            perturbation_scale=arguments.perturbation_scale,
            expected_seed_count=arguments.expected_seed_count,
            allow_partial=arguments.allow_partial,
            allow_incompatible_sources=arguments.allow_incompatible_sources,
        )
        print(f"Recorded {len(rows)} study fits in {arguments.output_dir}")
    elif arguments.command == "summarize":
        result = summarize_numerical_noise_study(
            arguments.output_dir, make_plots=not arguments.no_plots
        )
        print(f"Summarized {len(result['by_candidate'])} noise candidates")
    else:
        decision = decide_provisional_fixed_variance(arguments.output_dir)
        print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
