"""Symmetry-invariant GP and distance features for ellipse designs."""

from __future__ import annotations

import hashlib
import math

import torch


RAW_ELLIPSE_FEATURES = "raw"
SHAPE_MATRIX_ELLIPSE_FEATURES = "shape_matrix"
ELLIPSE_FEATURE_MODES = (
    RAW_ELLIPSE_FEATURES,
    SHAPE_MATRIX_ELLIPSE_FEATURES,
)
ELLIPSE_FEATURE_SCHEMA_VERSION = 1


def validate_ellipse_feature_mode(mode: str) -> str:
    """Return a supported ellipse feature mode or raise a clear error."""
    if mode not in ELLIPSE_FEATURE_MODES:
        choices = ", ".join(repr(value) for value in ELLIPSE_FEATURE_MODES)
        raise ValueError(f"ellipse_feature_mode must be one of {choices}.")
    return mode


def _symmetric_matrix_log_entries(matrix_00, matrix_01, matrix_11):
    """Return the log of batched 2-by-2 SPD matrices without eigendecomposition.

    The analytic isotropic limit avoids the undefined eigenvector gradients of
    a generic eigendecomposition when an ellipse is circular.
    """
    dtype = matrix_00.dtype
    epsilon = max(float(torch.finfo(dtype).eps), 1e-15)
    mean = 0.5 * (matrix_00 + matrix_11)
    half_difference = 0.5 * (matrix_00 - matrix_11)
    radius_squared = half_difference.square() + matrix_01.square()
    determinant = (matrix_00 * matrix_11 - matrix_01.square()).clamp_min(
        epsilon
    )
    mean = mean.clamp_min(math.sqrt(epsilon))
    relative_squared = (radius_squared / mean.square()).clamp(
        min=0.0, max=1.0 - epsilon
    )

    # atanh(z) / z = 1 + z^2/3 + z^4/5 + ... around the circular limit.
    series = (
        1.0
        + relative_squared / 3.0
        + relative_squared.square() / 5.0
        + relative_squared.pow(3) / 7.0
    ) / mean
    relative = relative_squared.clamp_min(epsilon).sqrt()
    general = torch.atanh(relative) / (mean * relative)
    coefficient = torch.where(relative_squared < 1e-8, series, general)
    mean_log_eigenvalue = 0.5 * torch.log(determinant)

    return (
        mean_log_eigenvalue + coefficient * half_difference,
        coefficient * matrix_01,
        mean_log_eigenvalue - coefficient * half_difference,
    )


def shape_matrix_design_features(designs, nholes: int, xsize, ysize):
    """Map ellipse blocks to symmetry-invariant log shape-matrix features.

    Each raw seven-value block remains seven-dimensional so the existing mixed
    GP's categorical indices are unchanged.  Ellipse blocks become
    ``[shape, cx, cy, L11, sqrt(2)L12, L22, 0]``, where ``L`` is the matrix
    logarithm of the domain-normalized physical shape matrix.  Triangle blocks
    are returned unchanged, which keeps this helper safe for future mixed-shape
    inputs; production triangle runs do not install this transform at all.
    """
    tensor = torch.as_tensor(designs)
    if tensor.shape[-1] != 7 * int(nholes):
        raise ValueError(
            f"Expected a final design dimension of {7 * int(nholes)}, "
            f"received {tensor.shape[-1]}."
        )
    xsize_value = torch.as_tensor(xsize, dtype=tensor.dtype, device=tensor.device)
    ysize_value = torch.as_tensor(ysize, dtype=tensor.dtype, device=tensor.device)
    if bool((xsize_value <= 0).item()) or bool((ysize_value <= 0).item()):
        raise ValueError("xsize and ysize must be positive.")

    blocks = tensor.reshape(*tensor.shape[:-1], int(nholes), 7)
    angle = math.pi * blocks[..., 5]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    semi_axis_a = xsize_value * blocks[..., 3]
    semi_axis_b = ysize_value * blocks[..., 4]
    a_squared, b_squared = semi_axis_a.square(), semi_axis_b.square()

    physical_00 = cosine.square() * a_squared + sine.square() * b_squared
    physical_01 = cosine * sine * (a_squared - b_squared)
    physical_11 = sine.square() * a_squared + cosine.square() * b_squared
    normalized_00 = physical_00 / xsize_value.square()
    normalized_01 = physical_01 / (xsize_value * ysize_value)
    normalized_11 = physical_11 / ysize_value.square()
    log_00, log_01, log_11 = _symmetric_matrix_log_entries(
        normalized_00, normalized_01, normalized_11
    )

    ellipse_blocks = torch.stack(
        (
            blocks[..., 0],
            blocks[..., 1],
            blocks[..., 2],
            log_00,
            math.sqrt(2.0) * log_01,
            log_11,
            torch.zeros_like(blocks[..., 6]),
        ),
        dim=-1,
    )
    transformed = torch.where(
        (blocks[..., 0] >= 0.5).unsqueeze(-1), ellipse_blocks, blocks
    )
    return transformed.reshape_as(tensor)


def design_features(designs, nholes: int, xsize, ysize, mode: str = "raw"):
    """Return raw or symmetry-invariant features under an explicit mode."""
    validate_ellipse_feature_mode(mode)
    if mode == RAW_ELLIPSE_FEATURES:
        return torch.as_tensor(designs)
    return shape_matrix_design_features(designs, nholes, xsize, ysize)


def ellipse_feature_metadata(mode: str, nholes: int, xsize, ysize):
    """Return serializable parameters needed to reproduce input features."""
    validate_ellipse_feature_mode(mode)
    return {
        "mode": mode,
        "schema_version": ELLIPSE_FEATURE_SCHEMA_VERSION,
        "nholes": int(nholes),
        "xsize": float(xsize),
        "ysize": float(ysize),
    }


def tensor_sha256(tensor) -> str:
    """Return a stable checksum for one losslessly stored feature tensor."""
    value = torch.as_tensor(tensor).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()
