"""Rasterisation and finite-element compliance calculations."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .runtime import DEVICE, TENSOR_KWARGS
from .shapes import area_ellipse, area_triangle, read_shape


VALID_DISPLACEMENT_DIRECTIONS = ("u", "v")
VALID_SUPPORT_DOFS = ("u", "v", "both")


def lk() -> np.ndarray:
    """Element stiffness matrix for a unit-modulus square Q4 element."""
    nu = 0.3
    a11 = np.array(
        [[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]]
    )
    a12 = np.array(
        [[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]]
    )
    b11 = np.array(
        [[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]]
    )
    b12 = np.array(
        [[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]]
    )
    return (
        np.block([[a11, a12], [a12.T, a11]])
        + nu * np.block([[b11, b12], [b12.T, b11]])
    ) / (1 - nu**2) / 24


def element_dofs(nelx: int, nely: int):
    nodenrs = np.arange((nelx + 1) * (nely + 1)).reshape(
        (nely + 1, nelx + 1), order="F"
    )
    edof_vec = 2 * nodenrs[:-1, :-1].reshape(nelx * nely, order="F") + 2
    offsets = np.array(
        [0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1]
    )
    return (edof_vec[:, None] + offsets[None, :]).astype(int), nodenrs


def _node_number(x: int, y: int, nely: int) -> int:
    """Return the column-major node number for integer grid coordinates."""
    return x * (nely + 1) + y


def _normalise_support_conditions(support_conditions):
    if support_conditions is None:
        return None
    return tuple((int(x), int(y), str(dof)) for x, y, dof in support_conditions)


def _normalise_load_conditions(load_conditions):
    if load_conditions is None:
        return None
    return tuple(
        (int(x), int(y), str(direction), float(magnitude))
        for x, y, direction, magnitude in load_conditions
    )


def default_loads_and_supports(nelx: int, nely: int):
    """Half-MBB load and symmetry/roller supports used by the reference."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    loads = [(nely, 1, -1.0)]
    fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), np.array([ndof - 1]))
    return loads, fixed.astype(int)


def configured_loads_and_supports(
    nelx: int,
    nely: int,
    *,
    beam_model: str = "half",
    support_conditions=None,
    load_conditions=None,
):
    """Build nodal loads and constrained DOFs from grid-coordinate settings.

    ``u`` and ``v`` denote horizontal and vertical displacement respectively.
    Half-beam models always apply the legacy symmetry constraint ``u=0`` along
    the complete left edge. Full-beam models add no automatic constraint.
    """
    if beam_model not in ("half", "full"):
        raise ValueError("beam_model must be 'half' or 'full'.")
    supports = _normalise_support_conditions(support_conditions)
    loads = _normalise_load_conditions(load_conditions)
    if beam_model == "full" and supports is None:
        raise ValueError("Full-beam models require explicit support_conditions.")
    if beam_model == "full" and loads is None:
        raise ValueError("Full-beam models require explicit load_conditions.")

    if beam_model == "half" and supports is None and loads is None:
        return default_loads_and_supports(nelx, nely)

    fixed = []
    if beam_model == "half":
        fixed.extend(2 * _node_number(0, y, nely) for y in range(nely + 1))
        if supports is None:
            supports = ((nelx, nely, "v"),)
        if loads is None:
            loads = ((0, nely, "v", -1.0),)

    for x, y, dof in supports or ():
        if not 0 <= x <= nelx or not 0 <= y <= nely:
            raise ValueError("Support coordinates must lie on the FEM node grid.")
        if dof not in VALID_SUPPORT_DOFS:
            raise ValueError("Support dof must be 'u', 'v', or 'both'.")
        node = _node_number(x, y, nely)
        if dof in ("u", "both"):
            fixed.append(2 * node)
        if dof in ("v", "both"):
            fixed.append(2 * node + 1)

    nodal_loads = []
    for x, y, direction, magnitude in loads or ():
        if not 0 <= x <= nelx or not 0 <= y <= nely:
            raise ValueError("Load coordinates must lie on the FEM node grid.")
        if direction not in VALID_DISPLACEMENT_DIRECTIONS:
            raise ValueError("Load direction must be 'u' or 'v'.")
        if not np.isfinite(magnitude) or magnitude == 0:
            raise ValueError("Load magnitude must be finite and non-zero.")
        node = _node_number(x, y, nely)
        dof = 2 * node + (direction == "v")
        nodal_loads.append((node, int(dof % 2), magnitude))
    return nodal_loads, np.unique(np.asarray(fixed, dtype=int))


@lru_cache(maxsize=None)
def _cached_fem(
    nelx: int,
    nely: int,
    beam_model: str,
    support_conditions,
    load_conditions,
):
    stiffness = lk()
    edof_mat, _ = element_dofs(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)
    ik = np.kron(edof_mat, np.ones((8, 1), dtype=int)).ravel()
    jk = np.kron(edof_mat, np.ones((1, 8), dtype=int)).ravel()
    loads, fixed = configured_loads_and_supports(
        nelx,
        nely,
        beam_model=beam_model,
        support_conditions=support_conditions,
        load_conditions=load_conditions,
    )
    free = np.setdiff1d(np.arange(ndof), fixed)
    load_vector = np.zeros(ndof)
    for node, dof, value in loads:
        load_vector[2 * node + dof] += value
    return stiffness, edof_mat, ndof, ik, jk, free, load_vector


def cached_fem(
    nelx: int,
    nely: int,
    beam_model: str = "half",
    support_conditions=None,
    load_conditions=None,
):
    """Return a cached FEM system for hashable normalized boundary settings."""
    return _cached_fem(
        nelx,
        nely,
        beam_model,
        _normalise_support_conditions(support_conditions),
        _normalise_load_conditions(load_conditions),
    )


def true_objective(
    x_phys,
    nelx: int,
    nely: int,
    emin: float,
    e0: float,
    penal: float,
    *,
    beam_model: str = "half",
    support_conditions=None,
    load_conditions=None,
):
    stiffness, edof_mat, ndof, ik, jk, free, load_vector = cached_fem(
        nelx,
        nely,
        beam_model,
        support_conditions,
        load_conditions,
    )
    young = emin + x_phys.ravel(order="F") ** penal * (e0 - emin)
    sk = (stiffness.ravel(order="F")[:, None] * young[None, :]).ravel(order="F")
    matrix = sp.coo_matrix((sk, (ik, jk)), shape=(ndof, ndof)).tocsc()
    matrix = (matrix + matrix.T) / 2
    displacement = np.zeros(ndof)
    displacement[free] = spla.spsolve(
        matrix[free[:, None], free], load_vector[free]
    )
    element_displacement = displacement[edof_mat]
    energy = np.sum(
        (element_displacement @ stiffness) * element_displacement, axis=1
    ).reshape((nely, nelx), order="F")
    return float(np.sum((emin + x_phys**penal * (e0 - emin)) * energy))


def analytical_solid_fraction(X, constraints):
    designs = torch.as_tensor(X, **TENSOR_KWARGS)
    if designs.ndim == 1:
        designs = designs.unsqueeze(0)
    fractions = []
    for design in designs:
        triangles, ellipses = read_shape(
            design, constraints.nholes, constraints.xsize, constraints.ysize
        )
        hole_area = sum(area_triangle(triangles)) + sum(area_ellipse(ellipses))
        fractions.append(1.0 - hole_area / (constraints.xsize * constraints.ysize))
    return fractions


def proposal_solid_fraction(constraints, unit_sample, solid_sampling_band=0.05):
    proposal_min = max(0.0, constraints.solid_max - solid_sampling_band)
    interior_sample = 0.05 + 0.90 * unit_sample
    return proposal_min + (constraints.solid_max - proposal_min) * interior_sample


def raster_grid(nelx: int, nely: int):
    return _raster_grid(nelx, nely, str(DEVICE), str(TENSOR_KWARGS["dtype"]))


@lru_cache(maxsize=None)
def _raster_grid(nelx: int, nely: int, _device: str, _dtype: str):
    xx, yy = torch.meshgrid(
        torch.arange(nelx, **TENSOR_KWARGS) + 0.5,
        torch.arange(nely, **TENSOR_KWARGS) + 0.5,
        indexing="xy",
    )
    return torch.stack((xx, yy), dim=-1)


def generate_geometry(
    nelx: int,
    nely: int,
    X,
    nholes: int,
    plot: bool = False,
    namesave: str | bool = False,
    title: str = "Proposed Geometry",
    dir_save=False,
    fol_save=False,
    show: bool = False,
) -> np.ndarray:
    design = torch.as_tensor(X, **TENSOR_KWARGS).detach().reshape(-1)
    expected = 7 * nholes
    if design.numel() != expected:
        raise ValueError(f"Expected {expected} design variables, received {design.numel()}.")

    scale = design.new_tensor([nelx, nely])
    points = raster_grid(nelx, nely)
    x_phys = torch.ones((nely, nelx), **TENSOR_KWARGS)
    for hole in range(nholes):
        shape = design[7 * hole : 7 * (hole + 1)]
        if shape[0] < 0.5:
            p0 = shape[1:3] * scale
            vector_1 = (2.0 * shape[3:5] - 1.0) * scale
            vector_2 = (2.0 * shape[5:7] - 1.0) * scale
            relative = points - p0
            denominator = vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0]
            if torch.abs(denominator) <= 1e-12:
                continue
            bary_1 = (
                relative[..., 0] * vector_2[1] - vector_2[0] * relative[..., 1]
            ) / denominator
            bary_2 = (
                relative[..., 1] * vector_1[0] - vector_1[1] * relative[..., 0]
            ) / denominator
            inside = (
                (bary_1 >= 0.0)
                & (bary_2 >= 0.0)
                & (bary_1 + bary_2 <= 1.0)
            )
        else:
            cx, cy, rx, ry, theta = shape[1:-1]
            theta = torch.pi * theta
            cx, cy = cx * scale[0], cy * scale[1]
            rx, ry = rx * scale[0], ry * scale[1]
            if rx <= 1e-12 or ry <= 1e-12:
                continue
            dx, dy = points[..., 0] - cx, points[..., 1] - cy
            u = dx * torch.cos(theta) + dy * torch.sin(theta)
            v = -dx * torch.sin(theta) + dy * torch.cos(theta)
            inside = u**2 / rx**2 + v**2 / ry**2 <= 1.0
        x_phys[inside] = 0.0

    geometry = x_phys.detach().cpu().numpy()
    if plot:
        figure, axis = plt.subplots()
        axis.imshow(1 - geometry, cmap="gray", origin="lower")
        axis.set_title(title)
        axis.axis("image")
        axis.set_xticks([])
        axis.set_yticks([])
        if namesave:
            save_directory = Path(dir_save) / fol_save if dir_save else Path(fol_save or ".")
            save_directory.mkdir(parents=True, exist_ok=True)
            figure.savefig(save_directory / f"{namesave}.png", dpi=500)
        if show:
            plt.show()
        plt.close(figure)
    return geometry


def rasterized_volume_feasibility(X, constraints):
    designs = torch.as_tensor(X, **TENSOR_KWARGS)
    if designs.ndim == 1:
        designs = designs.unsqueeze(0)
    fractions = torch.tensor(
        [
            float(
                np.mean(
                    generate_geometry(
                        constraints.xsize,
                        constraints.ysize,
                        design,
                        constraints.nholes,
                    )
                )
            )
            for design in designs
        ],
        **TENSOR_KWARGS,
    )
    return fractions <= constraints.solid_max + constraints.raster_tolerance, fractions


def geometry_signature(design, constraints):
    geometry = generate_geometry(
        constraints.xsize,
        constraints.ysize,
        design,
        constraints.nholes,
    )
    return np.packbits(geometry.astype(np.uint8), axis=None).tobytes()


def evaluate_designs(
    X,
    nelx: int,
    nely: int,
    nholes: int,
    emin: float,
    e0: float,
    penal: float,
    constraints,
    return_solid_fractions: bool = False,
    *,
    beam_model: str = "half",
    support_conditions=None,
    load_conditions=None,
):
    designs = torch.as_tensor(X, **TENSOR_KWARGS)
    if designs.ndim == 1:
        designs = designs.unsqueeze(0)
    if designs.ndim != 2 or designs.shape[-1] != 7 * nholes:
        raise ValueError(
            f"X must have shape (batch, {7 * nholes}); received {tuple(designs.shape)}."
        )

    compliances, fractions = [], []
    for design in designs:
        geometry = generate_geometry(nelx, nely, design, nholes)
        fraction = float(np.mean(geometry))
        permitted = constraints.solid_max + constraints.raster_tolerance
        if fraction > permitted:
            raise RuntimeError(
                "Rasterized design exceeds the upper solid-volume limit: "
                f"rasterized={fraction:.6f}, permitted={permitted:.6f}."
            )
        compliance = true_objective(
            geometry,
            nelx,
            nely,
            emin,
            e0,
            penal,
            beam_model=beam_model,
            support_conditions=support_conditions,
            load_conditions=load_conditions,
        )
        compliances.append(compliance if np.isfinite(compliance) else 1e30)
        fractions.append(fraction)

    result = torch.tensor(compliances, **TENSOR_KWARGS).unsqueeze(-1)
    if return_solid_fractions:
        return result, torch.tensor(fractions, **TENSOR_KWARGS)
    return result


# Compatibility aliases for the naming used by the historical scripts.
True_Objective = true_objective
Generate_Geometry = generate_geometry
Evaluate_Designs = evaluate_designs
