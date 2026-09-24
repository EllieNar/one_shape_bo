# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:31:42 2026

@author: pemb6626
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Sequence, Literal, Tuple
from pathlib import Path
from Functions.Shape import Read_Shape, Area_Triangle, Area_Ellipse

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {
    "device": device,
    "dtype": torch.double,
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def lk() -> np.ndarray:
    """Element stiffness matrix for a unit Young's modulus square Q4 element."""
    # This integral is determined through Gauss quadrature (see 260701_Gaussian_Quadrature_2D/py)
    nu = 0.3
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    return (np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]])) / (1 - nu**2) / 24

def element_dofs(nelx: int, nely: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return element degree-of-freedom matrix and node numbering."""
    nodenrs = np.arange((nelx + 1) * (nely + 1)).reshape((nely + 1, nelx + 1), order="F")
    edof_vec = 2 * nodenrs[:-1, :-1].reshape(nelx * nely, order="F") + 2
    offsets = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    edof_mat = edof_vec[:, None] + offsets[None, :]
    return edof_mat.astype(int), nodenrs

def default_loads_and_supports(nelx: int, nely: int):
    """Base boundary choices matching common variants in the paper."""
    " This assumes an MBB beam-type loading/supports"
    ndof = 2 * (nelx + 1) * (nely + 1)
    # Half MBB beam: downward load at upper-left node; symmetry/roller supports.
    loads = [(nely, 1, -1.0)]
    fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), np.array([ndof - 1]))
    return loads, fixed.astype(int)

_cached_fem_cache = {}
def cached_fem(nelx, nely):
    # The FE quantities which are constant per iteration are cached to save compute cost
    key = (nelx, nely)
    if key not in _cached_fem_cache:
        KE = lk()
        edof_mat, _ = element_dofs(nelx, nely)
        ndof        = 2*(nelx + 1)*(nely + 1)
        iK          = np.kron(edof_mat, np.ones((8, 1), dtype=int)).ravel()
        jK          = np.kron(edof_mat, np.ones((1, 8), dtype=int)).ravel()
        loads, fixed= default_loads_and_supports(nelx, nely)
        fixed       = fixed.astype(int)
        free        = np.setdiff1d(np.arange(ndof), fixed)
        F           = np.zeros(ndof)
        node, dof, value = loads[0]
        F[2*node + dof] = value
        _cached_fem_cache[key] = (KE, edof_mat, ndof, iK, jK, free, F)
        
    return _cached_fem_cache[key]

def True_Objective(xPhys, nelx, nely, Emin, E0, penal):
    # This is the compliance-directed objective, from the original MBB beam code
    KE, edof_mat, ndof, iK, jK, free, F = cached_fem(nelx, nely)
                                  
    young = Emin + xPhys.ravel(order="F") ** penal * (E0 - Emin)
    sK = (KE.ravel(order="F")[:, None] * young[None, :]).ravel(order="F")
    K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    K = (K + K.T) / 2
    U = np.zeros(ndof)
    U[free] = spla.spsolve(K[free[:, None], free], F[free])
    Ue = U[edof_mat]
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Determine objective: compliance
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    ce = np.sum((Ue @ KE) * Ue, axis=1).reshape((nely, nelx), order="F")
    c = np.sum((Emin + xPhys ** penal * (E0 - Emin)) * ce)
    
    return c

# Returns the exact triangle-area solid fraction (i.e. without rasterisation difference)    
def analytical_solid_fraction(X, constraints, Print=False):
    solid_fractions = []
    
    for iters, design in enumerate(X):
        T, E = Read_Shape(design, constraints.nholes, constraints.xsize, constraints.ysize)
        at      = Area_Triangle(T)
        ae      = Area_Ellipse(E)
        A       = sum(at) + sum(ae)
        sf = 1.0 - A / (constraints.xsize * constraints.ysize)
        solid_fractions.append(sf)
        
        if Print:
            print(f"Iteration {iters + 1}: Analytical Solid Fraction = {sf}")
    
    return solid_fractions

# Sample below the volume cap without necessity for lower bound constraint
def proposal_solid_fraction(constraints, unit_sample, solid_sampling_band=0.05):
    proposal_min        = max(0, constraints.solid_max - solid_sampling_band)
    interior_sample     = 0.05 + 0.90*unit_sample
    return proposal_min + (constraints.solid_max - proposal_min)*interior_sample

# Audit the actual binary grid fraction used by the objective
def rasterized_volume_feasibility(X, constraints):
    rasterized_fractions = []
    for design in X:
        geometry = Generate_Geometry(constraints.xsize, constraints.ysize, design, constraints.nholes, plot=False, namesave=False,)
        rasterized_fractions.append(float(np.mean(geometry)))
    
    rasterized_fractions = torch.tensor(rasterized_fractions, **tkwargs)
    permitted = constraints.solid_max + constraints.raster_tolerance
    valid = rasterized_fractions <= permitted
    return valid, rasterized_fractions

# Because a different X vector may have the same rasterised result, this gives it the same 'geometry signature'...
def geometry_signature(design, constraints):
    # Compare exact signature used to reject repeated binary geometries
    geometry = Generate_Geometry(constraints.xsize, constraints.ysize, design, constraints.nholes, plot=False, namesave=False,)
    return np.packbits(geometry.astype(np.uint8), axis=None).tobytes()

# Make the rasterised finite-element like grid
_raster_grid_cache = {}
def raster_grid(nelx, nely):
    # Cache element-centre co-ordinates shared by every geometry
    key = (nelx, nely, str(device), tkwargs["dtype"])
    if key not in _raster_grid_cache:
        xx, yy = torch.meshgrid(torch.arange(nelx, **tkwargs) + 0.5, torch.arange(nely, **tkwargs) + 0.5, indexing = "xy",)
        _raster_grid_cache[key] = torch.stack((xx, yy), dim=-1)
        
    return _raster_grid_cache[key]

def Generate_Geometry(nelx, nely, X, nholes, plot=False, namesave=False, title="Proposed Geometry", dir_save=False, fol_save=False) -> np.ndarray:
    
    """
    INPUTS
    ------
    X --> list tensor size nholes*7 (MUST BE xnew.squeeze(0))
    
    Be careful!! On test run, ellipses can collide with triangles even though the true geometry is with a spacing > 0
    Thus, ensure that smin is large enough to mitigate this!
    
    """
    
    # Generate Geometry is called within Evaluate Designs after x_new has been found
    # i.e., y_new, raster_new = Evaluate_Designs (etc)
    # The best design tensor xnew is recorded and given in its normalised form of dimension 7*nholes
    # So Generate_Geometry must first translate into REAL co-ordinates
    
    X = torch.as_tensor(X, **tkwargs).detach()
    expected_variables = 7*nholes
    
    if X.numel() != expected_variables:
        raise ValueError(f"Expected {expected_variables} design variables, but received {X.numel()}")
    X = X.reshape(-1)
    
    scale = X.new_tensor([nelx, nely])
    points = raster_grid(nelx, nely)
    xPhys = torch.ones((nely, nelx), **tkwargs)
    
    for i in range(nholes):
        shape = X[7*i:7*(i+1)]
        
        if shape[0] == 0: # triangle
            # shape is expected to be a list of dimension 7
            
            p0 = shape[1:3]*scale
            v1 = (2.0*shape[3:5] - 1.0)*scale
            v2 = (2.0*shape[5:7] - 1.0)*scale
            
            relative = points - p0
            denominator = v1[0]*v2[1] - v1[1]*v2[0]
            
            if torch.abs(denominator) <= 1e-12:
                continue
            
            bary_1 = (relative[..., 0]*v2[1] - v2[0]*relative[..., 1])/denominator
            bary_2 = (relative[..., 1]*v1[0] - v1[1]*relative[..., 0])/denominator
            inside = ((bary_1 >= 0.0) & (bary_2 >= 0.0) & (bary_1 + bary_2 <= 1.0))
        
        elif shape[0] == 1: # ellipse
            cx, cy, rx, ry, theta = shape[1:-1]
            theta = torch.pi*theta
            
            # Convert normalised ellipse parameters to grid coordinates
            cx = cx * scale[0]
            cy = cy * scale[1]
            rx = rx * scale[0]
            ry = ry * scale[1]
        
            # Ignore degenerate ellipses
            if rx <= 1e-12 or ry <= 1e-12:
                continue
            
            dx = points[..., 0] - cx
            dy = points[..., 1] - cy
            
            u = dx * torch.cos(theta) + dy * torch.sin(theta)
            v = -dx * torch.sin(theta) + dy * torch.cos(theta)
            inside = (u**2 / rx**2 + v**2 / ry**2) <= 1.0
        
        xPhys[inside] = 0.0
            
    geometry = xPhys.detach().cpu().numpy()
    
    # Plot is opt-in so doesn't waste compute power
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(1 - geometry, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("image")
        ax.set_xticks([])
        ax.set_yticks([])
        
        if namesave:
            save_directory = Path(dir_save) / fol_save if dir_save else Path(fol_save or ".")
            save_directory.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_directory / f"{namesave}.png", dpi=500)
        plt.show()
        
    return geometry

def Evaluate_Designs(
        X,
        nelx,
        nely,
        nholes,
        Emin,
        E0,
        penal,
        constraints,
        return_solid_fractions=False,
        Print=False,
        dir_save = False,
        fol_save = False):
    compliances = []
    rasterized_fractions = []
    
    X = torch.as_tensor(X, **tkwargs)
    if X.ndim == 1:
        X = X.unsqueeze(0)
    if X.ndim != 2 or X.shape[-1] != 7*nholes:
        raise ValueError(f"X must have shape (batch, {7*nholes}); received {tuple(X.shape)}.")
    
    for iters, design in enumerate(X):
        # CHANGED: objective evaluations never plot.  Compliance is evaluated
        # directly on the binary geometry controlled by the analytical bound.
        xPhys = Generate_Geometry(
            nelx,
            nely,
            design,
            nholes,
            plot=False,
            namesave=False,
            dir_save = False,
            fol_save = False,
        )
        rasterized_fraction = float(np.mean(xPhys))
        permitted = constraints.solid_max + constraints.raster_tolerance

        # CHANGED: never train the GP on a design whose actual binary grid
        # exceeds the audited upper material-volume limit.
        if rasterized_fraction > permitted:
            raise RuntimeError(
                "Rasterized design exceeds the upper solid-volume limit: "
                f"rasterized={rasterized_fraction:.6f}, "
                f"permitted={permitted:.6f}."
            )

        compliance = True_Objective(xPhys, nelx, nely, Emin, E0, penal)
        
        # To mitigate NaNs
        if not np.isfinite(compliance):
            compliance = 1e30
        
        if Print:
            print(f"Iteration {iters+1}: COMPLIANCE = {compliance}")
        compliances.append(float(compliance))
        rasterized_fractions.append(rasterized_fraction)

    compliance_tensor = torch.tensor(compliances, **tkwargs).unsqueeze(-1)
    if return_solid_fractions:
        return (compliance_tensor,torch.tensor(rasterized_fractions, **tkwargs),)
    return compliance_tensor
