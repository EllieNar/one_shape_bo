"""
See 01/07 Topology Optimisation ChatGPT

Python translation of Andreassen et al. (2011) top88 MATLAB code.

Base problem: minimum-compliance topology optimisation of a 2D linear-elastic
MBB/cantilever-like domain with a volume-fraction constraint.

Requires: numpy, scipy, matplotlib.

Marked additions beyond the original 88-line code are labelled:
    # ADDED: ...
These additions expose options discussed in Section 5 of the paper:
other boundary conditions, multiple load cases, passive elements, and a
Heaviside projection filter. Some convenience options for circular/elliptic/
rectangular passive holes are also added for perforation studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import binary_dilation
from scipy import interpolate
from skimage import measure
from skimage.draw import polygon

import matplotlib.pyplot as plt



FilterType = Literal[1, 2, 3]  # 1=sensitivity, 2=density, 3=Heaviside projection
BoundaryType = Literal["mbb", "cantilever"]

def get_closed_contours(contours, n_sides):
    xis = []
    yis = []
    
    for contour in contours:
        x_contour = contour[:, 1]
        y_contour = contour[:, 0]
        
        if x_contour[0] == x_contour[-1] and y_contour[0] == y_contour[-1]:
            tck, u = interpolate.splprep([x_contour, y_contour], s=0, per=True)
            xi, yi = interpolate.splev(np.linspace(0, 1, n_sides + 1), tck)
            xis.append(xi)
            yis.append(yi)
        
    return xis, yis

def spline_to_mask(xs, ys, shape):
    # fill polygon
    rr, cc = polygon(ys, xs, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask
    
def get_row_col(nelx, nely, nodes, dof):
    
    if nodes == "left":
        n = 1
        m = 0
    elif nodes == "right":
        n = nelx + 1
        m = 0
    elif nodes == "bottom":
        n = 0
        m = 1
    elif nodes == "top":
        n = 0
        m = nely + 1
    else:
        n = nodes[0]
        m = nodes[1]
        
    if dof == 0: # Apply at horizontal
        if n != 0:
            return np.arange(2*(n-1)*(nely+1), (2 + 2*(n-1))*(nely + 1), 2)
        elif m != 0:
            return 2*np.arange(m-1, (nelx + 1)*(nely + 1), nely + 1)
        
    elif dof == 1: # Apply at vertical
        if n != 0:
            return np.arange(2*(n-1)*(nely + 1) + 1, (2 + 2*(n-1))*(nely + 1) + 1, 2)
        elif m != 0:
            return 2*np.arange(m-1, (nelx + 1)*(nely + 1), nely + 1) + 1
        
            
@dataclass
class LoadCase:
    """A nodal force: node indices are zero-based; dof is 0 for x, 1 for y."""
    node: int
    dof: int
    value: float
# e.g. lc = LoadCase(node = 20, dof = 1, value = -1000)

@dataclass
class FixedDOF:
    node: int
    dof: int
# e.g. fd = FixedDOF(node = 2, dof = 0)

@dataclass
class PassiveHole:
    """Passive void/solid region in element coordinates, not physical units."""
    kind: Literal["circle", "ellipse", "rectangle"]
    centre_x: float
    centre_y: float
    radius: Optional[float] = None
    radius_x: Optional[float] = None
    radius_y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    value: int = 1  # 1=void, 2=solid

# THIS COMPUTES HOW STIFF AN ELEMENT IS
def lk() -> np.ndarray:
    """Element stiffness matrix for a unit Young's modulus square Q4 element."""
    # This integral is determined through Gauss quadrature (see 260701_Gaussian_Quadrature_2D/py)
    nu = 0.3
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    return (np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]])) / (1 - nu**2) / 24


# order F below forces fulling elements column by column, rather than row by row which is numpy convention
# THIS TELLS THE SOLVER WHERE THAT ELEMENT CONNECTS TO THE REST OF THE STRUCTURE
def element_dofs(nelx: int, nely: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return element degree-of-freedom matrix and node numbering."""
    # nodenrs creates the node numbering
    nodenrs = np.arange((nelx + 1) * (nely + 1)).reshape((nely + 1, nelx + 1), order="F")
    # edof_vec finds the first DOF of each element. Every node has 2 DOFs
    # For each element, edof_vec stores the first DOF of its lower-left node
    edof_vec = 2 * nodenrs[:-1, :-1].reshape(nelx * nely, order="F") + 2
    # offsets: encode the geometry of a Q4 element.
    offsets = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    # edof_mat: now each row contains the 8 dof for a given element 
    # This tells the program exactly where each 8 x 8 element stiffness matrix belongs
    edof_mat = edof_vec[:, None] + offsets[None, :]
    return edof_mat.astype(int), nodenrs

# BUILDS THE TOPOLOGY OPTIMISATION FILTER MATRIX H
# This averages nearby element values so that the design does not become noisy or checkerboard

# nelx and nely are the number of elements in the x- and y- directions respectively
# go through each element e1 and search its neighbouring elements e2 within a distance of rmin
def make_filter(nelx: int, nely: int, rmin: float) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Classical sparse density/sensitivity filter from Section 2.3."""
    rows, cols, vals = [], [], []    
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            # np.ceil rounds rmin up to the next integer and sets how far the code needs to search for neighbouring elements
            imin, imax = max(i1 - int(np.ceil(rmin)) + 1, 0), min(i1 + int(np.ceil(rmin)), nelx)
            jmin, jmax = max(j1 - int(np.ceil(rmin)) + 1, 0), min(j1 + int(np.ceil(rmin)), nely)
            for i2 in range(imin, imax):
                for j2 in range(jmin, jmax):
                    e2 = i2 * nely + j2
                    # This weight ensures that closer neighbours get more influence.
                    weight = max(0.0, rmin - np.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2))
                    if weight > 0:
                        rows.append(e1)
                        cols.append(e2)
                        vals.append(weight)
    # sp.coo builds a sparse matrix from rows (row indices), cols (column indices) and vals (nonzero values)
    # coo_matrix is convenient for constructing a matrix when you already know the nonzero entries
    # The sparse matrix is converted to CSR format, which is efficient for later multiplication
    H = sp.coo_matrix((vals, (rows, cols)), shape=(nelx * nely, nelx * nely)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def build_passive_mask(nelx: int, nely: int, holes: Optional[Sequence[PassiveHole]]) -> np.ndarray:
    """ADDED: create passive void/solid regions, including circular holes."""
    passive = np.zeros((nely, nelx), dtype=int)
    if not holes:
        return passive
    xx, yy = np.meshgrid(np.arange(nelx) + 0.5, np.arange(nely) + 0.5)
    for h in holes:
        if h.kind == "circle":
            if h.radius is None:
                raise ValueError("Circle hole requires radius")
            mask = (xx - h.centre_x) ** 2 + (yy - h.centre_y) ** 2 <= h.radius**2
        elif h.kind == "ellipse":
            if h.radius_x is None or h.radius_y is None:
                raise ValueError("Ellipse hole requires radius_x and radius_y")
            mask = ((xx - h.centre_x) / h.radius_x) ** 2 + ((yy - h.centre_y) / h.radius_y) ** 2 <= 1.0
        elif h.kind == "rectangle":
            if h.width is None or h.height is None:
                raise ValueError("Rectangle hole requires width and height")
            mask = (np.abs(xx - h.centre_x) <= h.width / 2) & (np.abs(yy - h.centre_y) <= h.height / 2)
        else:
            raise ValueError(f"Unknown passive kind: {h.kind}")
        passive[mask] = h.value
    return passive


def default_loads_and_supports(nelx: int, nely: int, boundary: BoundaryType):
    """Base boundary choices matching common variants in the paper."""
    # Each node has 2 dof: horizontal and vertical displacement
    # There are (nelx + 1)(nely + 1) nodes, so the total dof is...
    ndof = 2 * (nelx + 1) * (nely + 1)
    if boundary == "mbb":
        # Half MBB beam: downward load at upper-left node; symmetry/roller supports.
        # this means that node index 0 with DOF 1 (vertical displacement) under a force of -1 downwards
        # so a downward point force at upper left node.
        loads = [(nely, 1, -1.0)]
        # This fixes every other dof along the left edge (0, 2, 4...) these are the hoizontal displacements of left-edge nodes
        # In addition to the very last DOF in the model (the bottom right) thus behaving as a roller/symmetry-type support.
        #fixed = np.union1d(np.arange(0, 2 * (nely + 1), 2), np.array([ndof - 1]))
        
        # Middway along half beam, so a quarter along, set n = nelx/2 (this is the x index) and this is applied to every node in the y-direction
        #fixed = np.union1d(get_row_col(nelx, nely, (nelx/2, 0), 0), np.array([ndof - 1]))
        
        # At start of the beam
        fixed = np.union1d(get_row_col(nelx, nely, "left", 0), np.array([ndof - 1]))
        
        # so the left edge cannot move horizontally, and an extra DOF is fixed to prevent rigid-body motion
    elif boundary == "cantilever":
        # ADDED: Section 5.1-style alternative boundary condition.
        # The last node
        right_top_node = (nelx + 1) * (nely + 1) - 1
        
        # Middle right node
        mid_right_node = int((nelx + 1)*(nely) + (nely + 1)/2 - 1)
        
        # downward force at top right node
        loads = [(mid_right_node, 1, -1.0)]
        # all DOF are fixed on the left hand edge, so no movement in x or y
        fixed = np.arange(0, 2 * (nely + 1))
    else:
        raise ValueError(f"Unknown boundary: {boundary}")
        # The setup is returned i.e. the loads (a list of force locations and magnitudes) and fixed (an array of constrained DOF indices)
    return loads, fixed.astype(int)


def optimise_topology(
    nelx: int,
    nely: int,
    volfrac: float,
    penal: float = 3.0,
    rmin: float = 1.5,
    ft: FilterType = 2,
    boundary: BoundaryType = "mbb",
    
    # Example optional sequence for load_cases (simply a list of LoadCase objects)
    # load_cases = [
    #               LoadCase(node=20, dof=1, value=-1000),
    #               LoadCase(node=80, dof=0, value=500)
    #               ]
    load_cases: Optional[Sequence[LoadCase]] = None,  # ADDED: multiple/custom load cases
    
    # Example optional iterable for fixed_dof
    # fixed_dof = [
    #              FixedDOF(node=20, dof=0),
    #              FixedDOF(node=20, dof=1),
    #              FixedDOF(node=40, dof=0),
    #              FixedDOF(node=40, dof=1),]
    fixed_dofs: Optional[Iterable[int]] = None,        # ADDED: custom supports
    
    # Example optional passive holes
    # passive_holes = [
    #                  PassiveHole(kind="ellipse", centre_x=20, centre_y=10, radius_x=5, radius_y=3, value=1)
    #                  ]
    passive_holes: Optional[Sequence[PassiveHole]] = None,  # ADDED: fixed holes/solids
    
    max_iter: int = 50,
    tol: float = 0.01,
    plot: bool = False,
):
    """Run minimum-compliance topology optimisation.

    Returns a dictionary containing x, xPhys, compliance history, and volume history.
    """
    E0, Emin = 1.0, 1e-9
    KE = lk()
    edof_mat, _ = element_dofs(nelx, nely)
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Sparse assembly indices.
    #np.ravel returns a contiguous flattened array. Could use reshape?
    #np.kron computes the Kronecker product of two matrices
    iK = np.kron(edof_mat, np.ones((8, 1), dtype=int)).ravel()
    jK = np.kron(edof_mat, np.ones((1, 8), dtype=int)).ravel()

    # Loads/supports.
    default_loads, default_fixed = default_loads_and_supports(nelx, nely, boundary)
    if load_cases is None:
        load_cases = [LoadCase(*lc) for lc in default_loads]
    if fixed_dofs is None:
        fixed_dofs = default_fixed
    fixed = np.array(list(fixed_dofs), dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed)

    # ADDED: multiple load cases can be supplied as several LoadCase objects.
    # This creates a global force matrix where each row is the total degrees of freedom and each column is one loading scenario
    F = np.zeros((ndof, len(load_cases)))
    for k, lc in enumerate(load_cases):
        # k is the column number and lc is the corresponding LoadCase
        F[2 * lc.node + lc.dof, k] = lc.value
        
    # Constructs the filtering matrices:
        # H is the sparse matrix containing the filter weights between neighbouring elements
        # Hs is the row sums of H used to normalise the weighted average
    H, Hs = make_filter(nelx, nely, rmin)
    
    # Constructs the passive mask
    # Creates a matrix the same size as the element mesh:
        # 0 =  active element (optimiser may change it)
        # 1 = passive void
        # 2 = passive solid
    passive = build_passive_mask(nelx, nely, passive_holes)

    # Initialise the design, so that every element initially has the same density equal to volfrac
    # volfrac is the prescribed volume fraction, e.g. 0.5
    x = volfrac * np.ones((nely, nelx))
    # then checks the passive matrix and resets the element values according to whether they must be solid or void
    x[passive == 1] = 0.0
    x[passive == 2] = 1.0

    # ADDED: Section 5.4 Heaviside continuation parameter.
    # Initialise optimisaiton variables before iteration loop begins
    # The Heaviside filter is a modification of the original density filter with a heaviside step function which projects the density to a physical density
    # The paper gradually doubles beta from 1 to 512 either:
        # (a) Every 50 iterations, or
        # (b) When the change in terms of design variables between two consecutive designs becomes less than 0.01 (i.e. nearly converged)
    # hence loop_beta keeps count of how many optimisation iterations have occured since beta was last increased
    beta = 1.0
    loop_beta = 0
    # x are the design variables
    # x_tilde are the filtered (intermediate) densities
    # x_phys are the physical densities used in FE analysis
    x_tilde = x.copy()
    # checks whether user selected heaviside filter:
        # ft = 1 --> sensitivity filter
        # ft = 2 --> density filter
        # ft = 3 --> heaviside projection (pushes densities closer to void or solid and reducing grey regions)
    
    # AN IMPORTANT NOTE!
        # If sensitivity filter is chosen, then the FE analysis is performed on the unfiltered field, but checkerboards are prevented because the sensitivities are filtered before the densities are updated
        # If the density filter is chosen, although the first FE analysis is performed on the unfiltered field, it will use the filtered physical density in the next iteration
    
    if ft == 3:
        xPhys = 1 - np.exp(-beta * x_tilde) + x_tilde * np.exp(-beta)
    else:
        xPhys = x.copy()
    
    # history_c --> compliance at each iteration and history_vol --> volume fraction at each iteration
    # useful for plotting convergence
    history_c, history_vol = [], []
    # This is the move limit m (eq 3 of paper), it limits how much an element density can change in one iteration.
    # This prevents unstable, overly aggressive updates, and helps optimisation converges smoothly
    move = 0.2

    for loop in range(1, max_iter + 1):
        loop_beta += 1

        # FE analysis.
        # This turns densities into stiffness using SIMP (dense material is stiff and a void is almost zero stiffness)
        young = Emin + xPhys.ravel(order="F") ** penal * (E0 - Emin)
        sK = (KE.ravel(order="F")[:, None] * young[None, :]).ravel(order="F")
        # Assembles the sparse global stiffness matrix
        K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = (K + K.T) / 2
        # Equate to load cases matrix F
        U = (np.zeros_like(F)).squeeze()
        U[free] = spla.spsolve(K[free[:, None], free], F[free, :])
        if U.ndim == 1:
            U = U[:, None]

        # Objective and sensitivities. For multiple loads, sum compliances.
        # Compliance is a stiffness measure
        # Code is asking 'which elements matter the most for stiffness?'
        c = 0.0
        dc = np.zeros((nely, nelx))
        for k in range(F.shape[1]):
            Ue = U[edof_mat, k]
            ce = np.sum((Ue @ KE) * Ue, axis=1).reshape((nely, nelx), order="F")
            c += np.sum((Emin + xPhys**penal * (E0 - Emin)) * ce)
            dc += -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
        dv = np.ones((nely, nelx))

        # Filtering/modification of sensitivities.
        if ft == 1: # sensitivity filtering
            dc_vec = H @ (x.ravel(order="F") * dc.ravel(order="F"))
            dc = (dc_vec / Hs / np.maximum(1e-3, x.ravel(order="F"))).reshape((nely, nelx), order="F")
        elif ft == 2: # density filtering
            dc = ((H @ (dc.ravel(order="F") / Hs))).reshape((nely, nelx), order="F")
            dv = ((H @ (dv.ravel(order="F") / Hs))).reshape((nely, nelx), order="F")
        elif ft == 3: # heavyside projection filtering
            # ADDED: Heaviside projection filter sensitivity chain rule.
            dx = beta * np.exp(-beta * x_tilde) + np.exp(-beta)
            dc = (H @ ((dc * dx).ravel(order="F") / Hs)).reshape((nely, nelx), order="F")
            dv = (H @ ((dv * dx).ravel(order="F") / Hs)).reshape((nely, nelx), order="F")
        else:
            raise ValueError("ft must be 1, 2, or 3")

        # Optimality-criteria update using bisection on the Lagrange multiplier.
        l1, l2 = 0.0, 1e9
        xold = x.copy()
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            # Update the design, this is the optimality-criteria step. It increases material where compliance would drop a lot, and removes it where it is less useful while respecting the move limit, that 0<= x <=1 and the volume fraction
            candidate = x * np.sqrt(np.maximum(0.0, -dc / np.maximum(1e-30, dv) / lmid))
            xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, candidate))))

            # ADDED: enforce passive regions after each design update.
            xnew[passive == 1] = 0.0
            xnew[passive == 2] = 1.0

            if ft == 1:
                xPhys_trial = xnew
            elif ft == 2:
                # This is the filtered version of xnew
                xPhys_trial = ((H @ xnew.ravel(order="F")) / Hs).reshape((nely, nelx), order="F")
            else:
                # for the heavyside projection, it is filtered and then projected to black/white
                x_tilde_trial = ((H @ xnew.ravel(order="F")) / Hs).reshape((nely, nelx), order="F")
                xPhys_trial = 1 - np.exp(-beta * x_tilde_trial) + x_tilde_trial * np.exp(-beta)
            
            # Enforce passive regions which do not allow optimiser to change
            xPhys_trial[passive == 1] = 0.0
            xPhys_trial[passive == 2] = 1.0
            
            # Adjust the volume constraint to keep the total amount of material near volfrac
            if xPhys_trial.sum() > volfrac * nelx * nely:
                l1 = lmid
            else:
                l2 = lmid
                
        # store the new state
        x = xnew
        xPhys = xPhys_trial
        if ft == 3:
            x_tilde = x_tilde_trial

        change = np.max(np.abs(x - xold))
        history_c.append(c)
        history_vol.append(xPhys.mean())
        print(f"It.: {loop:4d} Obj.: {c:10.4f} Vol.: {xPhys.mean():6.3f} ch.: {change:6.3f}")

        # ADDED: continuation scheme for black-and-white Heaviside filter.
        if ft == 3 and beta < 512 and (loop_beta >= 50 or change <= tol):
            beta *= 2
            loop_beta = 0
            change = 1.0
            print(f"Heaviside beta increased to {beta:g}")

        if plot:
            
            plt.clf()
            plt.imshow(1 - xPhys, cmap="gray", origin="lower")
            plt.axis("equal")
            plt.axis("off")
            plt.pause(0.01)

        if change <= tol and not (ft == 3 and beta < 512):
            break

    return {"x": x, "xPhys": xPhys, "compliance": np.array(history_c), "volume": np.array(history_vol)}, xPhys


if __name__ == "__main__":
    # Example matching the spirit of the paper: MBB beam, density filter.
    NX = 100
    NY = 50
    
    #Hole = [PassiveHole(kind="ellipse", centre_x=NX/2, centre_y=NY/3, radius_x=20, radius_y=8, value=1)]
    #result = optimise_topology(nelx=NX, nely=NY, volfrac=0.5, penal=3, rmin=5, ft=2, passive_holes=Hole, plot=True)
    
    result, xPhys = optimise_topology(nelx=NX, nely=NY, volfrac=0.5, penal=3, rmin=6, ft=2, plot=True)
    
    # Ensure selection of outer contours
    min_out_border = 1
    xx, yy = np.meshgrid(np.arange(NX) + 0.5, np.arange(NY) + 0.5)
    outline = ((np.abs(xx - NX/2) < (NX/2 - min_out_border)) & (np.abs(yy - NY/2) < (NY/2 - min_out_border)))
    xPhys[outline == False] = 1
    
    # Find contours at a constant value of 0.8
    contours = measure.find_contours(xPhys, 0.5)
    
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    plt.imshow(1 - xPhys, cmap="gray", origin="lower")
    plt.axis("equal")
    ax.axis('image')
    
    xis, yis = get_closed_contours(contours, 3000)
    new_grid = np.ones((NY, NX))
    masks = []
    print(new_grid.shape)
    for xi, yi in zip(xis, yis):
        ax.plot(xi, yi, '-b')
        masks.append(spline_to_mask(xi, yi, new_grid.shape))
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    for m in masks:
        new_grid[m] = 0
    
    fig, ax = plt.subplots()
    plt.clf()
    plt.imshow(1 - new_grid, cmap="gray", origin="lower")
    plt.axis("equal")
    plt.axis("off")
    plt.pause(0.01)
    
# MONDAY
# Get contour for the outer parts i.e. unclosed too
# Option to fit e.g. square/ellipse in space.
# Bayesian loop to refine e.g. multiple perforations of regular shape out of single void?
    
    
    
