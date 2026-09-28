"""Read-only audit of the qualitative topology reference image."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage

from .physical import true_objective


@dataclass(frozen=True)
class BenchmarkAudit:
    image: str
    source_bbox_xyxy: tuple[int, int, int, int]
    grid_shape: tuple[int, int]
    solid_fraction: float
    permitted_solid_fraction: float
    volume_feasible: bool
    void_components: int
    recomputed_binary_compliance: float
    logged_continuous_compliance: float | None
    comparison_note: str


def _extract_domain(image: np.ndarray):
    rgb = image[..., :3]
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    dark = luminance < 0.5
    rows, columns = np.nonzero(dark)
    if not len(rows):
        raise ValueError("Reference image contains no dark domain pixels.")
    x0, x1 = int(columns.min()), int(columns.max())
    y0, y1 = int(rows.min()), int(rows.max())
    return luminance[y0 : y1 + 1, x0 : x1 + 1], (x0, y0, x1, y1)


def _resample_binary(domain: np.ndarray, nelx: int, nely: int):
    row_edges = np.linspace(0, domain.shape[0], nely + 1)
    column_edges = np.linspace(0, domain.shape[1], nelx + 1)
    solid = np.empty((nely, nelx), dtype=float)
    for row in range(nely):
        r0, r1 = int(np.floor(row_edges[row])), int(np.ceil(row_edges[row + 1]))
        for column in range(nelx):
            c0 = int(np.floor(column_edges[column]))
            c1 = int(np.ceil(column_edges[column + 1]))
            solid[row, column] = float(domain[r0:r1, c0:c1].mean() < 0.5)
    # The reference is rendered with ``origin="lower"``: the first FE row is
    # at the bottom of the PNG, whereas image arrays are indexed from the top.
    return np.flipud(solid)


def audit_reference_image(
    image_path,
    *,
    nelx: int = 100,
    nely: int = 50,
    solid_max: float = 0.5,
    raster_tolerance: float = 0.005,
    emin: float = 1e-9,
    e0: float = 1.0,
    penal: float = 3.0,
    logged_continuous_compliance: float | None = 96.8918,
    report_path=None,
) -> BenchmarkAudit:
    """Audit a stored PNG without executing or changing the reference script."""
    path = Path(image_path)
    domain, bbox = _extract_domain(mpimg.imread(path))
    x_phys = _resample_binary(domain, nelx, nely)
    solid_fraction = float(x_phys.mean())
    _, void_components = ndimage.label(1.0 - x_phys)
    compliance = true_objective(x_phys, nelx, nely, emin, e0, penal)
    permitted = solid_max + raster_tolerance
    report = BenchmarkAudit(
        image=str(path),
        source_bbox_xyxy=bbox,
        grid_shape=(nely, nelx),
        solid_fraction=solid_fraction,
        permitted_solid_fraction=permitted,
        volume_feasible=solid_fraction <= permitted,
        void_components=int(void_components),
        recomputed_binary_compliance=compliance,
        logged_continuous_compliance=logged_continuous_compliance,
        comparison_note=(
            "The PNG is a thresholded rendering of a filtered continuous-density "
            "topology. Its reconstructed binary volume and compliance are not the "
            "continuous optimizer's logged volume and objective, so it is a "
            "qualitative topology target rather than a numerical BO target."
        ),
    )
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report
