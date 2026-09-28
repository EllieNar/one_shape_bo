"""Shape encoding, canonicalisation, and separation calculations.

The seven-value per-hole encoding and the physical calculations here match the
legacy implementation.  Keeping them in the package avoids making production
code depend on the historical ``legacy`` import layout.
"""

from __future__ import annotations

import torch

from .runtime import TENSOR_KWARGS


def read_shape(X, nholes: int, xsize: float, ysize: float):
    triangles, ellipses = [], []
    for i in range(0, 7 * nholes, 7):
        if X[i] == 0:
            triangles.append(
                torch.stack(
                    [
                        xsize * X[i + 1],
                        ysize * X[i + 2],
                        xsize * (2 * X[i + 3] - 1),
                        ysize * (2 * X[i + 4] - 1),
                        xsize * (2 * X[i + 5] - 1),
                        ysize * (2 * X[i + 6] - 1),
                    ]
                )
            )
        elif X[i] == 1:
            ellipses.append(
                torch.stack(
                    [
                        xsize * X[i + 1],
                        ysize * X[i + 2],
                        xsize * X[i + 3],
                        ysize * X[i + 4],
                        torch.pi * X[i + 5],
                    ]
                )
            )

    triangle_tensor = (
        torch.stack(triangles).reshape(-1, 3, 2)
        if triangles
        else torch.empty((0, 3, 2), **TENSOR_KWARGS)
    )
    ellipse_tensor = (
        torch.stack(ellipses)
        if ellipses
        else torch.empty((0, 5), **TENSOR_KWARGS)
    )
    return triangle_tensor, ellipse_tensor


def triangle(X, i: int, xsize: float, ysize: float):
    return torch.stack(
        [
            xsize * X[i + 1],
            ysize * X[i + 2],
            xsize * (2 * X[i + 3] - 1),
            ysize * (2 * X[i + 4] - 1),
            xsize * (2 * X[i + 5] - 1),
            ysize * (2 * X[i + 6] - 1),
        ]
    )


def ellipse(X, i: int, xsize: float, ysize: float):
    return torch.stack(
        [
            xsize * X[i + 1],
            ysize * X[i + 2],
            xsize * X[i + 3],
            ysize * X[i + 4],
            torch.pi * X[i + 5],
        ]
    )


def area_triangle(triangles):
    return [
        0.5 * torch.abs(t[1, 0] * t[2, 1] - t[1, 1] * t[2, 0])
        for t in triangles
    ]


def area_ellipse(ellipses):
    return [torch.pi * e[2] * e[3] for e in ellipses]


def hole_areas(X, nholes: int, xsize: float, ysize: float) -> list[float]:
    """Return physical areas in encoded hole order."""
    areas = []
    for i in range(nholes):
        j = 7 * i
        if X[j] < 0.5:
            decoded = triangle(X, j, xsize, ysize)
            value = 0.5 * torch.abs(
                decoded[2] * decoded[5] - decoded[3] * decoded[4]
            )
        else:
            decoded = ellipse(X, j, xsize, ysize)
            value = torch.pi * decoded[2] * decoded[3]
        areas.append(float(value.detach().item()))
    return areas


def canonicalize_triangles(nholes: int, vertices):
    if nholes == 0:
        return torch.empty(0, **TENSOR_KWARGS)

    canonical = []
    for tri in vertices:
        start = min(
            range(3),
            key=lambda k: (float(tri[k, 0].item()), float(tri[k, 1].item())),
        )
        other = [k for k in range(3) if k != start]
        p0, p1, p2 = tri[start], tri[other[0]], tri[other[1]]
        cross = ((p1[0] - p0[0]) * (p2[1] - p0[1])) - (
            (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        if cross < 0:
            p1, p2 = p2, p1
        canonical.append(torch.stack((p0, p1, p2)))

    vertices = torch.stack(canonical)
    centroids = vertices.mean(dim=1)
    order = sorted(
        range(nholes),
        key=lambda k: (
            float(centroids[k, 0].item()),
            float(centroids[k, 1].item()),
        ),
    )
    vertices = vertices[order]
    p0 = vertices[:, 0, :]
    edge_1 = vertices[:, 1, :] - p0
    edge_2 = vertices[:, 2, :] - p0
    encoded = torch.cat(
        (p0, 0.5 * (edge_1 + 1.0), 0.5 * (edge_2 + 1.0)), dim=-1
    ).reshape(-1)
    shape_index = encoded.new_tensor([0.0])
    return torch.cat(
        [torch.cat((shape_index, chunk)) for chunk in torch.split(encoded, 6)]
    )


def canonicalize_ellipses(nholes: int, points):
    if nholes == 0:
        return torch.empty(0, **TENSOR_KWARGS)
    order = sorted(
        range(nholes),
        key=lambda k: (float(points[k, 0].item()), float(points[k, 1].item())),
    )
    points = points[order]
    shape_index = points.new_tensor([1.0])
    dummy = points.new_tensor([0.0])
    return torch.cat(
        [torch.cat((shape_index, point, dummy)) for point in points]
    )


def anchor_point(points, maximum: bool, constraints, normalise: bool):
    xmin, xmax, ymin, ymax = constraints.box
    if normalise:
        domain_min = points.new_tensor(
            [constraints.bx / constraints.xsize, constraints.by / constraints.ysize]
        )
        domain_max = points.new_tensor(
            [
                1.0 - constraints.bx / constraints.xsize,
                1.0 - constraints.by / constraints.ysize,
            ]
        )
    else:
        domain_min = points.new_tensor([xmin, ymin])
        domain_max = points.new_tensor([xmax, ymax])

    if points[0] == 0:
        vertices = points[1:].reshape(3, 2)
        return (
            domain_max - vertices.max(dim=0).values
            if maximum
            else domain_min - vertices.min(dim=0).values
        )

    cx, cy, rx, ry, theta = points[1:-1]
    if normalise:
        theta = torch.pi * theta
    dx = torch.sqrt((rx * torch.cos(theta)) ** 2 + (ry * torch.sin(theta)) ** 2)
    dy = torch.sqrt((rx * torch.sin(theta)) ** 2 + (ry * torch.cos(theta)) ** 2)
    lower = torch.stack((cx - dx, cy - dy))
    upper = torch.stack((cx + dx, cy + dy))
    return domain_max - upper if maximum else domain_min - lower


def smooth_min(x, tau: float = 1e-4, dim: int = 0):
    return -tau * torch.logsumexp(-x / tau, dim=dim)


def smooth_max_upper(x, tau: float = 1e-4, dim: int = 0):
    return tau * torch.logsumexp(x / tau, dim=dim)


def smooth_max_lower(x, tau: float = 1e-4, dim: int = 0):
    count = x.new_tensor(x.shape[dim])
    return tau * torch.logsumexp(x / tau, dim=dim) - tau * torch.log(count)


def separating_axis_theorem(
    points_i, points_j, xsize: float, ysize: float, dir_trials: int = 100,
    eps: float = 1e-12,
):
    """Return a conservative physical clearance between two encoded holes."""
    points = torch.stack((points_i, points_j), dim=0)
    _, indices = torch.sort(points[:, 0])
    points_i, points_j = points[indices][0], points[indices][1]

    if points_i[0] == 0 and points_j[0] == 0:
        vertices = []
        for encoded in (points_i, points_j):
            decoded = triangle(encoded, 0, xsize, ysize)
            p0 = decoded[0:2]
            vertices.append(torch.stack((p0, p0 + decoded[2:4], p0 + decoded[4:6])))
        tri_i, tri_j = vertices
        edges = torch.cat(
            (tri_i.roll(-1, dims=0) - tri_i, tri_j.roll(-1, dims=0) - tri_j)
        )
        axes = torch.stack((-edges[:, 1], edges[:, 0]), dim=-1)
        axes = axes / torch.sqrt((axes * axes).sum(dim=-1, keepdim=True) + eps**2)
        projection_i, projection_j = tri_i @ axes.T, tri_j @ axes.T
        gaps = torch.stack(
            (
                smooth_min(projection_j, dim=0)
                - smooth_max_upper(projection_i, dim=0),
                smooth_min(projection_i, dim=0)
                - smooth_max_upper(projection_j, dim=0),
            )
        )
        return smooth_max_lower(smooth_max_lower(gaps, dim=0), dim=0)

    if points_i[0] == 0 and points_j[0] == 1:
        decoded = triangle(points_i, 0, xsize, ysize)
        p0 = decoded[0:2]
        vertices = torch.stack((p0, p0 + decoded[2:4], p0 + decoded[4:6]))
        edges = vertices.roll(-1, dims=0) - vertices
        cx, cy, rx, ry, theta = ellipse(points_j, 0, xsize, ysize)
        centre = torch.stack((cx, cy))
        parameter = (
            ((centre - vertices) * edges).sum(dim=1)
            / (edges * edges).sum(dim=1).clamp_min(eps)
        ).clamp(0.0, 1.0)
        projected = vertices + parameter[:, None] * edges
        closest = projected[((projected - centre) ** 2).sum(dim=1).argmin()]
        fourth = closest - centre
        fourth = fourth / torch.sqrt((fourth * fourth).sum() + eps**2)
        axes = torch.stack((-edges[:, 1], edges[:, 0]), dim=-1)
        axes = axes / torch.sqrt((axes * axes).sum(dim=-1, keepdim=True) + eps**2)
        axes = torch.cat((axes, fourth.unsqueeze(0)), dim=0)
        projection = vertices @ axes.T
        tri_min = smooth_min(projection, dim=0)
        tri_max = smooth_max_upper(projection, dim=0)
        u = torch.stack((torch.cos(theta), torch.sin(theta)))
        v = torch.stack((-torch.sin(theta), torch.cos(theta)))
        support = torch.sqrt((axes @ u) ** 2 * rx**2 + (axes @ v) ** 2 * ry**2 + eps**2)
        centre_projection = centre @ axes.T
        gaps = torch.stack(
            (centre_projection - support - tri_max, tri_min - centre_projection - support)
        )
        return smooth_max_lower(smooth_max_lower(gaps, dim=0), dim=0)

    cx1, cy1, rx1, ry1, th1 = ellipse(points_i, 0, xsize, ysize)
    cx2, cy2, rx2, ry2, th2 = ellipse(points_j, 0, xsize, ysize)
    delta = torch.stack((cx2 - cx1, cy2 - cy1))
    phi = torch.arange(
        dir_trials, device=points_i.device, dtype=points_i.dtype
    ) * torch.pi / dir_trials
    axes = torch.stack((torch.cos(phi), torch.sin(phi)), dim=1)
    centre_gap = torch.abs(axes @ delta)
    u1 = torch.stack((torch.cos(th1), torch.sin(th1)))
    v1 = torch.stack((-torch.sin(th1), torch.cos(th1)))
    u2 = torch.stack((torch.cos(th2), torch.sin(th2)))
    v2 = torch.stack((-torch.sin(th2), torch.cos(th2)))
    radius_1 = torch.sqrt(rx1**2 * (axes @ u1) ** 2 + ry1**2 * (axes @ v1) ** 2 + eps**2)
    radius_2 = torch.sqrt(rx2**2 * (axes @ u2) ** 2 + ry2**2 * (axes @ v2) ** 2 + eps**2)
    return (centre_gap - radius_1 - radius_2).max()


def separating_axis_theorem_batch(
    points_i, points_j, xsize: float, ysize: float, dir_trials: int = 100,
    eps: float = 1e-12,
):
    """Vectorised clearances from encoded ``points_i`` to one encoded hole."""
    if points_i.ndim != 2 or points_i.shape[-1] != 7:
        raise ValueError("points_i must have shape (batch, 7).")
    points_j = points_j.reshape(1, 7).expand(points_i.shape[0], -1)
    shape_i, shape_j = int(points_i[0, 0].item()), int(points_j[0, 0].item())
    if shape_i > shape_j:
        points_i, points_j = points_j, points_i
        shape_i, shape_j = shape_j, shape_i
    scale = points_i.new_tensor([xsize, ysize])

    def triangle_vertices(points):
        p0 = points[:, 1:3] * scale
        v1 = (2.0 * points[:, 3:5] - 1.0) * scale
        v2 = (2.0 * points[:, 5:7] - 1.0) * scale
        return torch.stack((p0, p0 + v1, p0 + v2), dim=1)

    def ellipse_values(points):
        return points[:, 1:3] * scale, points[:, 3:5] * scale, torch.pi * points[:, 5]

    if shape_i == 0 and shape_j == 0:
        vertices_i, vertices_j = triangle_vertices(points_i), triangle_vertices(points_j)
        edges = torch.cat(
            (
                vertices_i.roll(-1, dims=1) - vertices_i,
                vertices_j.roll(-1, dims=1) - vertices_j,
            ),
            dim=1,
        )
        axes = torch.stack((-edges[:, :, 1], edges[:, :, 0]), dim=-1)
        axes = axes / torch.sqrt((axes * axes).sum(dim=-1, keepdim=True) + eps**2)
        projection_i = torch.einsum("nvc,nac->nva", vertices_i, axes)
        projection_j = torch.einsum("nvc,nac->nva", vertices_j, axes)
        separation = torch.maximum(
            projection_j.min(dim=1).values - projection_i.max(dim=1).values,
            projection_i.min(dim=1).values - projection_j.max(dim=1).values,
        )
        return separation.max(dim=1).values

    if shape_i == 0 and shape_j == 1:
        vertices = triangle_vertices(points_i)
        edges = vertices.roll(-1, dims=1) - vertices
        centre, radii, theta = ellipse_values(points_j)
        parameter = (
            ((centre[:, None, :] - vertices) * edges).sum(dim=2)
            / (edges * edges).sum(dim=2).clamp_min(eps)
        ).clamp(0.0, 1.0)
        projected = vertices + parameter[:, :, None] * edges
        closest_index = ((projected - centre[:, None, :]) ** 2).sum(dim=2).argmin(dim=1)
        closest = projected[
            torch.arange(points_i.shape[0], device=points_i.device), closest_index
        ]
        fourth = closest - centre
        fourth = fourth / torch.sqrt((fourth * fourth).sum(dim=1, keepdim=True) + eps**2)
        axes = torch.stack((-edges[:, :, 1], edges[:, :, 0]), dim=-1)
        axes = axes / torch.sqrt((axes * axes).sum(dim=-1, keepdim=True) + eps**2)
        axes = torch.cat((axes, fourth[:, None, :]), dim=1)
        projection = torch.einsum("nvc,nac->nva", vertices, axes)
        tri_min, tri_max = projection.min(dim=1).values, projection.max(dim=1).values
        u = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
        v = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=1)
        support = torch.sqrt(
            (axes * u[:, None, :]).sum(dim=2) ** 2 * radii[:, 0:1] ** 2
            + (axes * v[:, None, :]).sum(dim=2) ** 2 * radii[:, 1:2] ** 2
            + eps**2
        )
        centre_projection = (axes * centre[:, None, :]).sum(dim=2)
        return torch.maximum(
            centre_projection - support - tri_max,
            tri_min - centre_projection - support,
        ).max(dim=1).values

    centre_i, radii_i, theta_i = ellipse_values(points_i)
    centre_j, radii_j, theta_j = ellipse_values(points_j)
    delta = centre_j - centre_i
    phi = torch.arange(
        dir_trials, device=points_i.device, dtype=points_i.dtype
    ) * torch.pi / dir_trials
    axes = torch.stack((torch.cos(phi), torch.sin(phi)), dim=1)
    centre_gap = torch.abs(delta @ axes.T)
    u_i = torch.stack((torch.cos(theta_i), torch.sin(theta_i)), dim=1)
    v_i = torch.stack((-torch.sin(theta_i), torch.cos(theta_i)), dim=1)
    u_j = torch.stack((torch.cos(theta_j), torch.sin(theta_j)), dim=1)
    v_j = torch.stack((-torch.sin(theta_j), torch.cos(theta_j)), dim=1)
    support_i = torch.sqrt(
        radii_i[:, 0:1] ** 2 * (u_i @ axes.T) ** 2
        + radii_i[:, 1:2] ** 2 * (v_i @ axes.T) ** 2
        + eps**2
    )
    support_j = torch.sqrt(
        radii_j[:, 0:1] ** 2 * (u_j @ axes.T) ** 2
        + radii_j[:, 1:2] ** 2 * (v_j @ axes.T) ** 2
        + eps**2
    )
    return (centre_gap - support_i - support_j).max(dim=1).values


# Backwards-compatible names used in historical notebooks.
Read_Shape = read_shape
Area_Triangle = area_triangle
Area_Ellipse = area_ellipse
