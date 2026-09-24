# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:03:49 2026

@author: pemb6626
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {
    "device": device,
    "dtype": torch.double,
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# Takes X as a long tensor of nholes*7 and returns the triangle tensor, and then the ellipse tensor
def Read_Shape(X, nholes, xsize, ysize):
    
    #triangle bounds = [[0, xsize*(u1), ysize*(u2), xsize*(2*u3 - 1), ysize*(2*u4 - 1), xsize*(2*u5 - 1), xsize*(2*u6 - 1)]]
    #ellipse bounds  = [[1, xsize*(u1), ysize*(u2), xsize*(u3), ysize*(u4), torch.pi*u5, 0]]
    
    t, e = [], []
    for i in range(0, 7*nholes, 7):
        if X[i] == 0:
            t.append(torch.stack([xsize*X[i+1], ysize*X[i+2], xsize*(2*X[i+3] - 1), ysize*(2*X[i+4] - 1), xsize*(2*X[i+5] - 1), ysize*(2*X[i+6] - 1)]))
        elif X[i] == 1:
            e.append(torch.stack([xsize*X[i+1], ysize*X[i+2], xsize*X[i+3], ysize*X[i+4], torch.pi*X[i+5]]))
    
    if t:
        t = torch.stack(t).reshape(-1, 3, 2)
    else:
        t = torch.empty((0, 3, 2), **tkwargs)
    
    if e:
        e = torch.stack(e)
    else:
        e = torch.empty((0,5), **tkwargs)
    
    return t, e

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def triangle(X, i, xsize, ysize):
    return torch.stack([xsize*X[i+1], ysize*X[i+2], xsize*(2*X[i+3] - 1), ysize*(2*X[i+4] - 1), xsize*(2*X[i+5] - 1), ysize*(2*X[i+6] - 1)])

def ellipse(X, i, xsize, ysize):
    return torch.stack([xsize*X[i+1], ysize*X[i+2], xsize*X[i+3], ysize*X[i+4], torch.pi*X[i+5]])
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Takes a batch pf ellipses and computes its area
# Input after Read_Shape, E is the second output
# Output a list of nellipse tensor areas
def Area_Ellipse(E):
    A = []
    for ellipse in E:
        A.append(torch.pi*ellipse[2]*ellipse[3])
    return A

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Takes a batch of triangles and computes its area
# Input after Read_Shape, T is the first output
# Output a list of ntriangle tensor areas
def Area_Triangle(T):
    A = []
    for triangle in T:
        v1 = triangle[1, :]
        v2 = triangle[2, :]
        A.append(0.5*torch.abs(v1[0]*v2[1] - v1[1]*v2[0]))
    return A

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Takes a batch pf ellipses and computes its perimeter
# Input after Read_Shape, E is the second output
# Output a list of ellipse tensor perimeters
def Perimeter_Ellipse(E):
    P = []
    for ellipse in E:
        a = ellipse[2]
        b = ellipse[3]
        P.append(torch.pi*(3*(a + b) - torch.sqrt((3*a + b)*(a + 3*b))))
    return P

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Takes a batch of triangles and computes its perimeter
# Input after Read_Shape, T is the first output
# Output a list of ntriangle tensor perimeters
def Perimeter_Triangle(T):
    P = []
    for triangle in T:
        v1 = triangle[1, :]
        v2 = triangle[2, :]
        v3 = v2 - v1
        
        p = 0
        for v in [v1, v2, v3]:
            p += torch.linalg.vector_norm(v)
        P.append(p)
        
    return P

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def Area_Fraction(X, nholes, xsize, ysize):
    t, e    = Read_Shape(X, nholes, xsize, ysize)
    at      = Area_Triangle(t)
    ae      = Area_Ellipse(e)
    A       = sum(at) + sum(ae)
    AF      = A/(xsize*ysize)
    return AF

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def Perimeter_Fraction(X, nholes, xsize, ysize):
    t, e    = Read_Shape(X, nholes, xsize, ysize)
    pt      = Perimeter_Triangle(t)
    pe      = Perimeter_Ellipse(e)
    P       = sum(pt) + sum(pe)
    PF      = P/(xsize*ysize)
    return PF

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def Vertices(T, k):
    p0 = T[k, 0, :]
    return torch.stack((p0, p0 + T[k, 1, :], p0 + T[k, 2, :],))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def canonicalize_triangles(nholes, vertices):
    if nholes == 0:
        return torch.empty(0, **tkwargs)

    canonical_vertices  = []
    shape_index  = torch.as_tensor([0], **tkwargs)
    for triangle in vertices:
        start           = min(range(3), key = lambda k: (float(triangle[k, 0].item()), float(triangle[k, 1].item()),),)
        
        # This is selecting which of the 3 vertices should become p0. The min loop picks the index whose (x, y) pair is smallest in lexciographic order: smallest x wins. If x is tied: smallest y wins.
        other           = [k for k in range(3) if k != start]
        p0              = triangle[start]
        p1              = triangle[other[0]]
        p2              = triangle[other[1]]
        cross           = ((p1[0] - p0[0])*(p2[1] - p0[1]) - (p1[1] - p0[1])*(p2[0] - p0[0]))
        
        # Here, cross > 0 means that p0 -> p1 -> p2 in anticlockwise order, so sqapping when cross < 0 forces into anticlockwise order
        if cross < 0:
            p1, p2      = p2, p1
        canonical_vertices.append(torch.stack((p0, p1, p2)))
    
    vertices        = torch.stack(canonical_vertices)
    
    centroids       = vertices.mean(dim=1)
    centroid_order  = sorted(range(nholes), key=lambda k: (float(centroids[k, 0].item()), float(centroids[k, 1].item()),),)
    vertices        = vertices[centroid_order]
    p0              = vertices[:, 0, :]
    edge_1          = vertices[:, 1, :] - p0
    edge_2          = vertices[:, 2, :] - p0

    # Triangle edges are decoded elsewhere as size*(2*u - 1), so signed
    # normalised edge components in [-1, 1] must be encoded back to [0, 1].
    encoded_edge_1  = 0.5*(edge_1 + 1.0)
    encoded_edge_2  = 0.5*(edge_2 + 1.0)
    candidate       = torch.cat((p0, encoded_edge_1, encoded_edge_2,), dim=-1).reshape(-1)
    
    chunks = torch.split(candidate, 6)
    stacked = torch.cat([torch.cat((shape_index, c)) for c in chunks])
    
    return stacked

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def canonicalize_ellipses(nholes, points):
    
    shape_index = torch.as_tensor([1], **tkwargs)
    dummy = torch.as_tensor([0], **tkwargs)
    centroid_order = sorted(range(nholes), key=lambda k: (float(points[k, 0].item()), float(points[k, 1].item()),),)
    points = points[centroid_order]
    rows = [torch.cat((shape_index, ellipse, dummy)) for ellipse in points]
    
    return torch.cat(rows) if rows else torch.empty(0, **tkwargs)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# DELETE THIS AS NO LONGER IN USE ** CHECK **
def ellipse_anchor(points, find_max=False):
    """
    Return the point on a rotated ellipse with minimum x co-ordinate.
    If the minimum x co-ordinate is attained at two points, return the one with smaller y co-ordinate
    """
    cx, cy, rx, ry, theta = points
    
    # Parametric ellipse is:
        # x(t) = cx + rx*cos(t)*cos(theta) - ry*sin(t)*sin(theta)
        # y(t) = cy + rx*cos(t)*sin(theta) + ry*sin(t)*cos(theta)
    # Thus the minimum x-co-ordinate is when:
        # dx/dt = -rx*sin(t)*cos(theta) - ry*cos(t)*sin(theta) = 0
        # so then tan(t) = (-ry*sin(theta)/rx*cos(theta))
    t0 = torch.atan2(-ry*torch.sin(theta), rx*torch.cos(theta))
    t1 = t0 + torch.pi
    
    def point(t):
        x = cx + rx*torch.cos(t)*torch.cos(theta) - ry*torch.sin(t)*torch.sin(theta)
        y = cy + rx*torch.cos(t)*torch.sin(theta) + ry*torch.sin(t)*torch.cos(theta)
        return x, y
    
    x0, y0 = point(t0)
    x1, y1 = point(t1)
    
    if find_max:
        # Select maximum x; tie -> minimum y
        if x0 > x1:
            return torch.stack((x0, y0))
        elif x1 > x0:
            return torch.stack((x1, y1))
        else:
            return torch.stack((x0, y0)) if y0 < y1 else torch.stack((x1, y1))
    else:
        # Select minimum x; tie -> minimum y
        if x0 < x1:
            return torch.stack((x0, y0))
        elif x1 < x0:
            return torch.stack((x1, y1))
        else:
            return torch.stack((x0, y0)) if y0 < y1 else torch.stack((x1, y1))
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def anchor_point(points, maximum, constraints, Normalise):
    xmin, xmax, ymin, ymax  = constraints.box
    
    if Normalise:
        domain_min = points.new_tensor([constraints.bx / constraints.xsize, constraints.by / constraints.ysize,])
        domain_max = points.new_tensor([1.0 - constraints.bx / constraints.xsize, 1.0 - constraints.by / constraints.ysize,])
    else:
        domain_min = points.new_tensor([xmin, ymin])
        domain_max = points.new_tensor([xmax, ymax])
    
    # Triangle
    if points[0] == 0:
        value_points = points[1:].reshape(3,2)
        if maximum:
            return domain_max - value_points.max(dim=0).values
        else:
            return domain_min - value_points.min(dim=0).values
    
    # Ellipse
    else:
        cx, cy, rx, ry, theta = points[1:-1]
        if Normalise:
            theta = torch.pi*theta
        dx = torch.sqrt((rx*torch.cos(theta))**2 + (ry*torch.sin(theta))**2)
        dy = torch.sqrt((rx*torch.sin(theta))**2 + (ry*torch.cos(theta))**2)
        lower_point = torch.stack((cx - dx, cy - dy))
        upper_point = torch.stack((cx + dx, cy + dy))
        if maximum:
            return domain_max - upper_point
        else:
            return domain_min - lower_point

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def smooth_min(x, tau=1e-4, dim=0):
    # <= true min: conservative for constraint >= 0
    return - tau*torch.logsumexp(-x/tau, dim=dim)

def smooth_max_upper(x, tau=1e-4, dim=0):
    # >= true max
    return tau*torch.logsumexp(x/tau, dim=dim)

def smooth_max_lower(x, tau=1e-4, dim=0):
    # <= true max
    n = torch.tensor(x.shape[dim])
    return tau*torch.logsumexp(x/tau, dim=dim) - tau*torch.log(n)
    

def separating_axis_theorem(points_i, points_j, xsize, ysize, dir_trials=100, eps=1e-12):
    """Return a conservative physical clearance between two encoded holes."""
    
    # Order so that always 0 before 1
    Points = torch.stack((points_i, points_j), dim=0)
    _, indices = torch.sort(Points[:, 0])
    Points = Points[indices]
    
    points_i = Points[0]
    points_j = Points[1]
    
    # TT
    if points_i[0] == 0 and points_j[0] == 0:
        vertices = []
        for points in [points_i, points_j]:
            decoded = triangle(points, 0, xsize, ysize)
            p0 = decoded[0:2]
            vertices.append(torch.stack((p0, p0 + decoded[2:4], p0 + decoded[4:6],)))
        
        triangle_i, triangle_j = vertices
        edges = torch.cat((triangle_i.roll(-1, dims=0) - triangle_i, triangle_j.roll(-1, dims=0) - triangle_j,))
        axes = torch.stack((-edges[:, 1], edges[:, 0]), dim=-1)
        axis_lengths = torch.sqrt((axes*axes).sum(dim=-1, keepdim=True) + eps**2)
        axes = axes/axis_lengths
        
        projection_i = triangle_i @ axes.T
        projection_j = triangle_j @ axes.T
        
        i_min = smooth_min(projection_i, dim=0)
        i_max = smooth_max_upper(projection_i, dim=0)
        j_min = smooth_min(projection_j, dim=0)
        j_max = smooth_max_upper(projection_j, dim=0)
        
        gaps = torch.stack((j_min - i_max, i_min - j_max))
        axis_gap = smooth_max_lower(gaps, dim=0)
        separation = smooth_max_lower(axis_gap, dim=0)
        
        #separation = torch.maximum(projection_j.min(dim=0).values - projection_i.max(dim=0).values, projection_i.min(dim=0).values - projection_j.max(dim=0).values,)
        
        return separation
    
    # TE
    elif points_i[0] == 0 and points_j[0] == 1:
        decoded_triangle    = triangle(points_i, 0, xsize, ysize)
        p0                  = decoded_triangle[0:2]
        vertices_tri        = torch.stack((p0, p0 + decoded_triangle[2:4], p0 + decoded_triangle[4:6],))
        edges_tri           = vertices_tri.roll(-1, dims=0) - vertices_tri
        
        cx, cy, rx, ry, theta = ellipse(points_j, 0, xsize, ysize)
        centre = torch.stack((cx, cy))
        
        p = (((centre - vertices_tri)*edges_tri).sum(dim=1) / (edges_tri*edges_tri).sum(dim=1).clamp_min(eps)).clamp(0.0, 1.0)
        proj = vertices_tri + p[:, None]*edges_tri
        distances_sq = ((proj - centre)**2).sum(dim=1)
        closest_point = proj[torch.argmin(distances_sq)]
        
        fourth_axis             = (closest_point - centre)
        normalised_fourth_axis  = fourth_axis/(torch.sqrt((fourth_axis*fourth_axis).sum() + eps**2))
        
        # Triangle edge normals
        axes = torch.stack((-edges_tri[:, 1], edges_tri[:, 0]), dim=-1)
        axis_lengths = torch.sqrt((axes*axes).sum(dim=-1, keepdim=True) + eps**2)
        axes = axes/axis_lengths
        
        axes = torch.cat((axes, normalised_fourth_axis.unsqueeze(0)), dim=0)
        
        # Unit vectors along ellipse semi-axes
        u = torch.stack((torch.cos(theta), torch.sin(theta)))
        v = torch.stack((-torch.sin(theta), torch.cos(theta)))
        
        # Triangle projections
        projection_tri = vertices_tri @ axes.T
        
        tri_min = smooth_min(projection_tri, dim=0)
        tri_max = smooth_max_upper(projection_tri, dim=0)
        
        # Ellipse projections via support radius
        nu = axes @ u
        nv = axes @ v
        r_e = torch.sqrt(nu**2 * rx**2 + nv**2 * ry**2 + eps**2)
        
        centre_projection = centre @ axes.T
        ell_min = centre_projection - r_e
        ell_max = centre_projection + r_e
        
        gaps = torch.stack((ell_min - tri_max, tri_min - ell_max))
        axis_gap = smooth_max_lower(gaps, dim=0)
        separation = smooth_max_lower(axis_gap, dim=0)
        
        #separation = torch.maximum(ell_min - tri_max, tri_min - ell_max,)
        
        return separation
    
    # EE
    else:
        cx1, cy1, rx1, ry1, th1 = ellipse(points_i, 0, xsize, ysize)
        cx2, cy2, rx2, ry2, th2 = ellipse(points_j, 0, xsize, ysize)
        
        centre_i = torch.stack((cx1, cy1))
        centre_j = torch.stack((cx2, cy2))
        d        = centre_j - centre_i
        
        # Ellipse principal directions
        u1 = torch.stack((torch.cos(th1), torch.sin(th1)))
        v1 = torch.stack((-torch.sin(th1), torch.cos(th1)))
        u2 = torch.stack((torch.cos(th2), torch.sin(th2)))
        v2 = torch.stack((-torch.sin(th2), torch.cos(th2)))
        
        # All unit directions on [0, pi); n and -n are equivalent
        phi = torch.arange(dir_trials, device=points_i.device, dtype=points_i.dtype)*torch.pi/dir_trials
        axes = torch.stack((torch.cos(phi), torch.sin(phi)), dim=1)
        centre_gap = torch.abs(axes @ d)
        
        # Ellipse support radii
        ra = torch.sqrt(rx1**2 * (axes @ u1)**2 + ry1**2 * (axes @ v1)**2 + eps**2)
        rb = torch.sqrt(rx2**2 * (axes @ u2)**2 + ry2**2 * (axes @ v2)**2 + eps**2)
        
        separation = centre_gap - ra - rb
        
        return separation.max()


def separating_axis_theorem_batch(points_i, points_j, xsize, ysize, dir_trials=100, eps=1e-12):
    """Vectorised clearances from encoded ``points_i`` to one encoded hole."""
    if points_i.ndim != 2 or points_i.shape[-1] != 7:
        raise ValueError("points_i must have shape (batch, 7).")

    points_j = points_j.reshape(1, 7).expand(points_i.shape[0], -1)
    shape_i = int(points_i[0, 0].item())
    shape_j = int(points_j[0, 0].item())
    if shape_i > shape_j:
        points_i, points_j = points_j, points_i
        shape_i, shape_j = shape_j, shape_i

    scale = points_i.new_tensor([xsize, ysize])

    def triangle_vertices(points):
        p0 = points[:, 1:3]*scale
        v1 = (2.0*points[:, 3:5] - 1.0)*scale
        v2 = (2.0*points[:, 5:7] - 1.0)*scale
        return torch.stack((p0, p0 + v1, p0 + v2), dim=1)

    def ellipse_values(points):
        centre = points[:, 1:3]*scale
        radii = points[:, 3:5]*scale
        theta = torch.pi*points[:, 5]
        return centre, radii, theta

    if shape_i == 0 and shape_j == 0:
        vertices_i = triangle_vertices(points_i)
        vertices_j = triangle_vertices(points_j)
        edges = torch.cat((
            vertices_i.roll(-1, dims=1) - vertices_i,
            vertices_j.roll(-1, dims=1) - vertices_j,
        ), dim=1)
        axes = torch.stack((-edges[:, :, 1], edges[:, :, 0]), dim=-1)
        axes = axes/torch.sqrt((axes*axes).sum(dim=-1, keepdim=True) + eps**2)
        projection_i = torch.einsum("nvc,nac->nva", vertices_i, axes)
        projection_j = torch.einsum("nvc,nac->nva", vertices_j, axes)
        separation = torch.maximum(
            projection_j.min(dim=1).values - projection_i.max(dim=1).values,
            projection_i.min(dim=1).values - projection_j.max(dim=1).values,
        )
        return separation.max(dim=1).values

    if shape_i == 0 and shape_j == 1:
        vertices_tri = triangle_vertices(points_i)
        edges_tri = vertices_tri.roll(-1, dims=1) - vertices_tri
        centre, radii, theta = ellipse_values(points_j)

        edge_lengths_sq = (edges_tri*edges_tri).sum(dim=2).clamp_min(eps)
        projection_parameter = (
            ((centre[:, None, :] - vertices_tri)*edges_tri).sum(dim=2)
            / edge_lengths_sq
        ).clamp(0.0, 1.0)
        projected = vertices_tri + projection_parameter[:, :, None]*edges_tri
        closest_index = ((projected - centre[:, None, :])**2).sum(dim=2).argmin(dim=1)
        closest_point = projected[
            torch.arange(points_i.shape[0], device=points_i.device),
            closest_index,
        ]
        fourth_axis = closest_point - centre
        fourth_axis = fourth_axis/torch.sqrt((fourth_axis*fourth_axis).sum(dim=1, keepdim=True) + eps**2)

        axes = torch.stack((-edges_tri[:, :, 1], edges_tri[:, :, 0]), dim=-1)
        axes = axes/torch.sqrt((axes*axes).sum(dim=-1, keepdim=True) + eps**2)
        axes = torch.cat((axes, fourth_axis[:, None, :]), dim=1)

        projection_tri = torch.einsum("nvc,nac->nva", vertices_tri, axes)
        tri_min = projection_tri.min(dim=1).values
        tri_max = projection_tri.max(dim=1).values

        u = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
        v = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=1)
        nu = (axes*u[:, None, :]).sum(dim=2)
        nv = (axes*v[:, None, :]).sum(dim=2)
        support = torch.sqrt(
            nu**2*radii[:, 0:1]**2
            + nv**2*radii[:, 1:2]**2
            + eps**2
        )
        centre_projection = (axes*centre[:, None, :]).sum(dim=2)
        ell_min = centre_projection - support
        ell_max = centre_projection + support
        separation = torch.maximum(ell_min - tri_max, tri_min - ell_max)
        return separation.max(dim=1).values

    centre_i, radii_i, theta_i = ellipse_values(points_i)
    centre_j, radii_j, theta_j = ellipse_values(points_j)
    delta = centre_j - centre_i
    phi = torch.arange(dir_trials, device=points_i.device, dtype=points_i.dtype)*torch.pi/dir_trials
    axes = torch.stack((torch.cos(phi), torch.sin(phi)), dim=1)
    centre_gap = torch.abs(delta @ axes.T)

    u_i = torch.stack((torch.cos(theta_i), torch.sin(theta_i)), dim=1)
    v_i = torch.stack((-torch.sin(theta_i), torch.cos(theta_i)), dim=1)
    u_j = torch.stack((torch.cos(theta_j), torch.sin(theta_j)), dim=1)
    v_j = torch.stack((-torch.sin(theta_j), torch.cos(theta_j)), dim=1)
    support_i = torch.sqrt(
        radii_i[:, 0:1]**2*(u_i @ axes.T)**2
        + radii_i[:, 1:2]**2*(v_i @ axes.T)**2
        + eps**2
    )
    support_j = torch.sqrt(
        radii_j[:, 0:1]**2*(u_j @ axes.T)**2
        + radii_j[:, 1:2]**2*(v_j @ axes.T)**2
        + eps**2
    )
    return (centre_gap - support_i - support_j).max(dim=1).values
