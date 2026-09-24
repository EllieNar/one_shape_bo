# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:05:17 2026

@author: pemb6626
"""
# See 10/09 Burn In Sampling Chat GPT
import torch
import numpy as np
#from Sampling import feasible_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {
    "device": device,
    "dtype": torch.double,
}

#from Sampling import fixed_features
#from Shape import canonicalize_triangles

def sample_stratified_burnin(n_burn_in, constraints, lh_multiplier):
    """
    Generates an oversized Latin Hypercube pool of feasible designs.
    Enables unbiased DOE coverage
    Oversized pool to improve efficiency as a feasibility generator
    
    I N P U T S
    - - - - - -
    n_burn_in:      number of burn-in samples required
    
    constraints:    from the Bounds class
    
    lh_multiplier:  inflates the size of the lh matrix to improve feasibility generation
    
    
    O U T P U T S
    - - - - - - -
    strata:         a tensor of size n_burn_in x 7*nholes
    
    """
    
    shape_features = fixed_features(constraints.nholes)
    quotient, remainder = divmod(n_burn_in, len(shape_features))
    
    # Let the collections of shape features be sorted into 'strata'
    # e.g. for 3 holes --> TTT, TTE, TEE, EEE, so 4 shape collections
    # shape collections = nholes + 1
    # strata ensures that the n_burn_in is distributed equally among the 4 shape collections
    # so that no particular collection is preferential
    
    strata = [] # Make a tensor, see end of for loop: concatenate to strata in rows
    iters = 0
    
    for index, features in enumerate(shape_features):
        ns = quotient + int(index < remainder)
        if ns == 0:
            continue
        
        # Here is where sample LH and append to batch
        
        # Takes the features dictionary and returns the latent dimension
        ld, ffl = StratumSpec(features, constraints)
        
        # Make the dimensionless LH, returns U i.e. [0, 1]^(n_s x d) dimensional tensor
        U = Hypercube_Party(lh_multiplier*ns, ld).make_optimal_lh()
        
        # Decode the output (from latent space into physical feature space)
        U_pool, X_pool, domain_valid = decode_lhs(U, constraints, ffl).shape_decoder()
        
        if X_pool.shape[0] < ns:
            raise RuntimeError(
                "LHS pool produced fewer than ns domain-valid designs. "
                "Increase lh_multiplier or regenerate the LHS."
            )
        
        # Check feasibility and constraint violation.
        # Repair. Refine feasible pool down to K where ns < K < lh_multiplier*ns
        feasible_pool = constraint_violation_and_repair_lhs(U_pool, X_pool)
        
        # Check raster and volumne adjustment
        raster_volume_adjustment(feasible_pool)
        canonicalaise_design(feasible_pool)
        select_maximin_subset(feasible_pool) # -> Tensor of dimension (ns, nholes*7)
        
        # Concatenate to strata in rows
        strata.append(feasible_pool)
    
    if not strata:
        raise RuntimeError("Could not generate any feasible points.")
    return torch.cat(strata, dim=0) # -> Tensor of dimension (n_burn_in, nholes*7)
        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def StratumSpec(features, constraints):
    # features is a dictionary e.g. {0: 0.0, 7: 0.0, 14: 0.0}
    fixed_features_list = [int(features[7*h]) for h in range(constraints.nholes)]
    ld                  = constraints.nholes - 1
    ld += sum(5 if c == 0 else 4 for c in fixed_features_list)
    return ld, fixed_features_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

class Hypercube_Party:
    
    def __init__(self, size, ld):
        self.size       = size
        self.lf         = ld
        
        self.min_list   = [0]*ld
        self.max_list   = [1]*ld
    
    def random_in_range(self, mini, maxi, n_samples, rs=False):
        p = np.linspace(mini, maxi, n_samples+1)
        l = p[:-1]
        u = p[1:]
        o = np.random.uniform(low=l, high=u, size=[1,n_samples]).T
        if rs:
            np.random.shuffle(o)
            return torch.from_numpy(o)
        return torch.from_numpy(o)
    
    def latin_hypercube(self):
        for count, (mini, maxi) in enumerate(zip(self.min_list, self.max_list)):
            rs = False if count == 0 else True
            col = self.random_in_range(mini, maxi, self.size, rs)
            if count == 0:
                df = col
            else:
                df = torch.concat((df, col), axis=1)
        return df
    
    def maximin(self, df):
        min_dist = torch.inf
        
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                dist = torch.linalg.norm(df[i,:] - df[j,:])
                min_dist = min(min_dist, dist)
                
        return min_dist
    
    def phi_p(self, df, p = 50):
        phi = 0
        
        for i in range(self.size - 1):
            for j in range(i + 1, self.size):
                dist = torch.linalg.norm(df[i,:] - df[j,:])
                phi += dist**(-p)
        
        return phi**(1/p)    
    
    def make_optimal_lh(self, measure=0, n_candidates=100, succeeded = False):
        best_lh     = None
        
        if measure == 0: # maximun
            best_score = -torch.inf
        if measure == 1: # phi_p
            best_score  = torch.inf
        
        for _ in range(n_candidates):
            
            lh      = self.latin_hypercube()
            
            if succeeded.numel() != 0:
                
                lh = torch.concat((lh, succeeded), axis = 0)
            
            if measure == 0:
                score   = self.maximin(lh)
                if score > best_score:
                    best_lh = lh
                    best_score = score
                    
            elif measure == 1:
                score   = self.phi_p(lh)
                if score < best_score:
                    best_lh = lh
                    best_score = score
                    
        if succeeded.numel() != 0:
            
            lh = torch.drop((lh, succeeded), axis=0)
        
        return best_lh
    
# Implement within sample_stratified_burnin as:
#U = Hypercube_Party(lh_multiplier*ns, ld).make_optimal_lh()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

# Canonicalise triangle batch
# Compare this to canonicalise_triangles within Shape
def canonicalise_triangle_burnin(vertices, xsize, ysize):
    """
    I N P U T
    - - - - -
    vertices: (M, 3, 2), physical co-ordinates
    
    O U T P U T
    - - - - - -
    block: (M, 7)
    [0, p0x, p0y, v1x, v1y, v2x, v2y], normalized/encoded
    """
    
    # Lexicographic order: x first, then y
    # Stable y-sort followed by stable x-sort
    
    iy          = torch.argsort(vertices[:, :, 1], dim=1, stable=True)
    v           = torch.gather(vertices, 1, iy.unsqueeze(-1).expand(-1, -1, 2))
    
    ix          = torch.argsort(v[:, :, 0], dim=1, stable=True)
    v           = torch.gather(v, 1, ix.unsqueeze(-1).expand(-1, -1, 2))
    
    p0          = v[:, 0, :]
    p1          = v[:, 1, :]
    p2          = v[:, 2, :]
    
    # Force p0 -> p1 -> p2 anticlockwise
    cross       = ((p1[:, 0] - p0[:, 0])*(p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1])*(p2[:, 0] - p0[:, 0]))
    swap        = cross < 0
    old_p1      = p1.clone()
    
    p1          = torch.where(swap[:, None], p2, p1)
    p2          = torch.where(swap[:, None], old_p1, p2)
    
    edge_1      = p1 - p0
    edge_2      = p2 - p0
    
    scale       = p0.new_tensor([xsize, ysize])
    
    # Shape.py expects these normalisations
    p0_norm     = p0/scale
    e1_norm     = edge_1/scale
    e2_norm     = edge_2/scale
    
    encoded_e1  = 0.5*(e1_norm + 1.0)
    encoded_e2  = 0.5*(e2_norm + 1.0)
    
    return torch.cat((torch.zeros((vertices.shape[0], 1), **tkwargs), p0_norm, encoded_e1, encoded_e2), dim=1)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

class decode_lhs:
    
    def __init__(self, U, constraints, fixed_features_list):
        
        self.U          = U
        self.nholes     = constraints.nholes
        self.amin       = constraints.amin
        self.xsize      = constraints.xsize
        self.ysize      = constraints.ysize
        self.bx         = constraints.bx
        self.by         = constraints.by
        self.solid_max  = constraints.solid_max
        self.ffl        = fixed_features_list # (e.g. [0, 0, 1])
        self.decoded    = torch.empty((self.U.shape[0], 0), **tkwargs)
        
        self.global_xmin, self.global_xmax = self.bx, self.xsize - self.bx
        self.global_ymin, self.global_ymax = self.by, self.ysize - self.by
        
# Test __init__
# =============================================================================
#     def __init__(self, U):
#         
#         self.U          = U
#         self.nholes     = 3
#         self.amin       = 10
#         self.xsize      = 100
#         self.ysize      = 100
#         self.bx         = 1
#         self.by         = 1
#         self.solid_max  = 0.5
#         self.ffl        = [0, 0, 1]
#         self.decoded    = torch.empty((self.U.shape[0], 0), **tkwargs)
#         
#         self.global_xmin, self.global_xmax = self.bx, self.xsize - self.bx
#         self.global_ymin, self.global_ymax = self.by, self.ysize - self.by
# =============================================================================
        
        
    def decode_area(self):
        
        u1 = self.U[:, 0]
        u2 = self.U[:, 1]

        s   = torch.sqrt(u1)
        # Compare to the weight-allocation approach within Sampling --> random_shape_candidate
        # Must remove any randomness for proper LH representation
        # Could peturb the areas within some min/max limit later e.g. within the constraint_viloation_and_repair step
        target_void_area = (1 - self.solid_max)*(self.xsize*self.ysize)
        min_total_area              = self.nholes*self.amin
        
        if target_void_area < min_total_area:
            return None
        
        remaining_area      = target_void_area - min_total_area
        
        a1                  = self.amin + remaining_area*(1 - s)
        a2                  = self.amin + remaining_area*s*(1 - u2)
        a3                  = self.amin + remaining_area*s*u2
        
        return torch.stack((a1, a2, a3), dim=1)
    
    def decode_triangle(self, i, col, area):
        """
        Decode one triangle hole for all M LHS rows
        
        O U T P U T S
        - - - - - - -
        block : (M, 7)
        valid : (M,)
        """
        
        ux_latent       = self.U[:, col]
        uy_latent       = self.U[:, col+1]
        rho_latent      = self.U[:, col+2]
        phi_latent      = self.U[:, col+3]
        alpha_latent    = self.U[:, col+4]
        
        # Edge-length ratio for 0.2 <= rho <= 1
        rho             = 0.2 + 0.8*rho_latent
        # Included Angle for 0 <= phi <= pi
        # Avoid exactly degenerate triangles
        eps_phi = 1e-6
        phi = eps_phi + (torch.pi - 2*eps_phi)*phi_latent
        # Absolute orientation for 0 <= alpha <= 2*pi
        alpha           = 2*torch.pi*alpha_latent
        
        dir1            = torch.sqrt(2*area / (rho*torch.sin(phi)))
        dir2            = rho*dir1
        
        p0              = torch.zeros((self.U.shape[0], 2), **tkwargs)
        p1              = dir1.unsqueeze(1) * torch.stack((torch.cos(alpha), torch.sin(alpha)), dim=1)
        p2              = dir2.unsqueeze(1) * torch.stack((torch.cos(alpha + phi), torch.sin(alpha + phi)), dim=1)
        
        local_xmin      = torch.minimum(p1[:, 0], torch.minimum(p1[:, 0], p2[:, 0]))
        local_xmax      = torch.maximum(p1[:, 0] , torch.maximum(p1[:,0], p2[:, 0]))
        local_ymin      = torch.minimum(p1[:, 1], torch.minimum(p1[:, 1], p2[:, 1]))
        local_ymax      = torch.maximum(p1[:, 1] , torch.maximum(p1[:,1], p2[:, 1]))
        
        # Valid translation intervals
        tx_min          = self.global_xmin - local_xmin
        tx_max          = self.global_xmax - local_xmax
        ty_min          = self.global_ymin - local_ymin
        ty_max          = self.global_ymax - local_ymax
        
        valid = (tx_min <= tx_max) & (ty_min <= ty_max)

        tx = tx_min + ux_latent*(tx_max - tx_min)
        ty = ty_min + uy_latent*(ty_max - ty_min)
    
        translation = torch.stack((tx, ty), dim=1)
        
        vertices = torch.stack((p0 + translation, p1 + translation, p2 + translation), dim=1)
        
        block =  canonicalise_triangle_burnin(vertices, self.xsize, self.ysize)

        block = block.masked_fill(~valid[:, None], torch.nan)
        
        return block, valid
    
    def decode_ellipse(self, i, col, area):
        """
        Decode one ellipse hole for all M LHS rows
        
        O U T P U T S
        - - - - - - -
        block : (M, 7)
        valid : (M,)
        """
        
        ux_latent       = self.U[:, col]
        uy_latent       = self.U[:, col+1]
        q_latent        = self.U[:, col+2]
        theta_latent    = self.U[:, col+3]
        col += 4
        
        # q = minor-axis / major_axis
        # max aspect ratio = 20
        # so let 1/20 <= q <= 20
        q               = 1/20 + (20 - 1/20)*q_latent
        # Major axis orientation: theta ratio for 0 <= theta <= pi
        theta           = torch.pi*theta_latent
        
        rx              = torch.sqrt(area/(torch.pi*q))
        ry              = q*rx
        
        # Bounding box half extents of rotated ellipse
        dx              = torch.sqrt((rx*torch.cos(theta))**2 + (ry*torch.sin(theta))**2)
        dy              = torch.sqrt((rx*torch.sin(theta))**2 + (ry*torch.cos(theta))**2)
        
        cx_min          = self.global_xmin + dx
        cx_max          = self.global_xmax + dx
        cy_min          = self.global_ymin + dy
        cy_max          = self.global_ymax + dy
        
        valid           = ((cx_min <= cx_max) & (cy_min <= cy_max))
        
        cx              = cx_min + ux_latent*(cx_max - cx_min)
        cy              = cy_min + uy_latent*(cy_max - cy_min)
        
        block           = torch.stack((torch.ones_like(cx), cx/self.xsize, cy/self.ysize, rx/self.xsize, ry/self.ysize, theta/torch.pi, torch.zeros_like(cx),), dim=1)
        
        # Enforce normalisation
        valid           &= torch.isfinite(block).all(dim=1)
        valid           &= ((block >= 0 ) & (block <= 1)).all(dim=1)
        
        block           = block.masked_fill(~valid[:, None], torch.nan)
        
        return block, valid
    
    def shape_decoder(self):
        
        """
        Returns only those rows for which every induvidual fits within the domain
        
        u_valid :  (K, kd)
        x_valid :  (K, 7*nholes)
        valid   :  (M,)
        
        where K <= M
        """
        
        A               = self.decode_area()
        M               = self.U.shape[0]
        col             = self.nholes - 1
        
        blocks          = []
        valid_all       = torch.ones(M, dtype=torch.bool, device=self.U.device)
        
        for i, shape_type in enumerate(self.ffl):
            print(i, shape_type)
            if shape_type == 0:
                block, valid = self.decode_triangle(i=i, col=col, area=A[:,i])
                col += 5
            elif shape_type == 1:
                block, valid = self.decode_ellipse(i=i, col=col, area=A[:,i])
                col += 4
            else:
                raise ValueError("Shape type must be 0 or 1")
            
            blocks.append(block)
            valid_all &= valid
        
        # Still at M x (7*nholes) at this point
        X               = torch.cat(blocks, dim=1)
        valid_all       &= torch.isfinite(X).all(dim=1)
        valid_all       &= ((X >= 0.0) & (X <= 1.0)).all(dim=1)
        
        X_valid         = X[valid_all]
        U_valid         = self.U[valid_all]
        
        return U_valid, X_valid, valid_all
            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

class constraint_violation_and_repair_lhs:
    
    def __init__(self, U_pool, X_pool, constraints, lh_multiplier, ns, ld):
        """
        P A R A M E T E R S
        
        U_pool --> the feasible latent dimension space
        X_pool --> the feasible real dimension space
        
        """
        
        self.U_pool         = U_pool
        self.X_pool         = X_pool
        self.constraints    = constraints
        self.lh_multiplier  = lh_multiplier
        self.ns             = ns
        self.ld             = ld
        
    def check_feasible(self):
        
        X_feas = feasible_mask(self.X_pool, self.constraints)
        
        # Apply the mask onto U_pool:
            
        U_feas = # appropriate mask #
        
        return X_feas, U_feas
        
    def repair_lhs(self):
        
        X_feas, U_feas = self.check_feasible()
        
        U_new = Hypercube_Party(self.lh_multiplier*self.ns, self.ld).make_optimal_lh(U_feas)
        
        return U_new
        
        
        




lh_multiplier = 120
ns            = 4
ld = 16            

U = Hypercube_Party(lh_multiplier*ns, ld).make_optimal_lh()
U_valid, X_valid, valid_all  = decode_lhs(U).shape_decoder()

print(X_valid)
    