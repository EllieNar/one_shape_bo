# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:47:28 2026

@author: pemb6626
"""

import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import GammaPrior
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms import Normalize, Standardize
from botorch.models.hierarchical.conditional_kernel_gp import (
    HierarchicalConditionalKernel,
    HierarchicalConditionalKernelGP,
    _transform_hierarchical_dependencies,
)
from torch.nn import ModuleList


def _warm_start(target_gp, previous_model):
    """Reuse GP hyperparameters without reusing outcome-transform statistics."""
    if previous_model is None:
        return

    source_gp = getattr(previous_model, "gp", previous_model)
    for module_name in ("mean_module", "covar_module", "likelihood"):
        getattr(target_gp, module_name).load_state_dict(
            getattr(source_gp, module_name).state_dict()
        )

class Pred_Objective_Single_Task:
    # This is a GP of compliance. This is necessary to determine the acquisition and direct exploration of the design space, using BO.
    
    def __init__(self, train_x, train_y, bounds, previous_model=None):
        # The target values are transformed before GP fitting using log to minimise skew
        train_score     = - torch.log(train_y.clamp_min(1e-12)) 
        # The GP is constructed with a rough Matern-1.5 kernel. The prior and interval prevent extreme ARD lengthscales
        covar_module    = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(
                                                        nu                     = 1.5,
                                                        ard_num_dims           = train_x.shape[-1],
                                                        lengthscale_prior      = GammaPrior(3.0, 6.0),
                                                        lengthscale_constraint = Interval(0.05, 5.0),))
        
        # Outcome transform statistics are not reused
        self.gp = SingleTaskGP(train_x, train_score, covar_module=covar_module, input_transform=Normalize(d=train_x.shape[-1], bounds=bounds), outcome_transform=Standardize(m=1))
        
        # Copy previous GP hyperparameters as a warm-start
        _warm_start(self.gp, previous_model)
            
        self.best_f = train_score.max()
        mll = ExactMarginalLogLikelihood(likelihood=self.gp.likelihood, model=self.gp)
        fit_gpytorch_mll(mll)
        
    def posterior(self, X, **tkwargs):
        return self.gp.posterior(X, **tkwargs)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . #

class Pred_Objective:
    # The GP, as a MixedSingleTaskGP
    def __init__(self, train_x, train_y, nholes):
        # The target values are transformed before GP fitting using log to minimise skew
        train_score     = - torch.log(train_y.clamp_min(1e-12)) 
        # The GP is constructed with a rough Matern-1.5 kernel. The prior and interval prevent extreme ARD lengthscales
# =============================================================================
#         covar_module    = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(
#                                                         nu                     = 1.5,
#                                                         ard_num_dims           = train_x.shape[-1],
#                                                         lengthscale_prior      = GammaPrior(3.0, 6.0),
#                                                         lengthscale_constraint = Interval(0.05, 5.0),))
# =============================================================================
        categorical = list(range(0, 7*nholes, 7))
        continuous = [i for i in range(train_x.shape[-1]) if i not in categorical]
                
        # Outcome transform statistics are not reused
        # NB, MixedSingleTaskGP does not take covar_module! Find similar replacement
        # AND Normalize supports selecting only continuous dimensions via indices
        # train_y should have the shape (n, m), so for 1 output, this should be (n,1)
        # GP only sees [0, 1], so Normalize is not needed
        self.gp = MixedSingleTaskGP(train_X = train_x,
                                    train_Y = train_score,
                                    cat_dims = categorical,
                                    outcome_transform=Standardize(m=1))
            
        self.best_f = train_score.max()
        mll = ExactMarginalLogLikelihood(likelihood=self.gp.likelihood, model=self.gp)
        fit_gpytorch_mll(mll)
        
    def posterior(self, X, **tkwargs):
        return self.gp.posterior(X, **tkwargs)
        
    
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . #
def matern_factory(batch_shape, ard_num_dims, active_dims):
    # These conditions match those considered in Pred_Objective_Single_Task
    return MaternKernel(
        nu                      = 1.5,
        batch_shape             = batch_shape,
        ard_num_dims            = ard_num_dims,
        active_dims             = active_dims,
        lengthscale_prior       = GammaPrior(3.0, 6.0),
        lengthscale_constraint  = Interval(0.05, 5.0),)


class Pred_Objective_Multi_Task:
    """Single-output GP for mixed categorical and continuous design variables."""
    
    def __init__(self, train_x, train_y, nholes, previous_model=None):
        train_score     = - torch.log(train_y.clamp_min(1e-12)) 
        categorical = list(range(0, 7*nholes, 7))
    
        self.gp = MixedSingleTaskGP(train_X = train_x,
                                    train_Y = train_score,
                                    cat_dims = categorical,
                                    cont_kernel_factory = matern_factory,
                                    outcome_transform=Standardize(m=1))
        
        # Copy previous GP hyperparameters as a warm-start
        _warm_start(self.gp, previous_model)
            
        self.best_f = train_score.max()
        mll = ExactMarginalLogLikelihood(likelihood=self.gp.likelihood, model=self.gp)
        fit_gpytorch_mll(mll)
        
    def posterior(self, X, **tkwargs):
        return self.gp.posterior(X, **tkwargs)
    
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . #
class Hierarchical_Kernel(HierarchicalConditionalKernel):
    
    def construct_individual_kernels(self):
        kernels = ModuleList()
        
        parent_indices = set(self.hierarchical_dependencies)

        for indices in self.partition:
            # With separate_hierarchical_features=True, a block consisting only
            # of parent flags is genuinely categorical. The remaining blocks
            # are continuous and use the same Matérn kernel as the mixed GP.
            if set(indices).issubset(parent_indices):
                base = CategoricalKernel(
                    batch_shape=self.batch_shape,
                    ard_num_dims=len(indices),
                    active_dims=indices,
                    lengthscale_constraint=GreaterThan(1e-6),
                )
            else:
                base = MaternKernel(
                    nu                      = 1.5,
                    batch_shape             = self.batch_shape,
                    ard_num_dims            = len(indices),
                    active_dims             = indices,
                    lengthscale_prior       = GammaPrior(3.0, 6.0),
                    lengthscale_constraint  = Interval(0.05, 5.0),)
            
            kernels.append(ScaleKernel(base))
        
        return kernels
    
class Hierarchical_Conditional_GP(HierarchicalConditionalKernelGP):
    
    def __init__(self, train_X, train_Y, hierarchical_dependencies, eval_hierarchical_features=False, separate_hierarchical_features=True, train_Yvar=None, outcome_transform=Standardize(m=1), previous_model=None,):
        transformed_dependencies = _transform_hierarchical_dependencies(
        train_X=train_X,
        hierarchical_dependencies=hierarchical_dependencies,
        input_transform=None,)
        
        covar_module = Hierarchical_Kernel(
            dim                             = train_X.shape[-1],
            hierarchical_dependencies       = transformed_dependencies,
            eval_hierarchical_features      = eval_hierarchical_features,
            separate_hierarchical_features  = separate_hierarchical_features,
            use_saas_prior                  = False,
            use_outputscale                 = True,)
        
        SingleTaskGP.__init__(self, train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar, covar_module=covar_module, outcome_transform=outcome_transform,)
        
        _warm_start(self, previous_model)


class Pred_Objective_Hierarchical:
    
    def __init__(self, train_x, train_y, nholes, previous_model=None):
        train_score     = - torch.log(train_y.clamp_min(1e-12)) 
        # The final coordinate in each hole block is active for triangles (0)
        # and is a fixed dummy for ellipses (1).
        hierarchical_dep = {7*h: {0: [7*h + 6]} for h in range(nholes)}
        
        
        self.gp = Hierarchical_Conditional_GP(
            train_X=train_x,
            train_Y=train_score,
            hierarchical_dependencies=hierarchical_dep,
            eval_hierarchical_features=True,
            outcome_transform=Standardize(m=1),
            previous_model=previous_model,
        )
            
        self.best_f = train_score.max()
        mll = ExactMarginalLogLikelihood(likelihood=self.gp.likelihood, model=self.gp)
        fit_gpytorch_mll(mll)
        
    def posterior(self, X, **tkwargs):
        return self.gp.posterior(X, **tkwargs)


# Backwards-compatible alias for existing scripts using the original spelling.
Pred_Objective_Heirarchical = Pred_Objective_Hierarchical
