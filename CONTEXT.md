## CONTEXT.md

# This file is intended to record:
- A concise summary of what solutions/approaches were explored in latest query
- Durable project decisions
- Accepted engineering assumptions
- Validation results
- Unresolved issues

# Every entry should begin with the following formatting:
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 		            DD / MM / YY  - TIME
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                                            22/09/26    at    14:52
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

New One_Shape_Only folder with 260921_Topology_Optimisation_ONESHAPE.py

In comparison to 260820_Topology_Optimisation_MAIN, this has the following changes:

1. No 'improvement count': there is no 'if improvement_count >= 75' so that the code can continue way beyond when it might have otherwise concluded. This ensures whether it can escape local minima in later iterations. As per meeting on 21/09/26 (see blue book), S suggested that the plateau can see in the log-loss graph (see 260911_Meeting5.ppt, slide 6) is because there is not enough exploration in the middle iterations (and thus it is getting stuck in local minima).

2. Changed fixed_features in sampling so that it handles 'htype' which dictates whether **all** holes are triangular or whether **all** holes are elliptical (rather than a random combination of triangle/ellipse). This is because of previous issues with improper sampling of the design space (trying to move to LHS rather than a feasibility-driven sampler). The change from a feasibility-driven sampler to something LHS-based is an investigation required to ensure fair examination of the design space when both triangular and elliptic holes are present.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                                            23/09/26     at      16:32
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

In random_shape_candidate, weights are changed so that all areas have similar areas rather than randomly picking weights that can vary between 0 and 1.

Must increase batch_size, max_batches, anchor_trials and shape_trials in sample_random_feasible to generate enough burn-in (concerned with triangles).

Even still, the code is very slow.

FYI this is following the proposed change in the meeting with B and S on 21/09/26. S suggested two changes to get closer to the global optimum:

1. Let the code continue for longer (see previous date's log)
2. Let the algorithm explore larger perforations (i.e. after burn-in). Currently, the algorithm may be getting stuck in strong local minima because of residue large perforations that are echos from the burn-in.

Starting with triangular holes because need to try and get the same/similar result as see in the folders below (before can move onto assuming that the ellipse-only code gives a correct result):

Folder 1 (using the triangle only code): C:\Users\eleno\OneDrive - Nexus365\Documents\POSTDOC\Code\Topology_Optimisation\Bayesian_Optimisation\Outputs\260813_Test
Folder 2 (using the one_shape_only code, which could accommodate either all triangles or all ellipses) : C:\Users\eleno\OneDrive - Nexus365\Documents\POSTDOC\Code\Topology_Optimisation\Bayesian_Optimisation_Part2\Outputs\One_Shape\Triangle\260923

The results are again stuck in a local optima. Either:
1. Not exploring domain enough - try with n global restarts outnumbering n local?
2. Ask Codex to improve exploration around midway through the iterations?

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                                            25/09/26     at      08:06
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

See outputs --> 260924_Fewer_Local and 260924_Larger_Area_Variation
These contain the output log (so far!) of members with triangular hole optimisation. This starts to addresss Issues 01 and 02 (see ISSUES.md).

260924_Fewer_Local --> ng_rst = 10 and nl_rst = 5, with the weights within sample_random_candidate as:

if stage == "BO_Eval":
        # Exponentially distributed random weights
        weights             = -torch.log(torch.rand(nholes, **tkwargs).clamp_min(1e-12))
    elif stage == "Burn_In":
        # Random similar weights = similar area
        weights             = 0.9 + 0.2 * torch.rand(nholes, **tkwargs)

260924_Larger_Area_Variation --> ng_rst = 8 and nl_rst = 8, with the weights within sample_random_candidate as:

if stage == "BO_Eval":
        # Exponentially distributed random weights
        weights             = -torch.log(torch.rand(nholes, **tkwargs).clamp_min(1e-12))
    elif stage == "Burn_In":
        # Random similar weights = similar area
        weights             = 0.5 + torch.rand(nholes, **tkwargs)

From results so far, both are better than the output found when conduct the following (see 260924_Area_Prior):

ng_rst = 8 and nl_rst = 8, with the weights within sample_random_candidate as:

if stage == "BO_Eval":
        # Exponentially distributed random weights
        weights             = -torch.log(torch.rand(nholes, **tkwargs).clamp_min(1e-12))
    elif stage == "Burn_In":
        # Random similar weights = similar area
        weights             = 0.9 + 0.2 * torch.rand(nholes, **tkwargs)

260924_Fewer_Local is the most superior so far, but 260924_Larger_Area_Variation also outperforms 260924_Area_Prior.
This suggests that a modification might benefit from:

ng_rst = 10 and nl_rst = 5, with the weights within sample_random_candidate as:

if stage == "BO_Eval":
        # Exponentially distributed random weights
        weights             = -torch.log(torch.rand(nholes, **tkwargs).clamp_min(1e-12))
    elif stage == "Burn_In":
        # Random similar weights = similar area
        weights             = 0.5 + torch.rand(nholes, **tkwargs)

But why??

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                                            27/09/26     at      19:16
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

The results are in, see outputs --> 260925_Proposed_Modification
This output is worse than the output given in 260924_Fewer_Local.

Does this suggest that less area variation in the burn-in, with more global exploration is most superior?


Why? One would have expected that 260925_Proposed_Modification, which combines the benefits of 260924_Fewer_Local with 260924_Larger_Area_Variation to be most superior.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                            28/09/26     at      12:51
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Issues 01 and 02 implementation sequence

1. Package foundations and preserved models

The seven-variable hole encoding, canonicalisation, separating-axis calculations,
binary rasterisation, MBB finite-element compliance calculation, and existing
geometric/volume constraints were moved into self-contained modules under
`src/one_shape_bo`. This removes production dependence on the historical import
layout without changing the engineering model. A regression test confirms that
the source package produces exactly the same raster and compliance as the legacy
implementation for a mixed triangle/ellipse design.

2. Global candidate area sampling

`random_shape_candidate` now receives an explicit area-allocation strategy.
Near-equal weights and broad exponential weights are separate named policies.
Burn-in is quota-stratified over both policies, while BO global restart pools use
the broad policy. This prevents the burn-in prior from silently becoming the BO
exploration policy and preserves some broad-area coverage in the initial GP data.

3. Local feasible sampling

`sample_local_feasible` now uses several diverse high-quality observations as
anchors and always applies the user-configured `local_radius`; the previous hidden
attempt-dependent radius contraction was removed. Raster duplicates are rejected.
The configured-radius behaviour is covered by a deterministic test.

4. Restart ranking and batching

`rank_initial_condition_pool` retains global/local provenance, rejects repeated
raster geometries, filters starts close to observations or prior attempts, and
uses acquisition plus novelty when ranking global starts. `take_restart_batch`
selects spatially diverse starts while honouring the requested source allocation.
Shape identity is represented by arbitrary canonical tuples, so mixed tuples such
as `(0, 0, 1)` remain supported even though current runs fix all holes to one type.

5. Acquisition optimisation

`Acquisition` accepts labelled restart entries, keeps source and shape metadata,
records compact timing/warning/failure information, and removes duplicate optimized
raster geometries. Per-restart geometry tensors are no longer printed.

6. Gaussian-process fitting

`Pred_Objective_Multi_Task` remains a `MixedSingleTaskGP` with a Matérn-1.5
continuous kernel and standardized negative-log-compliance response. Mean,
covariance, and likelihood state are warm-started from the preceding iteration.
Lengthscales, outputscales, and likelihood noise are exposed for diagnostics.

7. Main Bayesian optimisation loop and reporting

The source-package driver keeps the total restart allocation fixed. It uses the
configured baseline global/local split until the record-improvement patience is
exceeded, then assigns 75% of starts to global exploration; a new record restores
the baseline split. The local radius is not adapted. Each iteration writes a
compact log line plus CSV diagnostics for restart source, GP hyperparameters, and
posterior standard deviation. The latter is explicitly labelled as uncertainty in
the transformed GP score space. Default output directories now include date and
time so same-day runs do not overwrite one another.

8. Read-only expected-output audit

The stored reference PNG was audited without executing or modifying
`260928_MBB_Beam_Basic.py`. Resampling the rendered topology to a 50 by 100 binary
grid gives solid fraction 0.5332, exceeding the BO allowance of 0.505, with three
void components and recomputed binary compliance 76.7532. The reference log's
96.8918 compliance belongs to the filtered continuous-density field. Therefore
the rendered reference is a qualitative topology target, not a feasible or
numerically equivalent target for the volume-constrained binary three-hole model.
The machine-readable report is in
`outputs/260928_Benchmark_Audit/benchmark_audit.json`.

9. Validation and repository hygiene

Invalid `itertools` and `sys` Conda entries were removed, allowing the declared
Python 3.12 environment to resolve. Python caches are ignored and previously
tracked cache files are removed. Fourteen focused tests cover area-policy quotas,
explicit area allocation, fixed-radius local sampling, source-aware ranking,
mixed-tuple restart selection, adaptive restart allocation, acquisition metadata,
GP warm-starting and real model diagnostics, physical-model regression, benchmark
read-only behaviour, the actual reference audit, and a mocked one-iteration BO
driver. The combined targeted run passed all 14 tests in 4.06 seconds; the only
warnings were upstream PyTorch notices that `torch.jit.script` is deprecated. No
full BO run or convergence benchmark was performed.

Overall summary

The changes address the identified mechanisms behind burn-in sensitivity and
late loss of exploration while preserving constrained GP Bayesian optimisation
and the existing physical assumptions. They make sampling policies explicit,
prevent silent local-radius collapse, preserve global diversity and restart
provenance, warm-start and diagnose the GP, and adapt exploration in response to
observed stagnation rather than absolute iteration count. Whether these changes
improve final compliance and topology across seeds remains unresolved until
separately approved multi-seed BO benchmarks are run.
