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

`sample_local_feasible` uses several diverse high-quality observations as anchors.
It first applies the user-configured `local_radius`, then explicitly falls back to
0.5, 0.25, and 0.125 times that radius after unsuccessful anchor sweeps. Raster
duplicates are rejected, the existing attempt cap is retained, and an unfilled
request returns a partial pool. Deterministic tests cover the initial radius,
contraction, early stopping, and partial return.

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
the baseline split. The configured local radius is not adapted between iterations.
Each iteration writes a
compact log line plus CSV diagnostics for restart source, GP hyperparameters, and
posterior standard deviation at the selected candidate, incumbent at model-fitting
time, and fixed best burn-in design. These are explicitly labelled as uncertainty
in the transformed GP score space. Default output directories now include date
and time so same-day runs do not overwrite one another.

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
use explicit bounded local-radius contraction, preserve global diversity and
restart provenance, warm-start and diagnose the GP, and adapt exploration in
response to observed stagnation rather than absolute iteration count. Whether
these changes improve final compliance and topology across seeds remains
unresolved until separately approved multi-seed BO benchmarks are run.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                                            29/09/26     at      10:20
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

Issue 04 approved correction

The `260928_Implemented_Corrections` run requested 15 raw local candidates in
most BO iterations but produced a median of zero. The local sampler perturbed
all active coordinates at the full configured radius for as many as 5000
attempts, so designs near the geometric and volume boundaries were usually
rejected before entering the restart pool. The user wants local sampling to aim
for `raw_pool_multiplier * nl_rst` candidates without increasing expensive
objective evaluations or preventing BO from proceeding when that target cannot
be filled.

The approved correction restores only the useful within-call radius contraction:
`local_radius` is the initial and maximum radius, followed after unsuccessful
anchor sweeps by factors 0.5, 0.25, and 0.125. The existing attempt ceiling and
partial-pool return are retained, and sampling still stops immediately when the
requested count is reached. The legacy iteration-to-iteration expansion or
contraction of `local_radius` will not be restored; restart allocation, LogEI,
GP fitting, physical constraints, and the number of objective evaluations remain
unchanged. Cheap diagnostics will distinguish requested from generated local
candidates, while posterior uncertainty will additionally be measured at the
current incumbent and at a fixed best burn-in design so that it can be compared
across iterations. Validation is limited to targeted unit and smoke tests rather
than a full BO or benchmark run.

Implementation outcome

The local sampler now advances through the approved four radius factors after two
complete sweeps of its available anchors, within the unchanged 5000-attempt cap.
It stops as soon as the requested count is filled and otherwise returns its partial
accepted set. The BO loop records the requested local count, generated count,
attempts used, and smallest radius reached. Its existing posterior query is batched
over the selected candidate, incumbent at model-fitting time, and fixed best
burn-in design, so the additional uncertainty diagnostics require no extra GP fit
or physical evaluation. The restart controller, LogEI, physical constraints, and
iteration-level radius remain unchanged. Thirteen targeted sampling, restart,
acquisition, GP, and one-iteration smoke tests passed in 3.84 seconds; no full BO
or benchmark run was performed.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                            01/10/26
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Issue 04 follow-up and acceptance criterion

The full local pool in `260929_104907_Triangle` changed the effective acquisition
allocation from mostly global starts to the intended global/local split, but it
also made 353 of 525 winners local and stalled at compliance 95.0638 from
iteration 93 to 379. Every local sampling call reached 0.125 times
`local_radius`. A shared random generator also meant that changing the number of
local attempts changed every later global candidate, so equal seeds did not give
a controlled comparison with `260928_Implemented_Corrections`.

The approved revision will use independent global and local random streams and a
quota-stratified local pool over radius factors 1.0, 0.5, 0.25, and 0.125. Failed
broad-radius quotas will remain unfilled rather than all transferring to the
smallest radius; the existing batch fallback can then use additional global
starts. Radius and anchor provenance will be retained through ranking,
acquisition optimisation, and diagnostics. The GP, LogEI, physical model,
feasibility constraints, total acquisition restart budget, and objective
evaluation count remain unchanged.

The user requires the revision to match or improve the seed-0 result of 83.6612.
A full 525-iteration run takes about nine hours, so the initial regression gate is
the real problem truncated at iteration 150: the revision must reach at most
94.0635, the accepted run's best value by iteration 139, and must not reproduce
the excessive local-winner concentration. This checkpoint detects the known
regression (`260929_104907_Triangle` remained at 95.0638) while avoiding the final
375 increasingly expensive GP fits. Deterministic tests will additionally prove
that local sampling cannot perturb the global random stream. Behavioural edits
will not be retained if this gate fails.

Acceptance-gate outcome and rollback

The proposed independent random streams and quota-stratified local-radius pool
passed 15 targeted tests, but failed the real seed-0 objective gate. A staged
checkpoint was added at iteration 50 because the accepted 260928 trajectory had
already reached 101.4588 there. The revised policy reached only 115.4623, also
worse than the 98.0742 reached by the already-rejected 260929 policy at the same
checkpoint, and local winners again dominated the observed acquisition choices.
The run was stopped at that point instead of spending several more hours on the
remaining iterations. It used the real constraints, GP, acquisition optimiser,
and physical evaluation; only plotting and the read-only benchmark audit were
disabled, so this was a direct early-trajectory regression check rather than a
synthetic unit proxy.

Because the user required the result to match or improve on the accepted seed-0
run, the proposed behavioural revision was rejected. The package source, public
documentation, and tests were restored exactly to commit `9536665`, the code
state that produced `260928_Implemented_Corrections` and its final compliance of
83.6612. This restores the former fixed-radius, partial local-pool behaviour and
shared random stream; it also removes the Issue 04 contraction diagnostics added
after that commit. No output directory or stored benchmark result was changed.
The earlier Issue 04 implementation sections above are therefore historical
records of experiments, not descriptions of the current package behaviour.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      06/10/26 at 10:43 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Issue 04 archive-restart implementation decision

The radius-contraction experiments treated `local_radius` as though it defined a
local search region, but it was only the per-coordinate standard deviation of
random acquisition-optimizer starts. The acquisition optimiser retained the full
box bounds, so contraction increased feasible starts without mathematically
constraining the resulting search. It filled the local pool but overrepresented
closely related starts and produced a worse seed-0 BO trajectory.

The approved alternative uses already evaluated, feasible designs only as local
initial conditions for constrained LogEI optimisation. It selects the incumbent
and then diverse members of a high-performing archive using distance induced by
the fitted GP kernel. These points are not proposed for physical re-evaluation:
the optimized endpoint must remain feasible, raster-valid, and novel. This is a
multi-start acquisition-initialisation policy, not particle swarm optimisation
or a replacement for GP Bayesian optimisation.

The existing radius method remains the default and is retained for a controlled
comparison. The archive method is opt-in, continues to honour the configured raw
pool multiplier and restart allocation, supports canonical mixed shape tuples,
and falls back to the available partial archive if necessary. Diagnostics will
record local start generation, acquisition improvement, start-to-end movement,
optimizer failures, and posterior uncertainty at comparable points. BoTorch
untransforms the outcome standardisation in `posterior()`, so uncertainty is to
be labelled in negative-log-compliance score units rather than standardized
units. No physical assumptions, feasibility tolerances, GP model, LogEI policy,
or expensive-evaluation count are changed.

Implementation in this task is limited to the sampling/acquisition policy,
diagnostics, documentation, and targeted tests. Multi-seed comparison, acceptance
decisions, and changing the default strategy are explicitly reserved for the user
and are not part of this implementation.

Implementation outcome

`OptimizationConfig.local_start_strategy` and the matching CLI option now select
between `radius` and `archive`, with `radius` unchanged as the default. Archive
selection deduplicates the high-performing observation set by raster geometry,
keeps the incumbent, represents previously unseen canonical shape tuples where
available, and applies greedy max-min selection in normalized GP-kernel distance.
Observed archive designs may pass restart ranking as initial conditions, but the
main BO loop still rejects any optimized endpoint whose raster signature has
already been evaluated.

Restart metadata now retains its starting design and archive provenance through
acquisition optimisation. The CSV and compact iteration log report requested and
generated local starts, acquisition gain, normalized start-to-end distance,
GP-kernel movement, failures, and posterior standard deviations at the selected
candidate, incumbent, and fixed best burn-in design. The diagnostic plot and
README use the corrected negative-log-compliance score-unit label. Fifteen
targeted sampling, mixed-shape, acquisition, GP, restart-policy, and one-iteration
driver tests passed in 3.60 seconds; the only warnings were upstream PyTorch
`torch.jit.script` deprecations. No BO benchmark, multi-seed comparison, default
strategy change, or acceptance decision was performed.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      06/10/26 at 15:53 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

outputs -> 261006_Archive_Seed0: the above change after ~ 10 iterations.

This was very slow - 10 iterations took ~ 1 hour.

Codex was asked:
I am now running the latest changes when local_start_strategy is archive. The evaluations are completed particularly slowly (see 261006_Archive_Seed0).
In plan mode: evaluate why the latest code is so slow and propose a modification which could accelerate this, without compromising the purpose of the modifications made in the latest commit.

Codex reported:
The slowdown is most likely caused by acquisition optimisation, specifically the five newly guaranteed archive restarts—not by archive selection itself.

The important difference is:
- 260928_Implemented_Corrections requested 10G/5L, but its radius sampler usually generated zero usable local starts.
- Archive mode reliably supplies all five local starts.
- Every supplied start is currently optimized independently with:
  - maxiter=250
  - ftol=1e-9
  - up to 60 seconds per restart
- Therefore archive mode adds five potentially long constrained optimisations per BO iteration.

Other observations:
- Archive selection initially computes only a small kernel-distance matrix over approximately 45 elite observations. This should take seconds at most, not minutes.
- The physical evaluation uses SciPy’s CPU sparse solver once per BO iteration. It is unchanged by the archive commit.
- Exact GP refitting will become progressively more expensive as the training set grows, but it was already present in the accepted version.
- CUDA provides limited benefit here because SciPy drives many small, sequential PyTorch evaluations with CPU–GPU synchronization. BoTorch normally gains speed by batching restarts, but the nonlinear constraints prevent that batching. BoTorch optimization documentation
- Starting exactly at observed archive points may also make LogEI locally flat or poorly conditioned. The small reported archive movements support this possibility, but it cannot currently be confirmed because per-restart timings are collected internally and then discarded.

Proposed modification
Use a two-stage budget only for archive restarts:
1. Optimize all five archive starts using a screening budget:
   - maxiter=50
   - ftol=1e-6
   - timeout_sec=15
2. Select the best feasible archive result from that screening pass.
3. Refine that one archive result using the existing full settings:
   - maxiter=250
   - ftol=1e-9
   - timeout_sec=60
4. Optimize all global restarts exactly as now.
This reduces as many as five full archive optimisations to five short searches plus one full refinement.
It preserves the purpose of the latest commit:
- the incumbent and GP-kernel-diverse archive members still provide all local starts;
- every archive start still participates in LogEI optimisation;
- the best archive basin still receives the original high-accuracy refinement;
- global exploration is unchanged;
- the GP, LogEI acquisition, physical objective, geometric constraints, restart allocation and novelty checks remain unchanged;
- no expensive physical evaluation is removed or approximated.
The tolerance change is limited to screening. The final archive candidate is still refined with the current tolerance. This is safer than globally reducing maxiter, removing restarts, or lowering the restart allocation.

The proposed modifications were accepted, and the new code was run. This produced:

ouputs -> 261006_Archive_Seed0_V2 (run for ~ 10 iterations).
Observe that this performs much worse than the baseline best in 260928_Implemented_Corrections.

The code is returned to the state before this modification, i.e. the commit 'Intelligent restart selection outputs and WSL diagnostics'.
This ends with the commit of the 261006_Archive_Seed0_V2 output 'Unintelligent restart selection results'.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      09/10/26 at 17:14 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

See 261009_One_Shape_BO_Runs.xlsx. This is the a comparison of the most important results in outputs, after commit 31a624a.
Expect a design similar to the warren truss, know that when this is found across multiple seeds, triangles is setup correctly, and thus can run ellipses confidently.
Note that will have to change the geometry, loading etc to satisfy actual constrains/conditions. Note that may also impose a smaller minimum spacing or a greater bx/by eventually.

- 260928_Implemented_Corrections used a fixed-radius sampling, but usually generated no local raw pool: median 0. It had 102 local winners and finished at a compliance of 83.6612.
- 260929_10407_Triangle contracted the radius until the local pool was filled: median 15. Local winners increased to 353/525, prolonged stagnation occured, and final compliance worsened to 93.1552, despite the same seed.

The later modification adds the 'archive'. This is a better way of searching for local restarts (see 261007_Additional_Notes.pdf - or appropriately named (similar date) in the Bayesian_Optimisation_Part3 folder). This is used by the final three outputs, which also use the same seed of 967. The differences are:
- 261008_085508_Triangle used 5 local and 10 global restarts (htype = 0, triangle) -> Reached smallest compliance yet of 79.982. It remained local-heavy, with 332 winners being local. It still experienced substantial stagnation (perhaps overcome with diversity enabled/disabled later on in iteration? Or perhaps this won't work when the same shape?)
- 261008_172944_Triangle used 8 local and 8 global restarts (htype = 0, triangle) -> Depite everything else the same as the above, compliance of only 96.4448, worse than even 260929_10407_Triangle
- 261008_173626_Ellipse used 5 local and 10 global restarts (htype = 1, ellipse)

Why, is 261008_085508_Triangle superior to 61008_172944_Triangle, when it uses fewer restarts.
Well, because it wasn't properly controlled! It used the wrong starting seed, which had a worst burn-in pipulation.
Further, global restarts decreased from 10 to 8, the increase was entirely local/archive restarts. More acquisition restarts improve the chance of maximising the model's acquisition function, not necessarily the true compliance objective. A GP trained on weaker data can confidently select less useful designs.

So, cannot make conclusion yet on the way to balance the global and local restarts. Next steps:
- A run with the correct starting seed, and with a different local/global restarts
- Look into the maths behind the local/global restarts and relate it to the way the code is written. Check in the context to see conclusions on what values they should hold.
- Run further tests for triangle (once satisfied with the local/global restarts) at different seeds. Want to see little deviation in the result.

Remember, perfection is the enemy to progress!


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      15/10/26 at 15:37 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Structured run persistence and GP replay

The observability work package from `docs/261014_Modifications_To_BO_Code.pdf`
has been implemented without changing the GP noise model, acquisition policy,
restart allocation, geometric constraints, FEM calculation, or console summary.
Every physical evaluation is now recorded once in a lossless ordered tensor
archive, with a compact CSV index. Burn-in and BO numbering are explicit, and
the archive includes encoded designs, compliance, negative-log-compliance GP
responses, feasibility, raster fractions, geometry signatures, and candidate
sources. Logging reuses values already produced by the optimisation; it does not
add FEM evaluations.

Each GP fit references the exact prefix of that archive used as training data.
Fit snapshots store the fitted model and outcome-transform states,
hyperparameters, likelihood observation noise, kernel output scales, RNG states,
warnings and failures. Run manifests add stable IDs, resolved configuration,
software/CUDA versions, and git/source revision information. Raw compliance
variance and transformed-response variance are explicitly distinguished from
likelihood observation noise.

Public read-only loaders reconstruct the exact training X, physical compliance,
and pre-standardisation GP response before or after a selected BO iteration.
Atomic replacement protects the previous valid JSON, CSV, and tensor snapshots
from partially written updates. Targeted archive, replay, indexing, RNG,
warning/failure, atomic-write, GP and one-iteration driver tests passed; the
driver test confirms that persistence does not increase the number of physical
evaluator calls.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      16/10/26 at 09:20 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Readable designs and GP warm-start provenance

The structured records now repeat each encoded design vector as a compact JSON
list in `evaluations.csv`, alongside a checksum tied to the lossless tensor row.
This makes designs directly inspectable without changing
`evaluation_archive.pt` as the source of truth.

GP fit records now label each initialization as cold or warm and identify the
source fit for a warm start. They separately report the inherited mean-module
state, kernel lengthscales, output scales and likelihood noise, and the post-fit
lengthscales and output scales. The full inherited state remains losslessly
available through a reference to the preceding fit snapshot. Outcome
standardisation is not labelled as inherited because it is recalculated from
the current training responses. These are observability-only changes; the
existing warm-start load and fitting sequence are unchanged.
The run-record schema version was advanced from 1 to 2 so readers can
distinguish these expanded records from earlier manual-test outputs.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      16/10/26 at 10:01 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

GP health monitoring and response policy

The GP will support either the existing learned homoskedastic likelihood noise
or a user-prescribed positive constant variance in pre-standardisation
negative-log-compliance units. `None` retains the existing learned-noise model.
For the mixed composite kernel, signal variance is defined as the median prior
kernel diagonal over feasible restart-pool designs. This is compared with the
likelihood noise on the GP's standardised scale to report SNR and signal
fraction. Learned-noise runs initially warn at signal fraction below 0.01.

Additional scale-normalised checks cover posterior-mean spread, posterior
standard deviation and rolling one-step-ahead predictive residuals. They do not
receive invented numerical defaults: each remains report-only until its user
threshold is supplied. The two posterior ratios use, respectively, the standard
deviation of posterior means and the 90th percentile of latent posterior
standard deviations, divided by training-response standard deviation.
Predictive agreement uses rolling RMS standardized residuals. Monitoring reuses
feasible acquisition pools and the next BO evaluation, so it adds neither FEM
evaluations nor optimizer-RNG draws.

The initially implemented response policy made learned-noise warnings force one
cold fit at the next BO iteration, while prescribed-noise warnings were
report-only. This default was subsequently superseded by the decision recorded
below. Prescribed runs rebuild their fixed noise vector for the current training
set rather than inheriting it; their mean and kernel state can be warm-started.
Detailed metrics, reasons and actions are saved with each GP fit, while the
console receives one concise warning.
This addition advances the run-record schema to version 3.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      16/10/26 at 10:47 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Default GP warm-start policy

Omitting `warm_start_policy` (or setting it to `None`) now resolves to
`"always"` for both learned and prescribed observation noise. Health warnings
remain visible in the console and persistent fit records but do not alter GP
initialisation under this default. The first fit remains necessarily cold
because no preceding model exists; later fits inherit the available preceding
model state. Users who want health warnings to force one cold fit at the next
iteration can explicitly select `warm_start_policy="health_reset"` with either
noise mode.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      16/10/26 at 10:47 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Post-run geometry plots and configurable restart ranking

Periodic geometry rendering will be removed from the optimisation loop. A
post-run command will instead reconstruct the best design available through a
requested burn-in index or BO iteration from the persistent evaluation record
and save the same rasterised geometry without rerunning FEM. The lossless tensor
archive remains the design source of truth, while `evaluations.csv` supplies
readable indexing and objective metadata. Final best-geometry and optimisation-
diagnostic plots remain part of the normal run output.

Global restart ranking will expose its acquisition and novelty weights through
`OptimizationConfig`, retaining the current resolved fractions of 0.55 and 0.45
by default. Finite, non-negative inputs with a positive total will be normalised
to fractions before use. Local starts remain ranked by acquisition alone, so
this change does not alter their established policy, the acquisition function,
the GP, feasibility constraints, or the physical model.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      16/10/26 at 11:44 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Best-so-far and exact-evaluation geometry inspection

The post-run best-so-far function is named `plot_best_geometry` to make its
selection semantics explicit. A separate `plot_evaluated_geometry` function
selects one exact, one-based evaluation record and reports its encoded design
vector as well as rendering it. The exact-evaluation path decodes the readable
CSV vector and verifies it against both its recorded checksum and the lossless
archive row before plotting. Both functions reuse recorded objectives and
raster fractions and do not rerun FEM.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      16/10/26 at 11:56 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Consistent geometry evaluation numbering

All post-run Python plotting functions now accept `eval_number`. BO and burn-in
records have separate best-so-far and exact-evaluation functions, so the phase
is explicit in the function name and a global archive-row number is never
mistaken for a BO iteration. The BO functions match `bo_iteration`; the burn-in
functions match `burn_in_index`. Exact-evaluation functions continue to verify
the CSV design vector against its checksum and lossless archive row.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      16/10/26 at 14:47 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Independent numerical observation-noise study

The offline numerical-noise study must not inherit GP hyperparameters between
archived snapshots, noise candidates or initializations. Each study fit now
constructs the same unfitted `MixedSingleTaskGP`, Matérn-1.5 kernel, priors and
outcome transform as production, applies one reproducible controlled cold
initialization, and performs a single fitting attempt. Three paired initial
states can therefore reveal initialization fragility without silently allowing
BoTorch retry initialization to replace the assigned state.

This does not change ordinary Bayesian optimization. The production wrapper
still applies `_warm_start` after GP construction: the first BO fit is cold,
later fits inherit the immediately preceding fit, and an explicit
`warm_start_policy="health_reset"` health warning suppresses inheritance for
the next fit only. The default `"always"` policy continues to report warnings
without changing warm-starting.

The archive-only study compares learned noise with fixed standardized
variances, by default 1e-6, 1e-5 and 1e-4, over five seeds, four snapshots and
three initializations. Requested standardized fixed variance is converted back
to pre-standardisation negative-log-compliance units using the snapshot's
sample response variance. Records distinguish requested and effective noise,
and include initialization checksums, warnings/failures, fitted
hyperparameters, representative-probe behavior and held-out predictive
metrics. Post-processing aggregates initialization behavior within each
seed/snapshot and variation across seed/snapshot groups. The decision function
can decline to select; otherwise it requires stability, held-out agreement and
useful posterior variation relative to learned noise before applying a paired
NLPD one-standard-error rule. The selected variance remains provisional rather
than being interpreted as physical FEM observation noise.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      16/10/26 at 16:39 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Offline GP health-threshold calibration

Once the numerical-noise study selects a provisional fixed standardized
variance, a separate archive-only replay calibrates health warnings without
additional FEM evaluations. Every source run is replayed one BO prefix at a
time using the production sequence of one initial cold fit followed by warm
fits. Each warm fit is compared with an independently cold-fitted reference,
and the next archived response supplies the one-step-ahead predictive result.

The selected standardized noise is converted to raw negative-log-compliance
variance separately for every training prefix. This exactly holds the studied
standardized value constant, but it is not equivalent to the current runtime
configuration's one constant raw variance. This limitation is recorded in the
study outputs rather than silently treating the units as interchangeable.

Provisional lower thresholds for signal variance, posterior-mean spread and
posterior standard deviation, together with an upper rolling predictive-
residual threshold and window, are chosen with leave-one-seed-out validation.
The independent adverse label is a failed fit, fitting warning, or absolute
one-step standardized residual above 2.576 by default. Candidate thresholds
must respect a configurable maximum false-positive rate and demonstrate
positive held-out discrimination; otherwise the recommendation remains null.
For prescribed noise, `health_nu_min` is correctly marked not applicable
because current runtime health logic uses `health_signal_variance_min` instead.

The replay uses one common deterministic probe comprising deduplicated feasible
archive designs. This supports seed-to-seed comparison but does not exactly
reproduce historical iteration-specific restart pools. Recommendations are
therefore written to `recommended_health_config.json` for review and never
alter `OptimizationConfig` defaults automatically.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      16/10/26 at 18:23 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Greedy global restart selection with dynamic novelty

Global restart selection now defaults to a greedy weighted acquisition/novelty
rule. Acquisition values are evaluated once for an eligible raw pool. At each
selection step, novelty is recomputed as normalized Euclidean distance to the
union of observations, previously attempted starts and global starts already
selected. Acquisition and current novelty are separately min-max normalized
over the remaining eligible global candidates, then combined with the configured
non-negative weights. The chosen start enters the reference set before the next
score calculation. Dynamic minimum-distance filtering can therefore remove a
candidate that becomes redundant after an earlier selection.

The existing `static_rank` path is retained as an explicit experimental
comparison and continues to use its former one-pass scoring and subsequent
diverse-batch selection. Local restart ranking remains acquisition-only. Greedy
global ordering takes precedence over the batch selector, including for mixed
shape tuples; reserve candidates are greedily ordered to preserve optimizer-
failure fallback batches.

Run records now include one row per optimized restart in
`restart_attempts.csv`, retaining start scores, optimized acquisition values,
optimizer runtime, failures and duplicate endpoint classifications. Iteration
diagnostics include normalized global-start pairwise diversity, reference
distance, ranking and optimization runtime, duplicate counts and compliance.
This is a sampling/acquisition-initialization change only: the GP, acquisition
function, feasibility constraints, physical model and FEM evaluation count are
unchanged.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      17/10/26 at 15:23 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Nested global/archive restart-budget study

The global selector and batch fallback behavior were deliberately left
unchanged. Global novelty is recomputed against observations, supplied attempted
starts and earlier global choices in its greedy ordering; local candidates are
not members of that global-choice set. The dynamic selector rescales both raw
LogEI and current novelty to [0, 1] over the remaining eligible global candidates
at every round. If one requested source is insufficient, `take_restart_batch`
may fill the batch from the other source. This cross-source substitution is
intentional for production robustness. Ordinary run diagnostics now distinguish
the planned allocation from actual attempts, source shortfalls and substitutions
so that this fallback is no longer hidden.

`adaptive_restart_allocation=False` was added to hold `ng_rst` and `nl_rst`
fixed when a controlled production run requires it. The default remains `True`,
preserving the stagnation controller's existing shift toward global restarts.

The new archive-only `restart_budget_study` reads the provisional fixed
standardized variance from the numerical-noise study's `decision.json` and
restores the matching saved cold-fit GP state for each seed/snapshot. It always
uses archive starts for the local source. Global and archive starts remain only
initial conditions: every acquisition optimization uses the same full box and
constraints, and no radius/trust region is imposed. No FEM quantity is
recomputed.

For every seed/snapshot, the study generates one reproducible maximum global
pool and one deterministic maximum archive pool, applies the current ranking
policy once, and stores their ordered sequences. Each retained start is
optimized exactly once with a frozen GP, fixed LogEI and common optimizer
settings. Every `(ng_rst, nl_rst)` comparison is then reconstructed from the
first `ng_rst` global and first `nl_rst` archive entries. The study does not use
production cross-source substitution: an unavailable prefix is recorded as an
explicit source shortfall. This separates the controlled experiment from the
intentionally resilient production fallback.

Lossless candidate pools and optimized endpoints are atomically checkpointed;
CSV summaries record requested and actual allocations, failures, feasibility,
endpoint duplicates, diversity, runtime, winning source and best feasible
LogEI. Final compliance is marked not applicable because a frozen-acquisition
comparison cannot supply counterfactual FEM outcomes. The provisional decision
uses paired seed/snapshot LogEI regret and selects the smallest budget whose
one-standard-error interval overlaps the best observed allocation. It never
changes runtime defaults and requires later sequential multi-seed BO
confirmation.

For the proposed recipes, the short five-seed/two-snapshot design optimizes at
most 200 unique starts, whereas the full five-seed/four-snapshot design with up
to 30 starts from each source optimizes at most 1,200. The acquisition-
optimization workload is therefore approximately six times larger for the full
study. With the 60-second per-start timeout, conservative upper bounds are about
3.3 hours and 20 hours respectively, although normal completion should be
faster. The short scientific study is not required before the full study; a
one-seed, one-snapshot smoke run is the more efficient way to validate paths,
device setup and checkpointing before starting the resumable full study.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      19/10/26 at 08:46 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Global acquisition/novelty-weight study

The acquisition and novelty configuration values affect global restart
selection only, and only their normalized ratio is identifiable. The manual
`global_restart_weight_study` therefore parameterizes the comparison by
acquisition fraction `alpha`, with novelty fraction `1 - alpha`, while leaving
the production selector unchanged. Its controlled frozen-GP stage uses no local
starts, one shared raw global pool per seed/snapshot/budget, the same fixed
LogEI and constraints, and the current greedy roundwise score normalization.
Novelty is recomputed after every selected global start exactly as in
production.

Every unique start selected by any tested weight is optimized once and
reused across comparisons. The primary offline result is paired feasible-LogEI
regret against the best endpoint in that shared union. Diversity, endpoint
duplicates, failures, shortfalls and runtime remain diagnostics rather than
surrogate measures of final compliance. Snapshot results are aggregated
within seed before inference because snapshots from one BO trajectory are not
independent.

Frozen acquisition results are screening evidence only. The study nominates at
most two challengers and never changes runtime defaults. A proposed default
must be confirmed on independent seeds by sequential BO runs in which the
weights are the only changed configuration. A global-only arm isolates the
selector mechanism; an optional fixed mixed global/archive arm tests whether
the effect survives the intended production allocation. Final compliance is
the primary sequential outcome. The sequential archive analyzer verifies exact
paired burn-in data, equal non-weight configuration, source provenance and BO
evaluation counts. Its decision procedure retains the current
0.55/0.45 setting or returns insufficient evidence when a challenger is not
clearly better under a predeclared practical margin. Results obtained solely
from the existing triangle archives cannot justify a package-wide default for
ellipses or other geometric regimes.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      19/10/26 at 13:59 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Standardized prescribed observation noise

The numerical-noise study selects fixed variance on the standardized
GP-output scale, whereas `OptimizationConfig.observation_noise_variance`
previously passed its value directly in raw pre-standardization
`-log(compliance)` units. Applying the study's value directly therefore gave a
different effective noise level as the training-response variance changed.

`observation_noise_variance` now denotes standardized GP-output variance. At
each production GP fit it is multiplied by that fit's training-response sample
variance before constructing the fixed-noise model. `None` continues to select
the learned homoskedastic model, and the lower-level GP factory retains its raw
response-unit interface for controlled offline studies. Persistent fit records
store both the requested standardized value and the converted raw value so the
conversion can be audited exactly.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      21/10/26 at 10:10 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

WP6 ellipse symmetry-invariant representation

Ellipse designs have exact parameter symmetries: rotating by pi, or swapping
the semi-axes while rotating by pi/2, describes the same hole; a circle is
independent of angle. Raw ellipse coordinates expose these equivalent designs
as different GP inputs and different novelty distances. The approved WP6
change adds the opt-in `ellipse_feature_mode="shape_matrix"`. It maps each
ellipse to its normalized physical shape matrix, takes the stable analytic
2-by-2 matrix logarithm, and uses `[shape, cx, cy, L11, sqrt(2)L12, L22, 0]`
for GP and restart-distance calculations. The analytic circular limit avoids
eigendecomposition ambiguity and preserves acquisition gradients near circles.

Raw encoded vectors remain the acquisition variables and the sole inputs to
geometry, feasibility, rasterisation, clearance and FEM calculations. The
existing `"raw"` representation remains the default so a paired raw/invariant
ellipse study can be performed before considering a default change. Triangle
runs are required to use `"raw"`; they do not construct an input transform and
retain the previous numerical path. WP7 clearance/feasibility changes were not
approved and were not included.

Invariant fit records persist the feature schema, domain dimensions and exact
transformed training tensor. Offline numerical-noise, health, restart-budget
and global-weight reconstruction now propagate and validate the archived
feature mode, preventing a saved invariant GP state from being replayed under
raw coordinates (or vice versa).

Validation completed with 78 repository tests passing. Focused checks cover
the three ellipse symmetries, finite feature and LogEI gradients at the circle
limit, invariant novelty distance, fit-state round trips and triangle-path
isolation. A direct construction check also confirmed that the default triangle
GP retains identical training inputs, state-dictionary keys and initial state
values to the pre-WP6 factory definition.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                                      21/10/26 at 12:02 BST
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Asymmetric horizontal margins and isolated burn-in sampling

`OptimizationConfig` now accepts `bx_left` and `bx_right`; either omitted side
falls back to the legacy `bx`, preserving existing symmetric configurations.
Containment, normalized anchor limits and sampler usable-area calculations use
the resolved interval `[bx_left, xsize - bx_right]`. Archived study
reconstruction reads the side-specific values when present and otherwise uses
legacy `bx`.

Larger margins and positive inter-hole clearance can make the initial design
packing extremely sparse or impossible. A burn-in preflight now checks usable
extent, required void area, per-shape area limits and a necessary convex-body
clearance-area bound before expensive rejection sampling. These conditions are
necessary rather than sufficient. The target-area interval begins at the
larger of the solid-fraction requirement and `nholes * amin`, avoiding proposals
that cannot satisfy the minimum hole areas.

Initial GP designs now use the separate `sample_burn_in_feasible` path. It keeps
the exact near-equal/broad quotas, performs bounded joint-placement retries and
has clearance-aware constructive fallbacks for all-triangle and all-ellipse
burn-ins. Returned designs are still filtered by the common analytical and
raster feasibility functions. BO iteration/global restart sampling deliberately
continues to use the pre-existing `sample_stratified_feasible` and
`random_shape_candidate` path, so the new burn-in placement policy does not
change evaluation-iteration sampling. `burn_in_sampling.json` records the
preflight quantities, attempts and rejection causes.

The required seed-0 acceptance gate used otherwise-default geometry with three
holes, `smin=5`, `bx_left=5` and `bx_right=2.5`. The sampler collected the full
105-point burn-in for triangles in 56.43 seconds and for ellipses in 10.73
seconds. Each run filled 53 near-equal and 52 broad-area designs, and every
returned design passed analytical and raster feasibility. The approved rollback
condition was therefore not triggered. Twenty-six targeted tests and the full
82-test repository suite passed; the only warning was the existing upstream
PyTorch `torch.jit.script` deprecation.
