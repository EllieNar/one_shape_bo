# Bayesian optimisation for shaped beam perforations

`one_shape_bo` minimizes finite-element compliance for an MBB beam containing
all-triangular (`htype=0`) or all-elliptic (`htype=1`) perforations. It uses a
Gaussian-process surrogate, logarithmic expected improvement, and constrained
mixed-variable acquisition optimization. Geometric containment, minimum area,
shape quality, inter-hole spacing, analytical volume, and rasterized volume are
enforced.

Feasible burn-in samples are split between a near-equal hole-area policy and a
broad area-allocation policy. BO global pools use the broad policy. The local
restart policy defaults to radius-based feasible perturbations around several
good observations. The opt-in `archive` policy instead uses the incumbent and
GP-kernel-diverse members of the feasible observation archive as local
acquisition-optimizer starts; these observed starts are not physically evaluated
again, and optimized endpoints must remain feasible and novel. After a
configurable period without a record improvement, a fixed restart budget shifts
toward global starts; an improvement restores the baseline global/local split.
Set `adaptive_restart_allocation=False` to keep the configured `ng_rst` and
`nl_rst` split fixed despite stagnation.

## Development environment

The supported Python version is 3.12. Create the development environment with:

```text
conda env create -f environment.yml
conda activate one-shape-bo
```

Install the package in that environment and run the tests with:

```text
python -m pip install -e .
pytest
```

## Running

The default three-hole triangle problem can be run with:

```text
python -m one_shape_bo --seed 0
```

Use `--shape ellipse`, `--burn-in`, `--iterations`,
`--local-start-strategy archive`, or `--output-dir` to override the common
command-line settings. For full control, construct `OptimizationConfig` and call
`run_optimization`:

```python
from one_shape_bo import OptimizationConfig, run_optimization

result = run_optimization(
    OptimizationConfig(
        seed=7,
        ng_rst=10,
        nl_rst=5,
        local_start_strategy="archive",
    )
)
```

`local_start_strategy="radius"` remains the default. The archive strategy is
experimental pending a separately run multi-seed comparison; selecting it does
not alter the GP, LogEI acquisition policy, or physical model, and does not add
extra physical evaluations to a successful BO iteration.

Unless `output_dir` is supplied, run artifacts are written below a dated,
time-stamped `outputs/YYMMDD_HHMMSS_<Shape>/` directory. Each run records its
configuration, a compact output log, per-iteration GP and restart diagnostics in
CSV, a diagnostic plot, the best geometry, and (for the standard triangle case)
a read-only audit of the stored reference PNG.

Each run also has a stable ID and a replayable, versioned record:

- `evaluation_archive.pt` stores every encoded burn-in and BO design, physical
  compliance, negative-log-compliance GP response, feasibility flags, raster
  fraction, geometry signature, source, and explicit burn-in/BO indices.
- `evaluations.csv` is a human-readable index into that tensor archive. It also
  includes each complete encoded design vector as a JSON list and a checksum of
  the corresponding lossless tensor row.
- `burn_in_summary.json` records the actual burn-in range, best and worst designs,
  and raw/transformed sample variances.
- `gp_fits.json`, `gp_fits.csv`, and `fit_states/` record the exact archive prefix
  used by each fit, fitted state and hyperparameters, outcome scaling, likelihood
  observation noise, RNG states, and fitting warnings or failures. Fit summaries
  label cold versus warm initialization and distinguish inherited lengthscales,
  output scales, noise and mean state from the corresponding post-fit values.
  They also report GP signal/noise health metrics, threshold reasons and any
  requested cold-fit action.
- `run_manifest.json` records resolved settings, software/CUDA details, git/source
  revision information, lifecycle status, and references to the other records.

The tensor files are lossless local run artifacts and should only be loaded from
trusted runs. Reconstruct the GP inputs and responses around a chosen iteration
without rerunning FEM with:

```python
from one_shape_bo import reconstruct_training_set

before = reconstruct_training_set(run_directory, 11, position="before")
after = reconstruct_training_set(run_directory, 11, position="after")
```

Here `physical_compliance` is the FEM output and
`gp_response_prestandardization` is `-log(compliance)`. The recorded raw and
transformed sample variances describe variation across evaluated designs; they
are not observation-noise estimates. Likelihood observation noise and kernel
output scale are labelled separately in the fit records.

The CSV reports requested and generated local starts, acquisition improvement,
start-to-end movement, optimizer failures, and posterior uncertainty at the
selected candidate, current incumbent, and fixed best burn-in design. Posterior
standard deviation is in the GP's negative-log-compliance score units, not
standardized units or compliance units.

Periodic geometry plotting is intentionally kept out of the optimisation loop.
The former `plot_every` configuration field has therefore been removed.
After a run, recreate the best rasterised design available through a selected
BO or burn-in evaluation without rerunning FEM:

```bash
python -m one_shape_bo.plot_evaluation outputs/MY_RUN --best-eval 20
python -m one_shape_bo.plot_evaluation outputs/MY_RUN --best-burn-in 10
```

The commands save `Best_Geometry_eval_0020.png` and
`Best_Burn_In_Geometry_eval_0010.png`, respectively, in the run directory and print
the selected evaluation metadata. Use `--output path/to/name.png` to choose a
different destination and `--show` to display the figure when a graphical
backend is available. The design comes from the lossless archive; the command
uses `evaluations.csv` to identify the requested step and best-so-far record.
The earlier best-so-far spellings remain accepted as command-line aliases.

To plot an exact BO or burn-in evaluation rather than the best design available
by that step, run:

```bash
python -m one_shape_bo.plot_evaluation outputs/MY_RUN --eval 20
python -m one_shape_bo.plot_evaluation outputs/MY_RUN --burn-in 10
```

These save `Geometry_eval_0020.png` and `Burn_In_Geometry_eval_0010.png` and
also print the complete encoded design vector. The equivalent Python interfaces
are:

```python
from one_shape_bo.plot_evaluation import (
    plot_best_burn_in_geometry,
    plot_best_geometry,
    plot_evaluated_burn_in_geometry,
    plot_evaluated_geometry,
)

best = plot_best_geometry("outputs/MY_RUN", eval_number=20)
exact = plot_evaluated_geometry("outputs/MY_RUN", eval_number=20)
burn_in_best = plot_best_burn_in_geometry("outputs/MY_RUN", eval_number=10)
burn_in_exact = plot_evaluated_burn_in_geometry(
    "outputs/MY_RUN", eval_number=10
)
print(exact["encoded_design_vector"])
```

`eval_number` is phase-specific: BO functions match the recorded BO iteration,
while burn-in functions match the burn-in index. Both exact-evaluation functions
verify the CSV vector against its stored checksum and lossless archive row.

Global restart starts are ranked using normalized acquisition and novelty
scores. The default `greedy_dynamic_novelty` strategy chooses global starts one
at a time. After each choice, novelty is recomputed as normalized Euclidean
distance from all observations, previously attempted starts and global starts
already selected for that batch. Acquisition and novelty are min-max scaled to
`[0, 1]` over the currently eligible candidates before applying their weights.
Their default fractions are 0.55 and 0.45 and can be configured as:

```python
config = OptimizationConfig(
    acquisition_weight=0.55,
    novelty_weight=0.45,
    global_restart_selection_strategy="greedy_dynamic_novelty",
)
```

Any finite, non-negative pair with a positive total is accepted and normalized
internally. Local starts continue to be ranked by acquisition alone. The former
one-pass ranking and separate diverse-batch selector remain available for
controlled comparison with:

```python
config = OptimizationConfig(
    global_restart_selection_strategy="static_rank",
)
```

`restart_attempts.csv` records every optimized start, its selection scores,
optimized acquisition value, runtime, failure status and duplicate-endpoint
classification. `iteration_diagnostics.csv` additionally reports global-start
pairwise diversity, ranking/optimization runtime, duplicate counts and best
compliance. These diagnostics add no physical evaluations.

## GP noise and health monitoring

By default, the GP retains its learned homoskedastic observation noise. Its
signal fraction is monitored with an initial warning threshold of `0.01`; a
warning is reported but does not change warm-starting. A deterministic FEM
noise variance can instead be prescribed in pre-standardisation
negative-log-compliance units:

```python
config = OptimizationConfig(
    observation_noise_variance=1e-8,
    health_signal_variance_min=1e-4,
)
```

When `warm_start_policy` is omitted or set to `None`, both learned-noise and
prescribed-noise runs always warm-start after the unavoidable first cold fit;
health warnings are report-only. Set `warm_start_policy="health_reset"` to make
any enabled health warning force one cold fit at the next BO iteration. The
appropriate positive prescribed variance and near-zero signal threshold should
be obtained from a numerical-stability study; the example values above are
illustrative, not recommended defaults. Optional posterior-mean,
posterior-standard-deviation and rolling predictive-residual thresholds remain
disabled until explicitly configured.

Signal and learned-noise variances are compared on the GP-standardised scale;
signal variance is the median kernel diagonal over the feasible restart pool.
The posterior checks divide the standard deviation of posterior means and the
90th percentile of latent posterior standard deviations by the training-response
standard deviation. Predictive agreement uses the rolling RMS of standardized
one-step-ahead residuals at evaluated BO candidates.

### Offline observation-noise study

`one_shape_bo.numerical_noise_study` compares learned noise with provisional
fixed standardized variances using existing persistent run archives. It does
not run FEM or generate new BO candidates. Every study fit is independently
cold-started from one of three controlled initializations; ordinary
`run_optimization` behavior is unchanged and continues to warm-start according
to `warm_start_policy`.

Prepare five learned-noise archive runs with seeds 0, 1, 2, 3 and 76. Each must
contain the burn-in and at least 30 BO evaluations so the default snapshots
after BO evaluations 0, 5, 10 and 20 each have ten subsequent held-out points.
Then run:

```bash
python -m one_shape_bo.numerical_noise_study run \
  --run 0=outputs/noise_study/learned_seed_0 \
  --run 1=outputs/noise_study/learned_seed_1 \
  --run 2=outputs/noise_study/learned_seed_2 \
  --run 3=outputs/noise_study/learned_seed_3 \
  --run 76=outputs/noise_study/learned_seed_76 \
  --standardized-variance 1e-6 1e-5 1e-4 \
  --fit-after 0 5 10 20 \
  --holdout-size 10 \
  --initializations 3 \
  --output-dir outputs/numerical_noise_study/five_seed_study
```

For every snapshot, a requested standardized fixed variance is converted to
the configuration/model response scale as
`requested variance * sample variance(-log(compliance))`. The output records
both that requested value and the effective fixed likelihood variance, exposing
any GPyTorch fixed-noise floor. The learned-noise model is fitted to exactly the
same archived training prefixes and remains the control.

The run is resumable and atomically updates
`gp_fits_numerical_study.csv`. Lossless fitted states are stored in
`fit_states/`; the CSV records initialization provenance, warnings/failures,
fitted hyperparameters, effective noise, probe mean/latent-uncertainty
variation, and held-out RMSE, residual, coverage and predictive-density
metrics. Summarize and cast the explicitly provisional decision with:

```bash
python -m one_shape_bo.numerical_noise_study summarize \
  outputs/numerical_noise_study/five_seed_study
python -m one_shape_bo.numerical_noise_study decide \
  outputs/numerical_noise_study/five_seed_study
```

The decision function first requires fit stability, held-out accuracy and
non-collapsed posterior variation relative to the learned control. It then uses
paired held-out negative log predictive density and a one-standard-error set;
warning rate and sensitivity to initialization precede larger fixed variance
as tie-breakers. It writes `decision.json` and returns `no_selection` rather
than forcing a choice when no fixed variance passes. These are transparent
provisional screening rules, not a claim that deterministic FEM has physical
measurement noise.

### Offline health-threshold study

After `decision.json` contains a provisional fixed standardized variance,
`one_shape_bo.health_monitoring_study` replays every archived BO prefix without
rerunning FEM. It uses the production initialization sequence—first fit cold,
then warm from the preceding fit—and independently cold-fits the same data as a
stability reference. The next archived evaluation supplies a genuine
one-step-ahead predictive check.

```bash
python -m one_shape_bo.health_monitoring_study run \
  outputs/numerical_noise_study/five_seed_study \
  --output-dir outputs/health_monitoring_study/five_seed_study \
  --window 3 5 10
python -m one_shape_bo.health_monitoring_study summarize \
  outputs/health_monitoring_study/five_seed_study
python -m one_shape_bo.health_monitoring_study calibrate \
  outputs/health_monitoring_study/five_seed_study
```

If `decision.json` is absent, the run command first invokes the numerical
study's decision function. It stops if that decision is `no_selection`.
Alternatively, `--selected-standardized-variance 1e-6` can explicitly supply a
reviewed value. At every replay fit, the standardized value is converted to
pre-standardisation response units using that prefix's response sample
variance.

`gp_health_replay.csv` records signal/noise measures, posterior behavior,
one-step residuals, warm/cold disagreement, warnings and fitted
hyperparameters. Calibration uses leave-one-seed-out validation and limits the
false-warning rate while testing lower signal, mean-spread and posterior-SD
thresholds and several rolling-residual windows. It writes
`recommended_health_config.json`, but never changes runtime defaults. A value
is left `null` when cross-seed evidence is insufficient. `health_nu_min` is
reported as not applicable because prescribed-noise runtime monitoring uses
signal variance rather than the learned-noise signal fraction.

The recommendations are provisional for two explicit reasons: the replay's
common probes are the deduplicated feasible archive rather than each historical
restart pool, and constant standardized noise requires a different raw
variance after response standardisation at each fit. Review those assumptions
before transferring the values into `OptimizationConfig`.

### Offline restart-budget study

`one_shape_bo.restart_budget_study` compares global/archive restart budgets
without running FEM. It reads the provisional fixed standardized variance from
the numerical-noise study's `decision.json`, restores the corresponding saved
GP at each archived snapshot, and freezes that GP and its LogEI acquisition.
For each seed/snapshot it saves one shared global pool and one deterministic
archive-start pool. Every retained start is optimized once; allocation results
then use nested global and archive prefixes, so a shared start has identical
optimizer settings and output in every comparison.

The study always treats the local source as `archive`. Archive starts and global
starts are initial conditions for optimization over the full acquisition box;
`local_radius` is not used. It also disables adaptive allocation and source
substitution inside the study. Ordinary BO behavior is unchanged: if one source
is short, `take_restart_batch` may fill the batch from the other source.

```python
from pathlib import Path

from one_shape_bo.restart_budget_study import (
    recommend_restart_allocation,
    run_restart_budget_study,
    summarize_restart_budget_study,
)

source = Path("outputs/manual_tests/M1_Numerical_Noise_Study")
runs = {seed: source / f"Seed_{seed}" for seed in (0, 15, 23, 46, 53)}
output = Path("outputs/manual_tests/M4_Restart_Budget_Study/full")

rows = run_restart_budget_study(
    run_directories=runs,
    numerical_study_directory=source / "numerical_noise_study",
    output_directory=output,
    fit_after=(0, 5, 10, 20),
    allocations=(
        (0, 5), (3, 2), (5, 0),
        (0, 10), (5, 5), (7, 3), (10, 0),
        (0, 15), (8, 7), (10, 5), (15, 0),
        (0, 20), (10, 10), (20, 0),
        (0, 30), (15, 15), (20, 10), (30, 0),
    ),
)
summary = summarize_restart_budget_study(output)
recommendation = recommend_restart_allocation(output)
print(recommendation)
```

The run is atomically checkpointed after every optimized start. The primary
metric is the greatest LogEI among unique, analytically and raster-feasible,
non-observed endpoints. The recommendation uses paired seed/snapshot LogEI
regret and a one-standard-error rule, but remains provisional until candidate
settings are confirmed with sequential multi-seed BO runs. `final_compliance`
is explicitly not applicable because the frozen study does not evaluate FEM.

The expected triangle image comes from a filtered, continuous-density topology
optimizer. Its displayed thresholded geometry does not preserve the optimizer's
continuous volume or objective, so it is treated as a qualitative topology
target rather than an equivalent numerical optimum for the binary three-hole
problem.
