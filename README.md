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

The expected triangle image comes from a filtered, continuous-density topology
optimizer. Its displayed thresholded geometry does not preserve the optimizer's
continuous volume or objective, so it is treated as a qualitative topology
target rather than an equivalent numerical optimum for the binary three-hole
problem.
