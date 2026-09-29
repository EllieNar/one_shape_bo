# Bayesian optimisation for shaped beam perforations

`one_shape_bo` minimizes finite-element compliance for an MBB beam containing
all-triangular (`htype=0`) or all-elliptic (`htype=1`) perforations. It uses a
Gaussian-process surrogate, logarithmic expected improvement, and constrained
mixed-variable acquisition optimization. Geometric containment, minimum area,
shape quality, inter-hole spacing, analytical volume, and rasterized volume are
enforced.

Feasible burn-in samples are split between a near-equal hole-area policy and a
broad area-allocation policy. BO global pools use the broad policy. The local
sampler uses several diverse good observations as anchors. It starts at the
configured `local_radius`, then falls back to 0.5, 0.25, and 0.125 times that
radius if complete anchor sweeps do not fill the requested local pool. It stops
at the existing attempt limit and returns a partial pool when necessary. After a
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

Use `--shape ellipse`, `--burn-in`, `--iterations`, or `--output-dir` to
override the common command-line settings. For full control, construct
`OptimizationConfig` and call `run_optimization`:

```python
from one_shape_bo import OptimizationConfig, run_optimization

result = run_optimization(
    OptimizationConfig(seed=7, ng_rst=10, nl_rst=5, local_radius=0.05)
)
```

Unless `output_dir` is supplied, run artifacts are written below a dated,
time-stamped `outputs/YYMMDD_HHMMSS_<Shape>/` directory. Each run records its
configuration, a compact output log, per-iteration GP and restart diagnostics in
CSV, a diagnostic plot, the best geometry, and (for the standard triangle case)
a read-only audit of the stored reference PNG.

The CSV distinguishes requested and generated local-pool sizes and records the
attempt count and smallest local radius reached. Reported posterior standard
deviations are uncertainties in the GP's standardized negative-log-compliance
score space, not in compliance units. They are recorded for the selected
candidate, the incumbent at model-fitting time, and the fixed best burn-in design;
the latter two make iteration-to-iteration comparisons more meaningful.

The expected triangle image comes from a filtered, continuous-density topology
optimizer. Its displayed thresholded geometry does not preserve the optimizer's
continuous volume or objective, so it is treated as a qualitative topology
target rather than an equivalent numerical optimum for the binary three-hole
problem.
