# Abaqus File Creator

Python package for finding the most optimal arrangement of holes.
The objective function is to minimise compliance. The holes can either be
all triangular or all elliptic. Unconstrained bayesian optimisation is used.

A feasibility-driven DOE is used to randomly select samples, rather than a
traditional space-filling DOE (i.e. Latin Hypercube or Sobol sampling in the
design space). This is because various geometric properties (e.g. separating axis
theorem, domain limits, spacing limits) would make traditional DOE difficult.

Although the files within the legacy folder provide a strong foundation to
the problem, issues are summaried in the ISSUES.md, and require immediate attention.

## Development environment

The supported Python version is 3.12. Create the development environment with:

```text
conda env create -f environment.yml
conda activate abaqus-file-creator
```

Run the foundation tests with:

```text
pytest
```
