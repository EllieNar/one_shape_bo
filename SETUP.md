# WSL setup and launch guide

This guide covers installation and launching `one_shape_bo` from WSL with either
CUDA or CPU execution. Run commands from a WSL terminal, not Command Prompt or
PowerShell.

The execution device must be chosen before Python imports `one_shape_bo`. The
package selects CUDA automatically when it is visible; changing environment
variables after importing the package is too late.

## 1. Open the repository and activate Conda

```bash
cd ~/projects/one_shape_bo
source ~/miniconda3/etc/profile.d/conda.sh
```

Create the environment once:

```bash
conda env create -f environment.yml
```

If Conda reports that `one-shape-bo` already exists, do not recreate it. Activate
the existing environment instead:

```bash
conda activate one-shape-bo
```

Install the repository as an editable package so that Python uses the source in
`src/one_shape_bo`:

```bash
python -m pip install -e .
python -c "import one_shape_bo; print(one_shape_bo.__file__)"
```

The printed path should end in:

```text
/projects/one_shape_bo/src/one_shape_bo/__init__.py
```

## 2. Run the targeted test suite on CPU

Some test fixtures contain explicitly CPU-resident tensors. Hide CUDA for the
test command so those fixtures are not mixed with CUDA package tensors:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest
```

This setting applies only to that command. It does not disable CUDA for a later
Python process.

## 3. CUDA setup and verification in WSL

First confirm that WSL can see the NVIDIA GPU:

```bash
nvidia-smi
```

With `one-shape-bo` activated, verify PyTorch CUDA support:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

If CUDA is available but Triton reports `fatal error: cuda.h: No such file or
directory`, install the tested CUDA development packages:

```bash
conda install -c conda-forge "cuda-driver-dev=13.4.92" "cuda-cudart-dev=13.4.92"
```

Verify the actual NVIDIA development header, rather than one of PyTorch's
internal files with the same name:

```bash
find "$CONDA_PREFIX/targets" -path "*/include/cuda.h" -print
```

The expected path is similar to:

```text
/home/eleno/miniconda3/envs/one-shape-bo/targets/x86_64-linux/include/cuda.h
```

Check CUDA differentiation before starting a long run:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(4, 4, device="cuda", dtype=torch.double, requires_grad=True)
loss = (x @ x).square().sum()
loss.backward()

print("CUDA differentiation successful:", x.grad is not None)
PY
```

A `_POSIX_C_SOURCE redefined` message while Triton compiles its helper is a
compiler warning, not a failure, if execution continues afterward.

## 4. Launch an optimization with CUDA

Start a fresh Python process normally. Do not set `CUDA_VISIBLE_DEVICES`:

```bash
cd ~/projects/one_shape_bo
conda activate one-shape-bo
python
```

At the Python prompt, verify and run:

```python
from one_shape_bo.runtime import DEVICE
print(DEVICE)  # expected: cuda

from one_shape_bo import OptimizationConfig, run_optimization

config = OptimizationConfig(
    seed=0,
    burn_in=105,
    iterations=525,
    output_dir="outputs/YYMMDD_CUDA_Seed0",
    audit_reference=False,
)

result = run_optimization(config)
print(result)
```

Replace `YYMMDD_CUDA_Seed0` with a unique output-directory name. Keep the terminal
open until the run finishes. Leave Python afterward with:

```python
exit()
```

## 5. Launch an optimization with CPU

CPU mode is required for the closest comparison with
`outputs/260928_Implemented_Corrections`. Exit any existing Python session, then
start a new process with CUDA hidden from the outset:

```bash
cd ~/projects/one_shape_bo
conda activate one-shape-bo
CUDA_VISIBLE_DEVICES="" python
```

At the Python prompt, verify and run:

```python
from one_shape_bo.runtime import DEVICE
print(DEVICE)  # expected: cpu

from one_shape_bo import OptimizationConfig, run_optimization

config = OptimizationConfig(
    seed=0,
    burn_in=105,
    iterations=525,
    output_dir="outputs/YYMMDD_CPU_Seed0",
    audit_reference=False,
)

result = run_optimization(config)
print(result)
```

The inline `CUDA_VISIBLE_DEVICES=""` affects only this Python process. A later
normally launched process can use CUDA again.

## 6. Supply only settings that differ from defaults

`OptimizationConfig` provides defaults for every field. Only overrides need to be
written. For example, this changes the seed, run length, and output location while
leaving every other setting at its package default:

```python
config = OptimizationConfig(
    seed=3,
    iterations=100,
    output_dir="outputs/YYMMDD_CPU_Seed3",
)
result = run_optimization(config)
```

The default burn-in is `5 * (7 * nholes)` and the default BO length is
`25 * (7 * nholes)` when `burn_in` and `iterations` are not supplied.

## Reproducibility warning

The same configuration and seed do not imply the same trajectory on CPU and
CUDA. Sampling uses a generator created on the selected tensor device, so the
burn-in population can differ immediately. Package versions can also affect GP
fitting and numerical optimization because `environment.yml` currently specifies
several dependencies without exact version pins.

For a controlled comparison:

1. use the same source revision;
2. use the same CPU/CUDA mode;
3. use the same package versions;
4. use the same configuration and seed; and
5. write each run to a new output directory.

## MO : Verification of reproducible outputs

cd ~/projects/one_shape_bo
conda activate one-shape-bo

test ! -e outputs/manual_tests/M0_Verify_Reproducible_Outputs \
  && echo "Output path is available" \
  || echo "STOP: choose a new output folder"

Run the test:
timeout 600s python - <<'PY'
from one_shape_bo import OptimizationConfig, run_optimization

config = OptimizationConfig(
    xsize=40,
    ysize=20,
    nholes=1,
    htype=0,
    amin=20.0,
    burn_in=8,
    iterations=4,
    ng_rst=2,
    nl_rst=1,
    raw_pool_multiplier=2,
    max_restart_batches=2,
    seed=0,
    plot_every=0,
    audit_reference=False,
    output_dir="outputs/manual_tests/M0_Verify_Reproducible_Outputs",
)

result = run_optimization(config)

print("\nMANUAL RUN COMPLETED")
print("best compliance:", result["best_compliance"])
print("best source:", result["best_source"])
print("feasible:", result["feasible"])
print("output directory:", result["output_dir"])
PY

Expected workload:
- 8 burn-in FEM evaluations
- Up to 4 BO FEM evaluations
- 4 GP fits
- Maximum wall time: 600 seconds
- Expected successful shell exit code: 0
- Timeout exit code: 124

During the run, supervise the console for four iter=... messages. An early “no novel feasible candidate” message is possible but should be noted.

Then, inspect the records:
find outputs/manual_tests/M0_Verify_Reproducible_Outputs \
  -maxdepth 2 -type f -printf '%P\n' | sort

python - <<'PY'
import json
from pathlib import Path

run = Path("outputs/manual_tests/M0_Verify_Reproducible_Outputs")
manifest = json.loads((run / "run_manifest.json").read_text())
fits = json.loads((run / "gp_fits.json").read_text())

print("status:", manifest["status"])
print("seed:", manifest["initial_seed"])
print("evaluations:", manifest["evaluation_count"])
print("GP fits:", manifest["fit_count"])
print("fit training counts:", [fit["train_count"] for fit in fits])
PY

For a complete run, expect:
status: completed
seed: 0
evaluations: 12
GP fits: 4
fit training counts: [8, 9, 10, 11]

Finally, verify exact reconstruction around BO iteration 2:

python - <<'PY'
from one_shape_bo import reconstruct_training_set

run = "outputs/manual_tests/M0_Verify_Reproducible_Outputs"
before = reconstruct_training_set(run, 2, position="before")
after = reconstruct_training_set(run, 2, position="after")

print("before iteration 2:", before["x"].shape, before["physical_compliance"].shape)
print("after iteration 2: ", after["x"].shape, after["physical_compliance"].shape)
print("previous rows unchanged:", (before["x"] == after["x"][:-1]).all().item())
PY

Expected result:

before iteration 2: torch.Size([9, 7]) torch.Size([9, 1])
after iteration 2:  torch.Size([10, 7]) torch.Size([10, 1])
previous rows unchanged: True