# Preparing the Windows PTv3 build environment

The `point-transformer-v3` target is different from the CPU targets: it uses a
**separate Conda environment named `pointcept`**. `-Bootstrap` now creates it
automatically and installs the CUDA runtime from
[point-transformer-v3-runtime.txt](../scripts/requirements/point-transformer-v3-runtime.txt).
PyInstaller/setuptools remain isolated in `.build/build-tools/point-transformer-v3`.
There is no manual Pointcept download step. NVIDIA drivers and any required
system CUDA/MSVC toolchain are still machine prerequisites.

## Reference profile

These versions were read from the existing development environment on
16 October 2026; they are a legacy compatibility profile, not recommendations to
update another model's environment in place:

| Component | Version |
|---|---|
| Python | 3.8.19, Windows x64 |
| NumPy | 1.24.3 |
| PyTorch | 1.12.1, CUDA 11.3 build |
| spconv-cu113 | 2.3.6 |
| cumm-cu113 | 0.4.11 |
| torch-scatter | 2.0.9 |
| ninja | 1.11.1.1 |

Python 3.8 is retained for this historical native stack. Do not use that interpreter
for all editor tooling or replace the CPython 3.13 host with it. A newer GPU stack
requires its own model/kernel compatibility validation.

## Automatic setup and build

```powershell
python .\create_installer.py --build-executables --bootstrap
```

This prepares PTv3 along with the five CPU targets and packages the installer.
For PTv3 alone, run the dedicated wrapper shown below; a missing environment
automatically enables bootstrap. On an existing environment, use `--bootstrap`
to apply the pinned requirements again. Downloads use pip's cache; rebuilding
without bootstrap reuses the environment. Back up a custom `pointcept` environment
before explicitly applying this legacy pinned profile.

## Manual diagnostic recipe (optional)

Use a separate environment rather than changing a working one. The following
recipe uses published Windows wheels matching the reference profile; a fresh
machine installation was **not** performed during this documentation review.
Availability of wheels does not validate a particular GPU, driver or native kernel.

```powershell
conda create --name pointcept python=3.8 pip --yes
conda run --name pointcept python -m pip install --only-binary=:all: "numpy==1.24.3" "torch==1.12.1+cu113" --extra-index-url https://download.pytorch.org/whl/cu113
conda run --name pointcept python -m pip install --no-index --find-links https://data.pyg.org/whl/torch-1.12.0+cu113.html "torch-scatter==2.0.9"
conda run --name pointcept python -m pip install --only-binary=:all: "spconv-cu113==2.3.6" "cumm-cu113==0.4.11" "ninja==1.11.1.1"
```

The wheel sources are the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/),
the [PyG PyTorch-1.12/CUDA-11.3 wheel index](https://data.pyg.org/whl/torch-1.12.0+cu113.html)
and [SpConv's versioned Windows distribution](https://pypi.org/project/spconv-cu113/2.3.6/).
Do not substitute an arbitrary torch-scatter wheel for another Python/CUDA ABI.
If a wheel is unavailable, resolve that compatibility issue before allowing pip
to compile an unrelated fallback version.

An NVIDIA GPU and compatible driver are required by this Seg4D runtime. Depending
on the GPU architecture and SpConv/CUMM kernel path, compatible CUDA Toolkit and
MSVC tools can also be needed for native/JIT compilation. The CUDA version shown
by `nvidia-smi` is not itself proof of the runtime installed in PyTorch.

## Verify before building every target

From the repository root:

```powershell
conda run --name pointcept python -c "import numpy, torch, spconv.pytorch, torch_scatter; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available()"
conda run --name pointcept python .\segmentation_methods\point_transformer_v3\point_transformer_v3_cli.py --self-test
```

The second command performs a small model forward pass on CUDA. Imports alone
can succeed while a native GPU kernel still fails. If it fails, resolve the
reported driver, architecture, DLL, kernel or dependency problem first. Keep the
full traceback and the environment's package list when reporting a problem:

```powershell
conda run --name pointcept python -m pip freeze
```

## Build and publish the bundle

```powershell
python .\segmentation_methods\point_transformer_v3\create_executable.py --bootstrap
```

Or use the central script, which prepares the environment automatically:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build-executables.ps1 -Target point-transformer-v3 -Bootstrap
```

The wrapper's `--conda` argument accepts a custom `conda.exe` path. Its
`--bootstrap` installs the pinned CUDA runtime and refreshes isolated build tools.
Later builds can reuse them.
The generated executable runs its CUDA self-test before the directory is
published to `segmentation_methods\point_transformer_v3_runtime`.

Deploy that **complete directory**, including `_internal`; keep architecture,
preprocessing and feature-schema compatibility when reusing model checkpoints.
The bundle's `--self-test` checks a small forward pass, not training accuracy,
large-cloud VRAM requirements or every saved model.
