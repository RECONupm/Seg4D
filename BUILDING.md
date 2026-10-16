# Building Seg4D on Windows

This builds Seg4D's standalone Python processing programs and its Inno Setup
installer. It does not compile CloudCompare or the PythonRuntime C++ plugin.
For installing a supplied binary, use [INSTALLATION.md](INSTALLATION.md).
For the editor workflow, use [the VS Code guide](docs/vscode.md).

## Three separate Python roles

| Role | Interpreter / dependency source |
|---|---|
| Build driver | A normal Python 3.10+ runs `create_installer.py` and the small build wrappers. Python 3.11 is a convenient editor environment. |
| CloudCompare host | Current installer targets CloudCompare 2.14 with **CPython 3.13 x64**; pins are in [pre-requisites/cloudcompare-python313.txt](pre-requisites/cloudcompare-python313.txt). |
| Standalone algorithms | Independent Conda environments with the versions in [scripts/executable-builds.json](scripts/executable-builds.json). They are frozen with PyInstaller and do not inherit the host's Python. |

The host's geometry helpers use NumPy/SciPy and do not require Open3D. No blanket
claim about the availability of Open3D wheels is needed to build this project.
Do not install every historical Conda requirements export into the host or a
single environment. Active standalone build requirements are the files under
`scripts/requirements`, selected by the JSON manifest.

## 1. Developer prerequisites

- Windows x64; Git; Windows PowerShell; Miniconda or Anaconda.
- A build-driver Python 3.10+ (separate from the target environments).
- [Inno Setup 6 or 7](https://jrsoftware.org/isdl.php), including `ISCC.exe`, for
  final packaging. Normal detection checks PATH and common installation folders.
- Network access to download the pinned build dependencies on first bootstrap.
- A compatible NVIDIA GPU/driver for the
  full PTv3 build and its CUDA smoke test.
- VS Code is optional. A separate Visual Studio C++/CUDA toolchain is relevant
  if a native dependency/kernel needs compilation; VS Code itself is an editor.

Do not work inside `Program Files` or the installed CloudCompare directory.
Clone/open the source in a writable development folder:

```powershell
git clone https://github.com/RECONupm/Seg4D.git
cd Seg4D
python --version
conda --version
```

All commands below run from that repository root. Paths may contain spaces;
the build scripts use argument lists. An administrator terminal is not normally
needed for compilation; installation into `Program Files` is a separate step.

## 2. Know the targets

| Target | Conda environment | Python | Published directory | Configured check |
|---|---|---|---|---|
| `jakteristics` | `seg4d-jakteristics` | 3.8 | `geometric-based_methods/jakteristics_runtime` | `--self-test` |
| `optimal-flow` | `seg4d-optimal-flow` | 3.10 | `segmentation_methods/optimalflow_runtime` | `--self-test` |
| `machine-learning` | `seg4d-machine-learning` | 3.10 | `segmentation_methods/sklearn_runtime` | `--self-test` |
| `tpot` | `seg4d-tpot` | 3.11 | `segmentation_methods/tpot_runtime` | `--self-test` |
| `point-transformer-v3` | `pointcept` | 3.8 | `segmentation_methods/point_transformer_v3_runtime` | CUDA `--self-test` |
| `anisotropic-denoising` | `seg4d-anisotropic` | 3.9 | `other_methods/anisotropic_denoising-1.0.4` | `--help` only |

These legacy versions are intentional compatibility boundaries. Do not upgrade
individual dependencies just because a newer version is published. A successful
`--help` for anisotropic denoising does not validate its numerical filter.

`-Bootstrap` creates missing Conda environments and installs their pinned
requirements, or updates existing environments. PTv3 also installs its CUDA
runtime from `point-transformer-v3-runtime.txt`; its PyInstaller tools remain
isolated under `.build`. See [the GPU guide](docs/point-transformer-v3-build.md)
for the pinned profile and hardware prerequisites.

## 3. First full build

After installing the developer prerequisites:

```powershell
python .\create_installer.py --build-executables --bootstrap
```

This explicitly bootstraps/builds all six configured bundles, runs their checks,
then compiles **one** final installer:

```text
installer/SEG4D_Installer Setup.exe
```

Keep the terminal open until the command succeeds. It prints the output path and
size. Downloads, native dependency analysis, bundle collection and installer
compression can take substantial time; no fixed build duration is promised.

The equivalent two-stage workflow lets you inspect bundles before packaging:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build-executables.ps1 -Bootstrap
python .\create_installer.py
```

To build only the five non-PTv3 targets while preparing GPU support:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build-executables.ps1 `
  -Target jakteristics,optimal-flow,machine-learning,tpot,anisotropic-denoising -Bootstrap
```

This creates those bundles, **not** a reduced final installer. The current
installer entry point requires all six configured bundles. Build PTv3 as well or
obtain its matching complete bundle before running final packaging.

## 4. Subsequent builds: choose what changed

| Change | Command |
|---|---|
| Application/GUI/numerical `.py` files copied directly to CloudCompare | `python .\create_installer.py` |
| Source frozen into the sklearn/prediction executable | `python .\create_installer.py --targets machine-learning` |
| TPOT and sklearn dependencies or frozen sources | `python .\create_installer.py --targets tpot,machine-learning --bootstrap` |
| Rebuild all bundles using the existing environments | `python .\create_installer.py --build-executables` |
| Create/update dependencies and rebuild all bundles | `python .\create_installer.py --build-executables --bootstrap` |

`--targets` implies `--build-executables`, then still packages the complete
installer. Other configured bundles must already be present. `--bootstrap`
alone does not package/rebuild and is rejected by the Python entry point.

For just one standalone bundle without final packaging:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build-executables.ps1 -Target machine-learning -Bootstrap
```

Omit `-Bootstrap` only when its environment already matches the requirements.
The TPOT and PTv3 dedicated build wrappers are also available:

```powershell
python .\conda_env\segmentation_methods\tpot-0.12.1\create_executable.py --help
python .\segmentation_methods\point_transformer_v3\create_executable.py --help
```

## 5. Custom tool locations

The builder detects Conda on PATH or under common Anaconda/Miniconda locations.
For a different location, first build with the explicit executable:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build-executables.ps1 -Target machine-learning -Bootstrap `
  -CondaExecutable "D:\Tools\Miniconda3\Scripts\conda.exe"
```

Use the appropriate target or all targets for your workflow. The installer helper
can then reuse those bundles. For a custom Inno Setup location:

```powershell
python .\create_installer.py --iscc "D:\Tools\Inno Setup 6\ISCC.exe"
```

VS Code F5/tasks are documented separately in [docs/vscode.md](docs/vscode.md).
There is no need to open `auto-py-to-exe` or manually choose output paths.

## 6. What the build checks and publishes

PyInstaller first writes into `.build/executables/<target>`. For most targets it
cleans the build cache; PTv3 is configured to reuse its cache. After compilation,
the central script runs the manifest's check, then replaces the published bundle
with rollback if that replacement fails. A smoke-test failure occurs before
publication and leaves the previously published bundle in place.

The installer entry point checks that each expected executable, `_internal`
folder and embedded Python DLL exists. This is **not** a source/binary freshness
check: stale bundles can pass existence checks. Rebuild a changed processing
source explicitly and retain your build logs/package versions for a release.

Generated runtime folders, `.build` and the installer EXE are ignored by Git.
They are release artifacts, not source files to commit. A source-only checkout
needs rebuilt bundles or compatible artifacts supplied by the maintainer.
Builds disable user-site imports with `PYTHONNOUSERSITE=1`; do not rely on a
package installed only into a developer's Windows user profile.

### AutoML model compatibility

Supervised prediction also loads models produced by TPOT. The `machine-learning`
bundle therefore includes **TPOT 0.12.1, XGBoost 2.0.2 and matching support pins**
(including DEAP 1.4.1), in addition to sklearn 1.3.2. Its smoke test fits,
serializes, reloads and predicts a pipeline containing a TPOT stacking operator
and XGBoost's native library. Newer transitive dependencies must not silently
replace this profile.

A `ModuleNotFoundError: xgboost` while loading a model indicates a missing module
in the processing bundle. Installing it into CloudCompare's private host runtime
will not add it to the frozen executable. Rebuild/update the complete
`segmentation_methods/sklearn_runtime` directory and keep the model's compatible
versions and feature schema.

### Console interfaces

Jakteristics, OptimalFlow, sklearn and TPOT support:

```powershell
.\segmentation_methods\sklearn_runtime\scikit-learn-1.3.2.exe --help
.\segmentation_methods\sklearn_runtime\scikit-learn-1.3.2.exe --self-test
```

Their processing contract is `--input <configuration.yaml> --output <directory>`.
PTv3 instead exposes `train` / `predict` subcommands and its CUDA `--self-test`.
Anisotropic denoising uses `--i` / `--o`. Do not assume one CLI fits every tool.

## 7. Optional and third-party payloads

The normal installer omits these externally supplied payloads:

- `point_clouds_examples`
- `segmentation_methods/point_transformer.7z` (legacy Point Transformer V1)
- `modified_libraries/tkPDFViewer`, its dist-info folder and `modified_libraries/7za`

For PTv1 alone, install 7-Zip and run:

```powershell
python .\create_installer.py --include-legacy-point-transformer
```

The archive is downloaded automatically from UPM, cached in
`segmentation_methods/point_transformer.7z`, validated with `7z t`, and included
with the extractor and its license. No manual archive download is needed.
Add this option to `--build-executables --bootstrap` for a first full build
including both PTv3 and legacy PTv1. The final install extracts PTv1 and reports
an extraction failure. A custom extractor can be supplied with `--sevenzip`.

To also include the example datasets and historical modified libraries, restore
those other required paths, then run:

```powershell
python .\create_installer.py --include-external-payloads
```

This broader flag also downloads PTv1 automatically, then validates the other
paths. It does not download examples/modified libraries or compile V1 from source.
V1 remains optional and separate from PTv3. See
[external payloads and Git size limits](docs/external-payloads.md) for the source,
cache, validation and publication rules.

PotreeConverter is a separate third-party executable under
`other_methods/potree-2.1.1/PotreeConverter.exe`. It is not one of the six Python
build targets. Check that payload and its license separately when distributing
a full installer. This workflow packages Seg4D, not CloudCompare's own binaries.

## 8. Test and validate installation

After bootstrapping the machine-learning environment, run the numerical and
CloudCompare-adapter tests from the repository root:

```powershell
conda run --no-capture-output --name seg4d-machine-learning python -B -m unittest discover -s tests -v
```

Adapter tests use a simulated host, not the real embedded GUI. They do not prove
all tools or the final installer operate on another clean Windows machine.

Install the final EXE into a matching CloudCompare installation and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\check-cloudcompare-runtime.ps1 `
  -CloudCompareDirectory "C:\Program Files\CloudCompare"
```

Then open the real plugin and run small representative jobs, including any GPU
and external tools you distribute. The installed runtime validator uses the
pinned host matrix; this separate PowerShell check verifies ABI/module origins.
A successful import check does not validate all algorithms or models.

The installer downloads host wheels into `seg4d_runtime` during installation;
it is not offline and does not provide transactional runtime rollback. See the
[installation/update instructions](INSTALLATION.md#5-updates) before upgrading.
`--skip-smoke-tests` (Python) / `-SkipSmokeTests` (PowerShell) exist for diagnosis;
a bundle built with checks skipped should not be presented as validated.

## Release/publication checklist

1. Commit the source, active requirements, manifest, build/check scripts,
   installer definition, `.vscode` configuration and linked documentation together.
2. Ensure a fresh checkout contains every relative link and every script referenced
   by the guides. Updating README alone is insufficient when new tooling is untracked.
3. Rebuild changed bundles, keep smoke tests enabled, and generate the installer.
4. Validate a clean installation and an update, including failure recovery.
5. Publish the installer/artifacts with their compatible host ABI and build details;
   keep source and binary download versions identifiable.

The published GitHub main branch was behind this development checkout during
the 16 October 2026 review. See the [publication review](docs/documentation-review-2026-10-16.md)
for the missing files and verification limits. This is not a claim that the local
build tooling or fixes have already been uploaded.
