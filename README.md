![Seg4D](https://github.com/Luisjupm/Seg4D/assets/107433987/07a2e9ef-6ae5-4121-a1aa-df4e7101e47a)

# Seg4D

Seg4D extends CloudCompare with tools for segmentation and geometric analysis
of historic constructions from 3D point clouds. It was developed within the
CAREEN research project (APOYO-JOVENES-21-RCDT1L-85-SL9E1R).

## Start here

| What you want to do | Guide |
|---|---|
| Install and use a supplied Windows binary | [Installation, updates and troubleshooting](INSTALLATION.md) |
| Compile the processing programs and final installer | [Windows build guide](BUILDING.md) |
| Build using VS Code, F5 and tasks | [VS Code workflow](docs/vscode.md) |
| Prepare the PTv3 CUDA environment | [Point Transformer V3 build](docs/point-transformer-v3-build.md) |
| Understand inclined-beam deflection settings/results | [Deflection Analysis](docs/deflection-analysis.md) |

These guides describe **this source tree**. Its host target is Windows x64,
CloudCompare 2.14 with **CPython 3.13 PythonRuntime**. Older installers and PDF
screenshots may describe the earlier 2.13 workflow. Verify the embedded Python
and the supplied installer's compatibility rather than simply choosing "latest".
The [official CloudCompare download page](https://www.danielgm.net/cc/release/)
lists 2.14 as beta as of 16 October 2026.

## Install a supplied binary

1. Install the matching CloudCompare x64 build with its PythonRuntime component,
   into a clean directory.
2. Obtain a matching Seg4D installer from the maintainer. The existing
   [external download](https://drive.upm.es/s/vKEtCr6SgbCUKbG) is separate from
   the Git source; check that package's supported host before installing it.
3. Close CloudCompare, run `SEG4D_Installer Setup.exe`, and select the directory
   containing the intended `CloudCompare.exe`. Internet is required while its
   pinned Python dependencies are installed.
4. Restart CloudCompare and open the **Seg4D** action in its Python plugin.

Installer users do not need Conda, VS Code or Inno Setup. The source ZIP does
not replace the compiled installer. For exact paths, verification commands,
updates and common errors, follow [INSTALLATION.md](INSTALLATION.md).

## Build from source

Use Windows x64 with Git, a Python 3.10+ build driver, Conda, PowerShell and
Inno Setup 6/7. `--bootstrap` creates the separate `pointcept` environment and
downloads its pinned CUDA dependencies automatically. A compatible NVIDIA
GPU/driver is still required for PTv3's CUDA check.

```powershell
git clone https://github.com/RECONupm/Seg4D.git
cd Seg4D
```

After the prerequisites in [BUILDING.md](BUILDING.md) and
[the GPU guide](docs/point-transformer-v3-build.md) are satisfied:

```powershell
python .\create_installer.py --build-executables --bootstrap
```

This explicitly builds/checks all six runtimes, then creates
`installer\SEG4D_Installer Setup.exe`.

To also include the legacy PTv1 runtime, install 7-Zip and add
`--include-legacy-point-transformer`. The builder downloads its 1.80 GiB archive
from the documented UPM share automatically and reuses it on later builds.
See [external payloads and Git size limits](docs/external-payloads.md).

For a later application-script update with complete bundles already present:

```powershell
python .\create_installer.py
```

That command reuses existing processing bundles. Changes to frozen algorithm
sources require a targeted rebuild; see the build guide. `auto-py-to-exe` is not
required. Host dependencies and standalone Python environments are separate.

### VS Code

Open the repository root, install its recommended Python/Python Debugger and
PowerShell extensions, and select a build-driver interpreter. Use **Run and
Debug > select configuration > F5** for a full build or installer packaging.
Use **Terminal > Run Task** to choose a single runtime or run diagnostics.
[The VS Code guide](docs/vscode.md) lists every configuration, prerequisite and
expected output. Running `seg4d.py` in a normal VS Code Python does not provide
CloudCompare's `pycc` API.

## Runtime/build layout

| Location | Purpose |
|---|---|
| `seg4d.py`, `seg4d_plugin/` | Main GUI and CloudCompare launcher |
| `main_module/` | Shared geometry, features, execution and deflection calculations |
| `geometric-based_methods/`, `radiometric-based_methods/`, `other_methods/` | Tool interfaces |
| `segmentation_methods/` | Classical ML, AutoML and deep-learning interfaces |
| `scripts/executable-builds.json` | Target Python versions, environments, entry points and bundle paths |
| `scripts/requirements/` | Active standalone build dependency pins |
| `pre-requisites/cloudcompare-python313.txt` | Private host dependencies for the current PythonRuntime ABI |
| `.vscode/` | Portable F5, task and extension configuration |
| `create_installer.py`, `seg4d-installer-x64.iss` | Final Windows installer build |
| `tests/` | Numerical and simulated-adapter regression tests |
| `assets_sheets/` | Existing scientific/user-method PDF guides |

Generated `*_runtime` directories, `.build` and installer EXEs are build/release
artifacts. Legacy PTv1, some examples and modified third-party payloads are
optional external inputs, not automatically downloaded by the normal build.
PotreeConverter is a separate supplied third-party executable. See BUILDING for
the complete target list and payload requirements.

## Manuals and interface examples

The existing scientific guides cover
[construction-system segmentation](assets_sheets/guide_construction-system-segmentation.pdf),
[damage evaluation](assets_sheets/guide_damage-evaluation.pdf) and
[other methods](assets_sheets/guide_other-methods.pdf).
The [archived installation PDF](installer/seg4d_installation_guide.pdf) is retained
for older distributions; the Markdown installation/build guides above describe
the current tree.

Screenshots below illustrate earlier interfaces and may differ from the current build.

![Seg4D action in CloudCompare](https://github.com/Luisjupm/Seg4D/assets/107433987/156c4ea9-1caa-4da5-ada3-c80e85a8b22b)

![Seg4D interface](https://github.com/Luisjupm/Seg4D/assets/107433987/2a84dd3b-5d17-4ca4-8d3f-852e7cc64ad1)

## Verification and publication status

Use the installation diagnostic and the target-specific smoke tests before
distributing a build. Unit tests and a successful import check do not substitute
for running the real plugin in CloudCompare or validating a clean installation.

The [16 October 2026 documentation review](docs/documentation-review-2026-10-16.md)
records differences between published GitHub main and this development checkout.
Publish the matching source/build tooling with these instructions; README alone
cannot make an older checkout support the new commands.


# Citing Seg4D
Sánchez-Aparicio, L.J.; Santamaría-Maestro, R.; Sanz-Honrado, P.; Villanueva-Llauradó, P.; Aira-Zunzunegui, J.R.; González-Aguilera, D. A Holistic Solution for Supporting the Diagnosis of Historic Constructions from 3D Point Clouds. Remote Sens. 2025, 17, 2018. https://doi.org/10.3390/rs17122018

Or you can cite the repository itself:

https://github.com/RECONupm/Seg4D.git

# References
CloudCompare-PythonRuntime, by Thomas Montaigu: [CloudCompare-PythonRuntime](https://github.com/tmontaigu/CloudCompare-PythonRuntime)

# Acknowledgement
The development of Seg4D plugin has been supported by the Community of Madrid and the Higher Polytechnic School of Madrid through the Project CAREEN (desarrollo de nuevos métodos basados en inteligenCia ARtificial para la caracterización de daños en construccionEs históricas a través de nubEs de puNtos 3D) with reference APOYO-JOVENES-21-RCDT1L-85-SL9E1R. Pablo Sanz's pre-doctoral contract is part of grant PID2022-140071OB-C21, funded by MCIN/AEI/10.13039/501100011033 and ESF+.
