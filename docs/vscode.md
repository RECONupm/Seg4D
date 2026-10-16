# Build and test from VS Code

VS Code edits the source and launches build helpers. CloudCompare itself runs
the Seg4D GUI and provides `pycc`/`cccorelib`; **Run Python File** on `seg4d.py`
in a normal Conda environment is not a way to run the plugin.

## Set up the workspace

1. Install Git, Miniconda/Anaconda and [VS Code](https://code.visualstudio.com/).
   Install [Inno Setup 6 or 7](https://jrsoftware.org/isdl.php) to generate the
   final installer. These are developer requirements, not end-user requirements.
2. Open an Anaconda/Miniconda Prompt. A dedicated build-driver environment avoids
   installing editor tools into the old numerical/GPU environments:

   ```powershell
   conda create --name seg4d-dev python=3.11 pip --yes
   conda activate seg4d-dev
   git clone https://github.com/RECONupm/Seg4D.git
   cd Seg4D
   code .
   ```

   If you already have the checkout, open **that folder** rather than cloning
   over it. If `code` is not on PATH, use VS Code's **File > Open Folder**.
3. Install the workspace's recommended extensions: Python, Python Debugger and
   PowerShell. The suggestions are in [extensions.json](../.vscode/extensions.json).
4. Run **Python: Select Interpreter** from `Ctrl+Shift+P` and select `seg4d-dev`
   (Python 3.11), or another suitable build-driver Python 3.10+. Open a new
   integrated terminal and verify `python --version` and `conda --version`.
5. Open [BUILDING.md](../BUILDING.md). The first **full** build automatically
   prepares [the `pointcept` GPU environment](point-transformer-v3-build.md). The selected
   VS Code interpreter does not replace any target-specific Conda environment.

The build driver uses Python's standard library. Its Python version need not be
3.13: that ABI applies to the **installed CloudCompare host**, not to the build
driver or every frozen processing runtime.

## F5: launch a specific build

Open **Run and Debug** (`Ctrl+Shift+D`), select one of the following configurations
from [launch.json](../.vscode/launch.json), then press **F5**. Selecting the
configuration explicitly matters; the first entry can rebuild every runtime.

| Configuration | What happens / prerequisite |
|---|---|
| `Build installer with legacy PTv1 (automatic download)` | Package existing bundles and download/cache legacy PTv1 if needed. Requires 7-Zip and Inno Setup. |
| `Build ALL runtimes + Seg4D installer (first/full)` | Download/install all target dependencies, including the CUDA stack; build/check all targets, then compile the installer. Requires a compatible GPU/driver and Inno Setup. |
| `Build ALL runtimes + Seg4D installer (reuse environments)` | Recompile all targets using existing environments, then package. Use the bootstrap variant after changing requirements. |
| `Build Seg4D installer` | Package existing complete bundles; includes current application `.py` changes. It does not rebuild standalone executables. |
| `Build TPOT executable` | Build just the AutoML bundle; the wrapper bootstraps a missing environment. Re-run its CLI with `--bootstrap` after requirements changes. |
| `Build Point Transformer V3 executable` | Build PTv3; its wrapper bootstraps a missing `pointcept` environment and isolated build tools automatically. Requires a compatible GPU/driver. |
| `Run Seg4D tests (selected interpreter)` | Run unittest discovery. First select `seg4d-machine-learning`, whose dependencies cover the numerical and adapter tests. |

Build output goes to the integrated terminal. The default installer result is
`installer\SEG4D_Installer Setup.exe`. Do not install that EXE while another
build is still writing it. Builds normally run configured smoke tests before
publishing the new bundle; a failed test leaves the previously published bundle.

For an installer including legacy PTv1 too, install 7-Zip and run
`python .\create_installer.py --build-executables --bootstrap --include-legacy-point-transformer`
in the integrated terminal. Its archive is downloaded from UPM automatically;
see [external payloads](external-payloads.md). The standard F5 build includes PTv3
and omits the optional PTv1 archive.

The debugger uses the selected interpreter by default, as documented in
[VS Code's Python debugger guide](https://code.visualstudio.com/docs/python/debugging).
It does not automatically debug all programs launched by Conda/PyInstaller.
Use the terminal traceback/log to diagnose those child processes.

## Tasks: build a target or run a check

Use **Terminal > Run Task** and choose an entry from
[tasks.json](../.vscode/tasks.json):

| Task | Purpose |
|---|---|
| `Seg4D: bootstrap and build all runtimes` | Full executable build, including PTv3; no final installer packaging. |
| `Seg4D: bootstrap and build CPU runtimes` | Five non-PTv3 targets. Useful before GPU setup; this alone does not meet the final installer's six-bundle requirement. |
| `Seg4D: bootstrap and build selected runtime` | Prompt for the target and create/update its dependencies. |
| `Seg4D: rebuild selected runtime` | Prompt for the target; use its existing environment. |
| `Seg4D: package installer (reuse runtimes)` | Use the selected VS Code Python to run `create_installer.py`. Also the default `Ctrl+Shift+B` task. |
| `Seg4D: check CloudCompare runtime` | Prompt for the installed CloudCompare directory and run the host check. |
| `Seg4D: run tests in machine-learning environment` | Run tests explicitly via Conda, without changing the selected build-driver interpreter. Requires `conda` on PATH. |

Task commands use argument arrays and the repository as working directory,
including when the checkout path contains spaces. They do not contain the
author's username or a fixed Conda environment path. See the official
[Tasks documentation](https://code.visualstudio.com/docs/debugtest/tasks)
and [Python environment selection](https://code.visualstudio.com/docs/python/environments)
for editor behaviour.

## Common editor problems

- **No launch entries/tasks:** open the root containing `.vscode`, `seg4d.py` and
  `create_installer.py`, not an individual methods subfolder. Ensure the checkout
  contains the new build tooling; the older GitHub source does not have it.
- **Conda not found:** reopen VS Code from an Anaconda/Miniconda Prompt, or use
  the terminal build command's `-CondaExecutable` option from BUILDING. Conda in
  a custom directory may not be found by the automatic fallback paths.
- **Inno Setup not found:** install it or pass `--iscc` with the compiler path.
- **F5 packages an old processing algorithm:** application Python files are copied
  by the installer; code frozen into a runtime must first be rebuilt for its target.
- **Tests cannot import NumPy/SciPy/matplotlib:** select `seg4d-machine-learning`
  for the test launch, or use the explicit Conda task.
- **Debugging legacy Python 3.8:** keep the modern editor/build driver separate.
  The build scripts call the target interpreter themselves; selecting `pointcept`
  as the global editor interpreter is not required to compile PTv3.
