# Installing Seg4D on Windows

This guide describes the current source tree and its CPython 3.13 installer.
Older installers and the archived PDF guide can target a different CloudCompare
version. Match the installer to its host; a version labelled simply "latest"
is not a compatibility guarantee.

## 1. Requirements

| Requirement | What to check |
|---|---|
| Windows x64 | This installer and its processing executables are Windows builds. |
| CloudCompare with PythonRuntime | The current installer expects CloudCompare 2.14 with **64-bit CPython 3.13**. |
| Internet during Seg4D installation | The installer downloads the pinned host dependencies with pip. It is not an offline installer. |
| Permission to install into the chosen CloudCompare directory | The default is `C:\Program Files\CloudCompare`; installing there normally requires elevation. |
| NVIDIA GPU for Point Transformer V3 | Other tools do not require a GPU just to open Seg4D. PTv3 needs a compatible NVIDIA driver/CUDA runtime. |

Get CloudCompare from the [official download page](https://www.danielgm.net/cc/release/).
Select its PythonRuntime/Python plugin option. As checked on 16 October 2026,
the download page lists **2.14 as beta** and **2.13.2 as stable**: selecting
the stable download alone does not meet this source tree's 3.13 requirement.
Check the embedded interpreter rather than assuming every 2.14 build is identical:

```powershell
& "C:\Program Files\CloudCompare\plugins\Python\python.exe" -c "import sys, struct; print(sys.version); print(struct.calcsize('P') * 8)"
```

The result must identify Python 3.13 and 64 bits. Change the directory if needed.
If that executable is absent, the PythonRuntime component is missing or the
selected CloudCompare directory is wrong.

Use a clean CloudCompare directory when changing host versions. Keep any older
working installation separately; copying new binaries over an old Python runtime
can leave incompatible `.pyd`/DLL files behind. PythonRuntime is the plugin used
here; installing the separate CloudComPy package into a system Python is not a
replacement. See the [PythonRuntime documentation](https://tmontaigu.github.io/CloudCompare-PythonRuntime/).

## 2. Obtain the matching Seg4D installer

The project's [external installer download](https://drive.upm.es/s/vKEtCr6SgbCUKbG)
is maintained separately from this source checkout. Check the supplied build's
compatibility information: its contents/version were not verified during the
documentation review. For the installer generated from this tree, follow this
guide; follow the matching archived guide for an older distribution.

If no matching binary has been supplied, a developer must build
`installer\SEG4D_Installer Setup.exe` using [BUILDING.md](BUILDING.md).
Downloading the source ZIP is not equivalent to downloading that installer.
An installer user does **not** need VS Code, Conda, Inno Setup or a separate
system Python.

PTv3 is bundled by the standard build. Legacy PTv1 is optional: a developer can
build with `--include-legacy-point-transformer`, which downloads its archive
automatically and includes the extractor. End users do not download that archive
or install 7-Zip separately. See [external payloads](docs/external-payloads.md).

## 3. Install

1. Close CloudCompare and any running Seg4D processing job.
2. Run `SEG4D_Installer Setup.exe`.
3. Select the directory containing the intended `CloudCompare.exe`, for example
   `C:\Program Files\CloudCompare`. Do not select the repository, `plugins`,
   or a second CloudCompare installation by mistake.
4. Allow dependency installation to finish. It can take several minutes; network
   failures are reported with a log path.
5. Reopen that same CloudCompare installation and use the **Seg4D** action in its
   Python plugin. A source script is not launched by double-clicking it in Explorer.

Expected installed layout:

```text
CloudCompare/
  CloudCompare.exe
  plugins/Python/python.exe
  plugins/Python/Lib/site-packages/
    seg4d/                       # application scripts and processing bundles
    seg4d_runtime/               # host dependencies for CPython 3.13
  plugins-python/seg4d_plugin/   # Seg4D menu action / launcher
```

The installer uses [pre-requisites/cloudcompare-python313.txt](pre-requisites/cloudcompare-python313.txt)
for the private host runtime. Processing programs have their own bundled Python
and libraries. Installing XGBoost into the host Python will not fix a missing
XGBoost dependency inside `sklearn_runtime`.

## 4. Validate and first launch

If you have the source checkout, run its diagnostic from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\check-cloudcompare-runtime.ps1 `
  -CloudCompareDirectory "C:\Program Files\CloudCompare"
```

It checks Python ABI and imports of the configured dependencies from the private
runtime. It does not exercise all GUI tools, models or third-party converters.

For an initial smoke test, open CloudCompare, load a small point cloud, select it
and open Seg4D. Use a small operation before processing a large project. Some
existing tools require a selected cloud/folder; launching with no selection can
still fail in modules that query the selection during import. Record the
traceback if opening Seg4D fails.

## 5. Updates

- Back up the working installation and keep models/configurations/results outside
  the installation directory. This installer does not provide transactional
  rollback of its private Python runtime: a failed dependency installation can
  require reinstalling the matching Seg4D build.
- A normal update uses a newly built installer. It includes updated GUI/numerical
  Python files without recompiling unchanged processing executables.
- A standalone-runtime update replaces the **whole matching runtime directory**,
  including its `.exe` and `_internal` folder. Preserve the previous directory
  separately; do not mix files from different builds.
- A source-only patch replaces just the files specified by that patch and adds
  any new helper modules. **Merge directories**; do not delete an entire
  `main_module` or methods directory to copy one patched file.
- Restart CloudCompare after an update: Python modules already imported into the
  host are not automatically replaced by editing files on disk.

## Troubleshooting

| Symptom | Action |
|---|---|
| PythonRuntime not found | Verify the selected install directory and its `plugins\Python\python.exe`; install the matching Python component. |
| Installer rejects the ABI | Check the interpreter command above. Current pins are for CPython 3.13 x64, not any installed Python. |
| pip/dependency installation failed | Read `<CloudCompare>\seg4d-pip-install.log`; check connectivity, proxy and directory permissions, then rerun the matching installer. |
| Seg4D is absent from the menu | Check `plugins-python\seg4d_plugin`, restart the correct CloudCompare installation and inspect the Python plugin output. |
| `No module named pycc` in VS Code | The GUI needs CloudCompare's embedded Python; run numerical tests or build helpers in VS Code instead. |
| `No module named xgboost` during prediction | Update/rebuild the complete `sklearn_runtime` bundle with the matching TPOT/XGBoost dependencies; see [BUILDING.md](BUILDING.md#automl-model-compatibility). |
| A model fails after an update | Use its compatible runtime and original feature schema; do not blindly upgrade sklearn/XGBoost or reorder features. |
| Missing DLL / invalid Win32 application | Check architecture, ABI and the runtime's complete `_internal` directory; keep version-specific installations separate. |
| PTv3 reports CUDA unavailable | Verify GPU/driver and the PTv3 bundle/environment; see [the GPU build guide](docs/point-transformer-v3-build.md). |

Processing jobs capture their subprocess output in the selected result directory.
When reporting a problem, include the **full execution log**, application/build
version, tool, parameters and whether the same small input succeeds. A screenshot
can omit the first useful cause in a traceback.
