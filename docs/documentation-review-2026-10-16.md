# Installation/build documentation review — 16 October 2026

## Published repository checked

The public repository is [RECONupm/Seg4D](https://github.com/RECONupm/Seg4D),
default branch `main`. Its README was fetched directly through GitHub during
this review, with blob SHA `4a5f7b0b5960e7a77db8407b5666801f9331c841`.
The GitHub repository root and release collection were also inspected.

Observed published state:

- README describes CloudCompare 2.13/2.13.1, separate manual environment creation,
  Python 3.10 for all executables and the `auto-py-to-exe` GUI.
- The root listing does not contain `scripts/`, `create_installer.py`, BUILDING,
  INSTALLATION or `.vscode`. Fetching `.vscode` returned 404.
- The GitHub Releases collection returned no releases at review time. The README
  links to a separate university-hosted download. The follow-up checked that
  public share directly and located `additional_libraries/point_transformer.7z`;
  this does not certify the old installer's contents or host compatibility.

The newer build pipeline, runtime fixes and VS Code configuration exist locally
but are not established as published by this review. **No push, release or GitHub
file edit was performed.**

## Documentation corrected in the working tree

| Previous gap | Updated explanation |
|---|---|
| Installation and developer compilation mixed together | README routes installer users to INSTALLATION and developers to BUILDING/VS Code. |
| "Latest CloudCompare" implied compatible | Current source target is CPython 3.13 x64; the official site lists 2.14 as beta, 2.13.2 as stable on the review date. |
| All algorithms described as Python 3.10 | Build-driver, embedded host and six target environments are documented separately. |
| Manual auto-py-to-exe output paths | Central manifest, PyInstaller script, target names and published directories documented. |
| PTv3 required manual GPU dependency preparation | Bootstrap now creates pointcept and downloads the pinned CUDA stack; driver/kernel requirements and verification limits remain explicit. |
| PTv1 required a manual archive download | Optional installer flag downloads the exact UPM archive, checks SHA-256 and runs 7-Zip integrity validation. |
| Large generated archives could enter Git | Ignore rules cover installer archives and generated bundles; a read-only size guard checks index blobs and publishable working files. |
| F5 usage not explicit | All launch configurations and new portable tasks/extensions explained. |
| Reusing bundles confused with recompiling sources | Command table distinguishes application-script updates, target rebuilds and installer packaging. |
| Every smoke test implied a numerical test | Anisotropic target currently checks help only; other checks and their limits are stated. |
| Source download confused with a complete distribution | External payloads, missing bundles, third-party Potree and release artifacts explained. |
| DLL/ABI/model failures lacked practical guidance | Installation diagnostics, full runtime replacement, XGBoost/TPOT compatibility and log paths added. |
| Open3D absence explained by a stale general wheel claim | Host helper choice described without claiming no 3.13 wheels exist. |

## Publication prerequisites

These instructions reference the new source tree. Publishing only the Markdown
files would leave broken commands on GitHub. Review and publish the source
migration coherently, including at least:

- `.gitignore`, `.vscode/launch.json`, `.vscode/tasks.json`,
  `.vscode/extensions.json` and `.vscode/settings.json`;
- `create_installer.py`, `seg4d-installer-x64.iss`, `scripts/executable-builds.json`,
  the build/check/validation scripts and all `scripts/requirements` files;
- `pre-requisites/cloudcompare-python313.txt`, the private-runtime launcher and
  `configs/executables.yml` stable bundle paths;
- the matching standalone entry points with their console/self-test contracts,
  the PTv3 source/build wrapper and other new helper modules they import;
- linked documentation and tests, including the deflection numerical helper.

This list is a dependency checklist, not an instruction to blindly stage the
entire working tree. The checkout already contained extensive unrelated changes
and legacy generated-binary removals. Review those separately, avoid committing
generated bundles/caches, and verify a fresh clone after publication.

## Validation scope

The documentation review validates local links, JSON configuration, referenced
paths/target names, PowerShell example syntax and safe CLI argument/help paths.
The link/configuration checks cover all seven current guides; all six target-table rows
match the build manifest and VS Code target picker; all launch/task names are
documented; PowerShell examples and both PowerShell scripts parse successfully;
four CLI help commands exit successfully. The documented invalid
`create_installer.py --bootstrap`-only invocation also returns the expected
argument error without launching a build.
It does not perform another complete six-runtime rebuild, reinstall CloudCompare,
test the VS Code GUI, or recreate the CUDA stack on a fresh Windows machine.
The GPU setup recipe is tied to observed development versions and published
wheel sources; its required imports and forward-pass checks remain necessary
on the actual build machine.

The earlier development run rebuilt the corrected sklearn prediction executable
and exercised synthetic TPOT/XGBoost models. That evidence does not certify the
other bundles or the contents of an external installer download.

The large-file/download follow-up also downloaded the actual PTv1 archive,
verified its full contents with `7z t`, pinned its observed SHA-256, and confirmed
cache reuse. The complete local test suite passed (38 tests, including nine
download/build tests). The Git size guard passed; both large local archives
are ignored. The PTv3 bootstrap command plan was checked for missing/existing
environments and separation of runtime dependencies from build tools.
Inno Setup compiled the standard and PTv1 code branches with output disabled
and a reduced payload inventory. This checks compiler/directive compatibility;
it is not a new complete installer build or a real installation/extraction test.

## Official references checked

- [CloudCompare downloads](https://www.danielgm.net/cc/release/)
- [CloudCompare PythonRuntime](https://tmontaigu.github.io/CloudCompare-PythonRuntime/)
- [VS Code Python environments](https://code.visualstudio.com/docs/python/environments)
- [VS Code Python debugging](https://code.visualstudio.com/docs/python/debugging)
- [VS Code tasks](https://code.visualstudio.com/docs/debugtest/tasks)
- [Inno Setup downloads](https://jrsoftware.org/isdl.php)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [PyG CUDA 11.3 wheels for PyTorch 1.12](https://data.pyg.org/whl/torch-1.12.0+cu113.html)
- [SpConv CUDA 11.3 version 2.3.6](https://pypi.org/project/spconv-cu113/2.3.6/)
