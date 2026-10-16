r"""Build the replaceable Point Transformer V3 console bundle for Seg4D.

Quick use from VS Code
----------------------
1. Open the Seg4D repository folder in VS Code.
2. Install the Microsoft Python extension and Anaconda/Miniconda.
3. Ensure a compatible NVIDIA GPU/driver is available for the CUDA self-test.
4. Press F5 and select ``Build Point Transformer V3 executable``.

The selected VS Code interpreter does not matter. This script locates Conda,
creates a missing Pointcept environment and installs its pinned CUDA runtime,
installs PyInstaller inside ``.build``, builds a console application and runs its
CUDA self-test. The replaceable result is the complete folder:

    segmentation_methods\point_transformer_v3_runtime

Seg4D calls only ``point_transformer_v3.exe`` from that folder. Therefore a
future compatible V3 update can be deployed by replacing the whole folder,
without editing Seg4D.

Equivalent terminal command:

    python .\segmentation_methods\point_transformer_v3\create_executable.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


SOURCE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SOURCE_DIRECTORY.parents[1]
BUILD_SCRIPT = REPOSITORY_ROOT / "scripts" / "build-executables.ps1"
BUILD_CONFIGURATION = REPOSITORY_ROOT / "scripts" / "executable-builds.json"
BUILD_TOOLS = REPOSITORY_ROOT / ".build" / "build-tools" / "point-transformer-v3"
CONDA_CANDIDATES = (
    Path.home() / "anaconda3" / "Scripts" / "conda.exe",
    Path.home() / "miniconda3" / "Scripts" / "conda.exe",
    Path(r"C:\ProgramData\anaconda3\Scripts\conda.exe"),
    Path(r"C:\ProgramData\miniconda3\Scripts\conda.exe"),
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build the Seg4D Point Transformer V3 Windows console bundle."
    )
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--conda", type=Path)
    return parser.parse_args()


def resolve_conda(explicit: Optional[Path]) -> Path:
    if explicit:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Conda executable not found: {candidate}")
        return candidate
    command = shutil.which("conda.exe") or shutil.which("conda")
    if command:
        return Path(command).resolve()
    for candidate in CONDA_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Conda was not found. Install Anaconda/Miniconda or pass --conda."
    )


def conda_environment_exists(conda: Path, name: str) -> bool:
    result = subprocess.run(
        [str(conda), "env", "list", "--json"],
        cwd=str(REPOSITORY_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return any(
        Path(environment).name == name
        for environment in json.loads(result.stdout)["envs"]
    )


def main():
    arguments = parse_arguments()
    try:
        conda = resolve_conda(arguments.conda)
        environment_exists = conda_environment_exists(conda, "pointcept")
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if not powershell:
            raise FileNotFoundError("Windows PowerShell was not found.")
        bootstrap = arguments.bootstrap or not environment_exists or not (
            BUILD_TOOLS / "PyInstaller"
        ).is_dir()
        command = [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD_SCRIPT),
            "-Target",
            "point-transformer-v3",
            "-CondaExecutable",
            str(conda),
        ]
        if bootstrap:
            command.append("-Bootstrap")
        if arguments.skip_smoke_test:
            command.append("-SkipSmokeTests")
        print("\n==> Building Point Transformer V3 console bundle", flush=True)
        print("    " + subprocess.list2cmdline(command), flush=True)
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            command,
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            check=True,
        )
        with BUILD_CONFIGURATION.open(encoding="utf-8") as config_file:
            definition = json.load(config_file)["targets"]["point-transformer-v3"]
        bundle = REPOSITORY_ROOT / definition["outputDirectory"]
        executable = bundle / f"{definition['executableName']}.exe"
        if not executable.is_file() or not (bundle / "_internal").is_dir():
            raise FileNotFoundError(f"Incomplete generated bundle: {bundle}")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    size = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    print(
        f"\nPoint Transformer V3 created: {executable}\n"
        f"Complete replaceable bundle: {size / (1024 * 1024):.1f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
