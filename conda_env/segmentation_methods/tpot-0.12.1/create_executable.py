r"""Build the standalone TPOT executable used by Seg4D.

Quick use from VS Code
----------------------
1. Open the Seg4D repository folder in VS Code.
2. Install the Microsoft Python extension and select any Python interpreter.
3. Install Anaconda or Miniconda.
4. Press F5 and select ``Build TPOT executable`` if VS Code asks.

The build uses the pinned ``seg4d-tpot`` Conda environment. If that environment
does not exist, it is created automatically. The generated application is:

    segmentation_methods/tpot_runtime/tpot.exe

The equivalent terminal command is:

    python .\conda_env\segmentation_methods\tpot-0.12.1\create_executable.py

Use ``--bootstrap`` to reinstall the pinned build dependencies explicitly.
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


TPOT_SOURCE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = TPOT_SOURCE_DIRECTORY.parents[2]
BUILD_SCRIPT = REPOSITORY_ROOT / "scripts" / "build-executables.ps1"
BUILD_CONFIGURATION = REPOSITORY_ROOT / "scripts" / "executable-builds.json"

CONDA_CANDIDATES = (
    Path.home() / "anaconda3" / "Scripts" / "conda.exe",
    Path.home() / "miniconda3" / "Scripts" / "conda.exe",
    Path(r"C:\ProgramData\anaconda3\Scripts\conda.exe"),
    Path(r"C:\ProgramData\miniconda3\Scripts\conda.exe"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build segmentation_methods/tpot_runtime/tpot.exe."
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Create/update the pinned Conda environment before building.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the generated executable runtime self-test.",
    )
    parser.add_argument(
        "--conda",
        type=Path,
        help="Full path to conda.exe.",
    )
    return parser.parse_args()


def resolve_conda(explicit: Optional[Path]) -> Path:
    if explicit is not None:
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
        "Conda was not found. Install Anaconda/Miniconda or pass "
        "--conda C:\\path\\to\\conda.exe."
    )


def read_tpot_definition() -> dict:
    with BUILD_CONFIGURATION.open(encoding="utf-8") as configuration_file:
        configuration = json.load(configuration_file)
    return configuration["targets"]["tpot"]


def conda_environment_exists(conda: Path, environment_name: str) -> bool:
    completed = subprocess.run(
        [str(conda), "env", "list", "--json"],
        cwd=str(REPOSITORY_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    environments = json.loads(completed.stdout)["envs"]
    return any(Path(environment).name == environment_name for environment in environments)


def build_tpot(arguments: argparse.Namespace) -> Path:
    conda = resolve_conda(arguments.conda)
    definition = read_tpot_definition()
    environment_name = definition["environment"]
    bootstrap = arguments.bootstrap or not conda_environment_exists(
        conda, environment_name
    )

    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        raise FileNotFoundError("Windows PowerShell was not found.")

    command = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(BUILD_SCRIPT),
        "-Target",
        "tpot",
        "-CondaExecutable",
        str(conda),
    ]
    if bootstrap:
        command.append("-Bootstrap")
    if arguments.skip_smoke_test:
        command.append("-SkipSmokeTests")

    print("\n==> Building the Seg4D TPOT executable", flush=True)
    print("    " + subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        command,
        cwd=str(REPOSITORY_ROOT),
        env=environment,
        check=True,
    )

    output_directory = REPOSITORY_ROOT / definition["outputDirectory"]
    executable = output_directory / f"{definition['executableName']}.exe"
    runtime_dll = output_directory / "_internal" / "python311.dll"
    if not executable.is_file() or not runtime_dll.is_file():
        raise FileNotFoundError(
            "The build completed without producing a complete onedir bundle: "
            f"{output_directory}"
        )
    return executable


def main() -> int:
    try:
        executable = build_tpot(parse_arguments())
    except (
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    bundle_size = sum(
        file_path.stat().st_size
        for file_path in executable.parent.rglob("*")
        if file_path.is_file()
    )
    size_mib = bundle_size / (1024 * 1024)
    print(
        f"\nTPOT executable created: {executable} "
        f"(complete bundle: {size_mib:.1f} MiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
