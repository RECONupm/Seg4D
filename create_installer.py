"""Build the Seg4D Windows installer without using auto-py-to-exe.

Quick use from VS Code
----------------------
1. Open the Seg4D repository folder in VS Code.
2. Install the Microsoft Python extension and select a Python interpreter.
3. Install Inno Setup 6 or 7.
4. Press F5 and select ``Build Seg4D installer`` if VS Code asks.

For a first/full build, ``--bootstrap`` also prepares the pinned Pointcept CUDA
environment. Optional PTv1 payloads are downloaded when requested. See BUILDING.md and
docs/vscode.md for all prerequisites and launch configurations.

The F5 configuration is stored in ``.vscode/launch.json`` and runs this file
from the repository root. Existing standalone executables are reused; changes
to regular Seg4D Python scripts are included automatically. The resulting file
is written to ``installer/SEG4D_Installer Setup.exe``.

The equivalent terminal command is:

    python .\\create_installer.py

Use ``--build-executables --bootstrap`` only on a new build machine or when the
standalone processing executables must also be rebuilt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from scripts.download_external_payloads import download_payload


REPOSITORY_ROOT = Path(__file__).resolve().parent
INNO_SCRIPT = REPOSITORY_ROOT / "seg4d-installer-x64.iss"
INSTALLER_OUTPUT = REPOSITORY_ROOT / "installer" / "SEG4D_Installer Setup.exe"
EXECUTABLE_BUILD_SCRIPT = REPOSITORY_ROOT / "scripts" / "build-executables.ps1"
EXECUTABLE_BUILD_CONFIG = REPOSITORY_ROOT / "scripts" / "executable-builds.json"

EXTERNAL_PAYLOADS = (
    REPOSITORY_ROOT / "point_clouds_examples",
    REPOSITORY_ROOT / "segmentation_methods" / "point_transformer.7z",
    REPOSITORY_ROOT / "modified_libraries" / "tkPDFViewer",
    REPOSITORY_ROOT / "modified_libraries" / "tkPDFViewer-0.1.dist-info",
    REPOSITORY_ROOT / "modified_libraries" / "7za",
)

INNO_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
    Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    Path(r"C:\Program Files (x86)\Inno Setup 7\ISCC.exe"),
    Path(r"C:\Program Files\Inno Setup 7\ISCC.exe"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create installer/SEG4D_Installer Setup.exe. Existing standalone "
            "executables are reused unless --build-executables is requested."
        )
    )
    parser.add_argument(
        "--include-legacy-point-transformer",
        action="store_true",
        help="Download/cache PTv1 and include it using an installed 7-Zip extractor.",
    )
    parser.add_argument(
        "--sevenzip", type=Path,
        help="Optional path to 7z.exe (with adjacent 7z.dll and License.txt) or 7za.exe.",
    )
    parser.add_argument(
        "--build-executables",
        action="store_true",
        help=(
            "Rebuild the standalone processing executables before the installer. "
            "Without this option they are never modified."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Create/update the pinned Conda build environments.",
    )
    parser.add_argument(
        "--targets",
        help=(
            "Comma-separated executable targets. This implies --build-executables; "
            "omit it to build every target."
        ),
    )
    parser.add_argument(
        "--skip-smoke-tests",
        action="store_true",
        help="Skip the configured generated-executable runtime checks.",
    )
    parser.add_argument(
        "--include-external-payloads",
        action="store_true",
        help=(
            "Include point_transformer.7z, point_clouds_examples and "
            "modified_libraries. They are excluded by default."
        ),
    )
    parser.add_argument(
        "--iscc",
        type=Path,
        help="Full path to the Inno Setup command-line compiler (ISCC.exe).",
    )
    return parser.parse_args()


def resolve_executable(explicit: Optional[Path], candidates: Sequence[Path]) -> Path:
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Executable not found: {resolved}")
        return resolved

    command = shutil.which("ISCC.exe") or shutil.which("ISCC")
    if command:
        return Path(command).resolve()

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Inno Setup Compiler was not found. Install Inno Setup 6/7 or pass "
        "--iscc C:\\path\\to\\ISCC.exe."
    )


def run_checked(command: Sequence[str], description: str) -> None:
    print(f"\n==> {description}", flush=True)
    print("    " + subprocess.list2cmdline(list(command)), flush=True)
    subprocess.run(command, cwd=str(REPOSITORY_ROOT), check=True)


def configured_executables() -> dict[str, Path]:
    with EXECUTABLE_BUILD_CONFIG.open(encoding="utf-8") as configuration_file:
        configuration = json.load(configuration_file)

    executables = {}
    for target_name, definition in configuration["targets"].items():
        executables[target_name] = (
            REPOSITORY_ROOT
            / definition["outputDirectory"]
            / f"{definition['executableName']}.exe"
        )
    return executables


def missing_executable_targets() -> List[str]:
    return [
        target_name
        for target_name, executable in configured_executables().items()
        if not (
            executable.is_file()
            and (executable.parent / "_internal").is_dir()
            and any((executable.parent / "_internal").glob("python*.dll"))
        )
    ]


def build_standalone_executables(arguments: argparse.Namespace) -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        raise FileNotFoundError("Windows PowerShell was not found.")

    command: List[str] = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(EXECUTABLE_BUILD_SCRIPT),
    ]
    if arguments.bootstrap:
        command.append("-Bootstrap")
    if arguments.skip_smoke_tests:
        command.append("-SkipSmokeTests")
    if arguments.targets:
        command.extend(("-Target", arguments.targets))

    run_checked(command, "Build standalone processing executables")


def ensure_standalone_executables(arguments: argparse.Namespace) -> None:
    if arguments.build_executables:
        build_standalone_executables(arguments)

    missing_targets = missing_executable_targets()
    if not missing_targets:
        return

    missing_paths = "\n".join(
        f"  - {configured_executables()[target]}"
        for target in missing_targets
    )
    raise FileNotFoundError(
        "The installer cannot be created without these standalone executables:\n"
        f"{missing_paths}\n"
        "They were not rebuilt automatically. Run create_installer.py "
        "--build-executables --bootstrap only when you explicitly want to "
        "regenerate them."
    )


def assert_external_payloads_exist() -> None:
    missing = [path for path in EXTERNAL_PAYLOADS if not path.exists()]
    if not missing:
        return

    relative_paths = "\n".join(
        f"  - {path.relative_to(REPOSITORY_ROOT)}" for path in missing
    )
    raise FileNotFoundError(
        "External payload inclusion was requested, but these paths are missing:\n"
        f"{relative_paths}"
    )


def prepare_legacy_payload(explicit_extractor: Optional[Path]) -> List[str]:
    candidates = [explicit_extractor] if explicit_extractor else [
        Path(command) for command in (shutil.which("7za.exe"), shutil.which("7z.exe")) if command
    ] + [Path(r"C:\Program Files\7-Zip\7z.exe"),
         REPOSITORY_ROOT / "modified_libraries" / "7za" / "7za.exe"]
    extractor = next((path.resolve() for path in candidates if path and path.is_file()), None)
    if extractor is None:
        raise FileNotFoundError("PTv1 packaging requires 7-Zip. Install it or pass --sevenzip <7z.exe/7za.exe>.")
    dll = extractor.with_name("7z.dll")
    license_file = extractor.with_name("License.txt")
    if not license_file.is_file():
        raise FileNotFoundError(f"7-Zip redistribution requires its adjacent License.txt: {extractor}")
    if extractor.name.lower() == "7z.exe" and not dll.is_file():
        raise FileNotFoundError(f"7z.exe requires its adjacent 7z.dll: {extractor}")
    archive = download_payload()
    run_checked([str(extractor), "t", str(archive), "-bso0", "-bsp0"],
                "Validate the complete PTv1 archive with 7-Zip")
    defines = ["/DIncludeLegacyPointTransformer=1", f"/DLegacyExtractor={extractor}",
               f"/DLegacyExtractorLicense={license_file}"]
    if extractor.name.lower() == "7z.exe":
        defines.append(f"/DLegacyExtractorDLL={dll}")
    return defines


def compile_installer(arguments: argparse.Namespace) -> None:
    iscc = resolve_executable(arguments.iscc, INNO_CANDIDATES)
    INSTALLER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    command = [str(iscc), "/Qp"]
    if arguments.include_legacy_point_transformer or arguments.include_external_payloads:
        command.extend(prepare_legacy_payload(arguments.sevenzip))
    if arguments.include_external_payloads:
        assert_external_payloads_exist()
        command.append("/DIncludeExternalPayloads=1")
    command.append(str(INNO_SCRIPT))

    run_checked(command, "Compile the Seg4D Inno Setup installer")

    if not INSTALLER_OUTPUT.is_file():
        raise FileNotFoundError(
            "Inno Setup completed, but the expected output was not created: "
            f"{INSTALLER_OUTPUT}"
        )

    size_mib = INSTALLER_OUTPUT.stat().st_size / (1024 * 1024)
    print(
        f"\nInstaller created: {INSTALLER_OUTPUT} ({size_mib:.1f} MiB)",
        flush=True,
    )


def main() -> int:
    arguments = parse_arguments()
    if arguments.targets:
        arguments.build_executables = True
    if arguments.bootstrap and not arguments.build_executables:
        print(
            "ERROR: --bootstrap is only used with --build-executables.",
            file=sys.stderr,
        )
        return 2

    try:
        ensure_standalone_executables(arguments)
        compile_installer(arguments)
    except (
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
