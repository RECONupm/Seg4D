"""Validate Seg4D's private dependency directory inside CloudCompare."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import re
import sys
from pathlib import Path


MODULE_BY_DISTRIBUTION = {
    "pillow": "PIL.Image",
    "pymupdf": "fitz",
    "pyyaml": "yaml",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
}


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirements(path: Path) -> dict[str, str]:
    expected = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise ValueError(f"Requirement must use an exact version: {line}")
        distribution, version = line.split("==", 1)
        expected[normalized_name(distribution)] = version
    return expected


def is_inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def validate(runtime: Path, requirements: Path, expected_python: str) -> None:
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise RuntimeError(
            f"Expected CPython {expected_python}, found CPython {actual_python}."
        )

    runtime = runtime.resolve()
    if not runtime.is_dir():
        raise FileNotFoundError(f"Seg4D runtime directory does not exist: {runtime}")

    expected = read_requirements(requirements)
    installed = {
        normalized_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(runtime)])
        if distribution.metadata["Name"]
    }
    errors = []
    for distribution_name, expected_version in expected.items():
        installed_version = installed.get(distribution_name)
        if installed_version != expected_version:
            errors.append(
                f"{distribution_name}: expected {expected_version}, "
                f"found {installed_version or 'not installed'}"
            )

    sys.path.insert(0, str(runtime))
    for distribution_name, module_name in MODULE_BY_DISTRIBUTION.items():
        if distribution_name not in expected:
            continue
        try:
            module = importlib.import_module(module_name)
            module_file = Path(module.__file__).resolve()
            if not is_inside(module_file, runtime):
                errors.append(
                    f"{module_name} was loaded outside Seg4D runtime: {module_file}"
                )
            else:
                print(f"{module_name}: OK ({module_file})")
        except Exception as error:
            errors.append(f"{module_name}: {type(error).__name__}: {error}")

    if errors:
        raise RuntimeError(
            "Seg4D CloudCompare runtime validation failed:\n - "
            + "\n - ".join(errors)
        )

    print(
        f"Seg4D runtime is compatible with CPython {actual_python}: {runtime}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--expected-python", default="3.13")
    arguments = parser.parse_args()
    validate(
        runtime=arguments.runtime,
        requirements=arguments.requirements,
        expected_python=arguments.expected_python,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
