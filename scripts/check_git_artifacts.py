"""Read-only check of files Git would publish (tracked + nonignored untracked).

By default also checks the index blobs, so staging a large file then replacing
its working copy cannot hide it. Gitignore patterns cannot express size limits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def git(*arguments: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["git", *arguments], input=data, check=True,
                          stdout=subprocess.PIPE).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-mib", type=float, default=100,
                        help="Reject files at or above this size (default: 100 MiB).")
    arguments = parser.parse_args()
    if arguments.limit_mib <= 0:
        parser.error("--limit-mib must be positive")
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    limit = arguments.limit_mib * 1024**2
    failures = set()
    warnings = set()
    records = git("ls-files", "--stage", "-z").split(b"\0")
    entries = []
    for record in filter(None, records):
        metadata, name = record.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if mode != b"160000":  # submodules refer to commits, not file blobs
            entries.append((oid, name.decode("utf-8", "surrogateescape")))
    if entries:
        sizes = git("cat-file", "--batch-check=%(objectsize)",
                    data=b"\n".join(oid for oid, _ in entries) + b"\n").splitlines()
        for (_, name), size in zip(entries, sizes):
            count = int(size)
            if count >= limit:
                failures.add(("index", name, count))
            elif count > 50 * 1024**2:
                warnings.add(("index", name, count))
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    for name in set(filter(None, paths.decode("utf-8", "surrogateescape").split("\0"))):
        path = root / name
        if path.is_file() and not path.is_symlink():
            size = path.stat().st_size
            if size >= limit:
                failures.add(("working tree", name, size))
            elif size > 50 * 1024**2:
                warnings.add(("working tree", name, size))
    for label, items in (("WARNING", warnings), ("ERROR", failures)):
        for location, name, size in sorted(items):
            print(f"{label}: {name} ({size / 1024**2:.2f} MiB; {location})")
    if failures:
        print("Remove generated files from the index before committing; .gitignore does not untrack files.")
        return 1
    print("OK: no publishable file reaches the configured Git size limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
