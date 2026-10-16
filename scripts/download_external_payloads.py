"""Download the optional legacy PTv1 archive; never execute its contents.

Only a fully downloaded, validated file replaces the destination. Existing
archives are reused. The manifest pins a locally verified SHA-256; the source
currently publishes no independent digest for publisher authentication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
import zlib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "external-payloads.json"


def validate_archive(path: Path, definition: dict) -> str:
    if path.stat().st_size != definition["sizeBytes"]:
        raise ValueError(f"Unexpected archive size: {path}; expected {definition['sizeBytes']} bytes")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        header = stream.read(32)
        if (len(header) != 32 or header[:6] != b"7z\xbc\xaf\x27\x1c"
                or zlib.crc32(header[12:32]) != struct.unpack("<I", header[8:12])[0]):
            raise ValueError(f"Not a valid 7z start header: {path}")
        digest.update(header)
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    if definition.get("sha256") and checksum != definition["sha256"].lower():
        raise ValueError(f"SHA-256 mismatch: {path}")
    return checksum


def download_payload(name: str = "point-transformer-v1", *, root: Path = ROOT,
                     manifest: Path = MANIFEST, opener=urlopen) -> Path:
    definition = json.loads(manifest.read_text(encoding="utf-8"))[name]
    destination = (root / definition["destination"]).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError("Payload destination must be inside the repository")
    url = definition["url"]
    if urlsplit(url).scheme != "https":
        raise ValueError("Payload downloads require HTTPS")
    if destination.is_file():
        checksum = validate_archive(destination, definition)
        print(f"Reusing {destination} (SHA-256 {checksum})", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Unique same-directory temporary file: interruption cannot publish a partial archive.
    import tempfile
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part",
                                                 dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        received = 0
        last_progress = time.monotonic()
        print(f"Downloading {name}: {definition['sizeBytes'] / 1024**3:.2f} GiB", flush=True)
        with os.fdopen(descriptor, "wb") as output:
            with opener(Request(url, headers={"User-Agent": "Seg4D-build/1.0"}), timeout=60) as response:
                if urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("Refusing a redirect to an insecure download")
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Download returned an HTML/login page instead of the archive")
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    received += len(block)
                    if received > definition["sizeBytes"]:
                        raise ValueError("Download exceeds the expected archive size")
                    output.write(block)
                    if time.monotonic() - last_progress >= 10:
                        print(f"  {received / definition['sizeBytes']:.0%}", flush=True)
                        last_progress = time.monotonic()
        checksum = validate_archive(temporary, definition)
        temporary.replace(destination)
        print(f"Downloaded {destination} (SHA-256 {checksum})", flush=True)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        download_payload()
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
