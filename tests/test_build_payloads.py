"""Network-free checks for optional payload acquisition and installer routing."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import download_external_payloads as downloads
import create_installer as installer


class Response(io.BytesIO):
    headers = {"Content-Type": "application/x-7z-compressed"}

    def geturl(self):
        return "https://example.org/archive.7z"


class PayloadTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".build").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".build")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        tail = struct.pack("<QQI", 0, 0, 0)
        self.archive = b"7z\xbc\xaf\x27\x1c\0\4" + struct.pack("<I", zlib.crc32(tail)) + tail
        self.definition = {"url": "https://example.org/archive.7z", "destination": "payload.7z",
                           "sizeBytes": len(self.archive),
                           "sha256": hashlib.sha256(self.archive).hexdigest()}
        self.manifest = self.root / "manifest.json"

    def fetch(self, data=None, opener=None):
        self.manifest.write_text(json.dumps({"point-transformer-v1": self.definition}))
        return downloads.download_payload(root=self.root, manifest=self.manifest,
                                          opener=opener or (lambda *_a, **_kw: Response(data)))

    def test_download_validates_and_reuses_cache_without_network(self):
        result = self.fetch(self.archive)
        self.assertEqual(result.read_bytes(), self.archive)
        self.fetch(opener=lambda *_a, **_kw: self.fail("Cached payload must not download again"))

    def test_truncated_download_is_never_published(self):
        with self.assertRaises(ValueError):
            self.fetch(self.archive[:-1])
        self.assertFalse((self.root / "payload.7z").exists())
        self.assertFalse(list(self.root.glob("*.part")))

    def test_html_is_rejected(self):
        response = Response(b"<html>login</html>")
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        with self.assertRaisesRegex(ValueError, "HTML"):
            self.fetch(opener=lambda *_a, **_kw: response)

    def test_invalid_header_and_digest_are_rejected(self):
        for data in (b"x" * len(self.archive), self.archive[:-1] + b"x"):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    self.fetch(data)

    def test_destination_cannot_escape_repository(self):
        self.definition["destination"] = "../outside.7z"
        with self.assertRaisesRegex(ValueError, "inside"):
            self.fetch(self.archive)

    def test_insecure_redirect_is_rejected(self):
        response = Response(self.archive)
        response.geturl = lambda: "http://example.org/archive.7z"
        with self.assertRaisesRegex(ValueError, "insecure"):
            self.fetch(opener=lambda *_a, **_kw: response)

    def test_legacy_flag_routes_download_validation_and_inno_defines(self):
        extractor = self.root / "7z.exe"
        extractor.touch()
        (self.root / "7z.dll").touch()
        (self.root / "License.txt").touch()
        with patch.object(installer, "download_payload", return_value=self.root / "payload.7z") as fetch:
            with patch.object(installer, "run_checked") as run:
                defines = installer.prepare_legacy_payload(extractor)
        fetch.assert_called_once()
        self.assertEqual(run.call_args.args[0][1], "t")
        self.assertIn("/DIncludeLegacyPointTransformer=1", defines)
        self.assertTrue(any(value.startswith("/DLegacyExtractorDLL=") for value in defines))

    def test_ptv3_wrapper_bootstraps_missing_environment(self):
        spec = importlib.util.spec_from_file_location(
            "ptv3_build_wrapper", ROOT / "segmentation_methods/point_transformer_v3/create_executable.py")
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["wrapper"]))
            stack.enter_context(patch.object(wrapper, "resolve_conda", return_value=Path("conda.exe")))
            stack.enter_context(patch.object(wrapper, "conda_environment_exists", return_value=False))
            stack.enter_context(patch.object(wrapper.shutil, "which", return_value="powershell.exe"))
            run = stack.enter_context(patch.object(wrapper.subprocess, "run"))
            # Stop after command dispatch: do not depend on an existing compiled bundle.
            run.side_effect = subprocess.CalledProcessError(17, "mock-build")
            self.assertEqual(wrapper.main(), 1)
            self.assertIn("-Bootstrap", run.call_args.args[0])

    def test_size_guard_checks_index_even_after_working_file_is_replaced(self):
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True)
        artifact = self.root / "large.dat"
        artifact.write_bytes(b"x" * 2048)
        subprocess.run(["git", "-C", str(self.root), "add", "large.dat"], check=True)
        artifact.write_text("small now")
        result = subprocess.run([sys.executable, str(ROOT / "scripts/check_git_artifacts.py"),
                                 "--limit-mib", "0.001"], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("index", result.stdout)
        self.assertIn("large.dat", result.stdout)


if __name__ == "__main__":
    unittest.main()
