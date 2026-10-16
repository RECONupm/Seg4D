# Large files, downloads and optional Point Transformer V1

Source code belongs in Git; generated installers, archives, frozen runtime
directories, environments and example point clouds are excluded by `.gitignore`.
In particular, `installer/SEG4D_Installer Setup.zip` is about 1.57 GiB and the
legacy PTv1 archive is 1.80 GiB. Distribute these as release artifacts or through
the project's external download service, not as ordinary Git blobs.

GitHub warns above 50 MiB and blocks files above 100 MiB in ordinary Git pushes:
see [GitHub's file-size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
`.gitignore` cannot express a size condition and does not remove tracked files
or previous history; see [Git's ignore documentation](https://git-scm.com/docs/gitignore).
Before committing, run this read-only check from the repository root:

```powershell
python .\scripts\check_git_artifacts.py
```

It checks index blobs and nonignored working files, rejects files at or above
100 MiB, and reports warnings above 50 MiB. If an artifact is already tracked,
remove only that path from the index with `git rm --cached -- <path>` while
keeping the local file. Existing history needs a separate coordinated cleanup
if a previous commit contains a blocked file. No history rewrite is performed
by these tools. The existing 30 MiB `pretrained_model.pt` is below the limit;
it is not deleted or silently removed from the index.

## Include PTv1 automatically

Install 7-Zip on the build machine, then package the existing six bundles:

```powershell
python .\create_installer.py --include-legacy-point-transformer
```

For a new build machine, prepare all six targets (including PTv3) and include PTv1:

```powershell
python .\create_installer.py --build-executables --bootstrap --include-legacy-point-transformer
```

The [UPM share linked from GitHub](https://drive.upm.es/s/vKEtCr6SgbCUKbG)
contains `additional_libraries/point_transformer.7z`. Its direct HTTPS source,
repository destination and expected size (1,936,617,942 bytes, checked on
16 October 2026) are recorded in [external-payloads.json](../scripts/external-payloads.json).
The builder downloads only that archive, not the whole share or the old installer.
It stores the archive at `segmentation_methods/point_transformer.7z` and reuses
it later. Allow several GiB of free space for the archive, installer and extraction.

Downloads stream into a temporary `.part` file. HTML responses, unexpected size,
invalid 7z headers and insecure redirects are rejected before publication.
The downloader prints SHA-256; packaging also runs a full `7z t` integrity test.
The manifest pins SHA-256
`7a11f241fe795b9d33556d5ff2f88f15587cde58e0229afc65b11c92d0fb0720`,
calculated from the archive downloaded during this review. Future downloads and
cached files must match it. The publisher has not supplied an independent
SHA-256, so this local reference is not an independently authenticated checksum.
An invalid existing cache causes an error; inspect/remove that archive explicitly
before retrying. A changed upstream artifact requires reviewing the manifest.

7-Zip is discovered on PATH or in its standard Windows installation directory;
use `--sevenzip "D:\Tools\7-Zip\7z.exe"` for a custom location. `7z.exe` needs
its adjacent `7z.dll` and `License.txt`; a standalone `7za.exe` needs its adjacent
`License.txt`. The extractor and license are bundled. During installation the
archive is extracted with installer permissions and failures are reported.
The end user does not need a separate 7-Zip installation or a PTv1 download.

The archive alone can also be fetched before packaging:

```powershell
python .\scripts\download_external_payloads.py
```

PTv1 inclusion does not require restoring example datasets or `modified_libraries`.
The older `--include-external-payloads` option also enables automatic PTv1 download,
but still requires the other historical payloads to be present locally.
The standard installer includes PTv3 and omits optional PTv1. PTv1 extraction
does not establish GPU compatibility or validate legacy models on a new machine.

For automatic PTv3 dependency setup, see [the GPU build guide](point-transformer-v3-build.md).
