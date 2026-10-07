"""Download the five MVTec AD categories used by the benchmark (about 840 MB).

MVTec AD (Bergmann et al., CVPR 2019) is distributed by MVTec Software GmbH under
CC BY-NC-SA 4.0: non-commercial use only, attribution required, derivatives share alike.
The archives are fetched from the links shown on the official downloads page
(https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads), verified against
the SHA-256 recorded when this repository was built, extracted, and never committed.

Usage:  python scripts/fetch_mvtec.py [--root data/mvtec_ad] [--categories bottle grid]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from vqgate.data import GOOD, index_category

HOST = "https://www.mydrive.ch/shares"
CHUNK_BYTES = 1 << 20
USER_AGENT = "visual-quality-gate dataset fetcher"


@dataclass(frozen=True)
class Archive:
    url: str
    n_bytes: int
    sha256: str
    train_good: int
    test_good: int
    test_defective: int


ARCHIVES = {
    "bottle": Archive(
        f"{HOST}/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz",
        155_880_244,
        "726512129cb3b1f47d4f6cd7c7fc9170db7fc209132249c788fec06029cb5dc3",
        209,
        20,
        63,
    ),
    "grid": Archive(
        f"{HOST}/150456/bb0b2e3dc804ccb8b4485e01f8e4493b/download/420937487-1629959044/grid.tar.xz",
        160_763_852,
        "d94a9a651cbfaee22c94d06cf8fa8204a590c872362de0489eb34f1645328de8",
        264,
        21,
        57,
    ),
    "metal_nut": Archive(
        f"{HOST}/150459/c68856a21dca589b0f8ff6d4ee0f18f4/download/420937637-1629959294/metal_nut.tar.xz",
        165_414_484,
        "86e3c8e163ceb19146ad3ddc408e85cc34ae02e0da430e07a73bac6bbb48532a",
        220,
        22,
        93,
    ),
    "screw": Archive(
        f"{HOST}/150461/242f454cc6385e5693c4fd4b94567d1e/download/420938130-1629960389/screw.tar.xz",
        195_344_332,
        "0a27a365a21e472d19b9fb48cb8e4002b49fabe083bff7f4c16fa57d78154f10",
        320,
        41,
        119,
    ),
    "zipper": Archive(
        f"{HOST}/150466/bd155b557520edaf692d9bfdb915c24a/download/420938385-1629960680/zipper.tar.xz",
        159_484_784,
        "7ab46e0e195da145db30052baea3d3a5d14c5f0e710137c968e0423f02cb942d",
        240,
        32,
        119,
    ),
}


def image_counts(root: Path, category: str) -> tuple[int, int, int]:
    samples = index_category(root, category)
    train_good = sum(s.split == "train" for s in samples)
    test_good = sum(s.split == "test" and s.defect == GOOD for s in samples)
    test_defective = sum(s.split == "test" and s.is_defective for s in samples)
    return train_good, test_good, test_defective


def is_complete(root: Path, category: str) -> bool:
    archive = ARCHIVES[category]
    expected = (archive.train_good, archive.test_good, archive.test_defective)
    try:
        return image_counts(root, category) == expected
    except FileNotFoundError:
        return False


def download(archive: Archive, destination: Path) -> None:
    """Stream the archive to disk and fail loudly if size or SHA-256 differ."""
    request = urllib.request.Request(archive.url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    received = 0
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        while chunk := response.read(CHUNK_BYTES):
            handle.write(chunk)
            digest.update(chunk)
            received += len(chunk)
    if received != archive.n_bytes or digest.hexdigest() != archive.sha256:
        destination.unlink()
        raise RuntimeError(
            f"{destination.name}: got {received} bytes, sha256 {digest.hexdigest()}; "
            f"expected {archive.n_bytes} bytes, sha256 {archive.sha256}"
        )


def fetch(root: Path, category: str) -> None:
    if is_complete(root, category):
        print(f"{category}: already present")
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVES[category]
    archive_path = root / f"{category}.tar.xz"
    print(f"{category}: downloading {archive.n_bytes / 1e6:.0f} MB")
    download(archive, archive_path)
    with tarfile.open(archive_path) as tar:
        tar.extractall(root, filter="data")
    archive_path.unlink()
    if not is_complete(root, category):
        raise RuntimeError(f"{category}: unexpected image counts {image_counts(root, category)}")
    print(f"{category}: ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--categories", nargs="+", choices=sorted(ARCHIVES), default=list(ARCHIVES))
    args = parser.parse_args()
    print("MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial); do not redistribute the images.")
    for category in args.categories:
        fetch(args.root, category)
    return 0


if __name__ == "__main__":
    sys.exit(main())
