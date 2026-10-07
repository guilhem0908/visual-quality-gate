"""AUROC values (in %) published by the authors, copied from the papers for the comparison.

* PaDiM-R18-Rd100, pixel level: Defard et al. (arXiv:2011.08785), Table III. The paper gives
  image-level AUROC only as averages over all 15 categories and not for this backbone, so
  there is no per-category image-level reference.
* PatchCore-10% and PatchCore-1% (WideResNet-50, 224 x 224): Roth et al. (arXiv:2106.08265),
  Table S1 (image level) and Table S2 (pixel level).
"""

from __future__ import annotations

PUBLISHED = {
    "padim_r18": {
        "pixel": {"bottle": 98.1, "grid": 94.9, "metal_nut": 96.7, "screw": 97.4, "zipper": 98.2},
    },
    "patchcore_wr50_10": {
        "image": {"bottle": 100.0, "grid": 97.9, "metal_nut": 100.0, "screw": 97.0, "zipper": 99.5},
        "pixel": {"bottle": 98.6, "grid": 98.7, "metal_nut": 98.4, "screw": 99.4, "zipper": 98.9},
    },
    "patchcore_wr50_1": {
        "image": {"bottle": 100.0, "grid": 98.6, "metal_nut": 99.7, "screw": 96.4, "zipper": 99.2},
        "pixel": {"bottle": 98.5, "grid": 98.6, "metal_nut": 98.4, "screw": 99.2, "zipper": 98.8},
    },
}
