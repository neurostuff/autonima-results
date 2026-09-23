#!/usr/bin/env python3
"""The one brain mask every map comparison uses.

WHY THIS EXISTS

NiMARE writes zeros, not NaNs, outside the brain. Every map comparison in this repo built its
voxel set with `np.isfinite(...)`, which therefore selected nothing: all 902,629 voxels of the
91 x 109 x 91 grid passed, and 674,146 of them (75%) sat outside the brain at exactly 0 in every
map. That shared zero mass inflates r-squared -- measured at +0.06 to +0.08 on the social
contrasts -- because three quarters of the correlation's n is two constants agreeing.

It did NOT create the reported margins: the inflation applies to the pipeline and baseline arms
alike, and restricting to the brain makes the margin slightly larger, not smaller. But the
absolute r-squared values were not comparable with brain-masked correlations from other work,
so the mask is now applied everywhere.

DICE IS UNAFFECTED, BY CONSTRUCTION

No voxel outside the brain mask exceeds zero in any of the 472 FDR-corrected z maps in the repo,
so no out-of-brain voxel can pass the z > 1.96 threshold. Dice values must be byte-identical
before and after this change; if one moves, something else broke. That is the regression anchor
for this refactor.

WHY A FILE AND NOT A CALL

`make_brain_map_figure.py` runs under system python3 (nilearn 0.13), which carries a different
NiMARE than the pixi environment. Materialising the mask once and loading it with nibabel means
both interpreters index exactly the same voxels, and the mask is an auditable artifact rather
than whatever the installed NiMARE happens to return.
"""

from __future__ import annotations

import functools
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MASK_PATH = REPO_ROOT / "reports" / "brain_mask_mni152_2mm.nii.gz"
SPACE = "mni152_2mm"
SHAPE = (91, 109, 91)


def _build() -> None:
    """Materialise NiMARE's mni152_2mm brain mask. Run once; the file is committed."""
    from nimare.utils import get_template

    img = get_template(space=SPACE, mask="brain")
    data = (np.asarray(img.dataobj) > 0).astype(np.uint8)
    MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, img.affine), MASK_PATH)


@functools.lru_cache(maxsize=1)
def brain_mask() -> np.ndarray:
    """The estimation mask, as a boolean array. 228,483 voxels."""
    if not MASK_PATH.exists():
        _build()
    return np.asarray(nib.load(str(MASK_PATH)).dataobj).astype(bool)


def common_mask(*arrays: np.ndarray, brain: bool = True) -> np.ndarray:
    """Voxels finite in every array and, unless told otherwise, inside the brain.

    Pass `brain=False` only for maps that are genuinely not on the estimation grid. A shape
    mismatch raises rather than silently falling back, because a silent fallback here is exactly
    the failure this module exists to prevent.
    """
    if not arrays:
        raise ValueError("common_mask needs at least one array")
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"arrays must share a shape; got {sorted(shapes)}")
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    if not brain:
        return mask
    bm = brain_mask()
    if bm.shape != mask.shape:
        raise ValueError(
            f"maps are {mask.shape}, brain mask is {bm.shape}; pass brain=False only if these "
            "maps are genuinely off the mni152_2mm estimation grid"
        )
    return mask & bm


if __name__ == "__main__":
    _build()
    bm = brain_mask()
    print(f"wrote {MASK_PATH.relative_to(REPO_ROOT)}  shape={bm.shape}  voxels={int(bm.sum()):,}")
