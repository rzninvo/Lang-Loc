"""Instance and semantic mask rendering and 16-bit PNG export."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

# Duplicated from visibility.py to avoid a cross-module dependency.
VOID_ID: int = 0


def pix_to_instance_mask(
    pix_to_face_np: np.ndarray, face_obj_ids: np.ndarray, void_val: int = VOID_ID
) -> np.ndarray:
    """
    Convert per-pixel face index to a 16-bit instance mask.

    The value written is the objectId. Unlabeled faces map to `void_val`.

    Args:
        pix_to_face_np: (H,W) int64 array of face indices (-1 = no face).
        face_obj_ids:   (Nf,) int32 object id per face (-1 = unlabeled).
        void_val:       background value for unlabeled regions.

    Returns:
        (H,W) uint16 array with objectId per pixel (or void).
    """
    h, w = pix_to_face_np.shape
    mask = np.full((h, w), void_val, dtype=np.uint16)
    valid = pix_to_face_np >= 0
    faces = pix_to_face_np[valid]
    inst = face_obj_ids[faces]  # -1 for unlabeled
    inst[inst < 0] = void_val
    mask[valid] = inst.astype(np.uint16)
    return mask


def pix_to_semantic_mask(
    pix_to_face_np: np.ndarray,
    face_obj_ids: np.ndarray,
    obj_to_sem_id: Dict[int, int],
    void_val: int = VOID_ID,
) -> np.ndarray:
    """
    Convert per-pixel face index to a 16-bit semantic class mask.

    We map: face -> objectId -> semanticId, using obj_to_sem_id.
    Unmapped objects become `void_val`.

    Args:
        pix_to_face_np: (H,W) int64 face index per pixel.
        face_obj_ids:   (Nf,) object id per face (-1 = unlabeled).
        obj_to_sem_id:  dict {objectId: semanticId} (from TSV & aggregation).
        void_val:       background value.

    Returns:
        (H,W) uint16 semanticId mask.
    """
    h, w = pix_to_face_np.shape
    mask = np.full((h, w), void_val, dtype=np.uint16)
    valid = pix_to_face_np >= 0
    faces = pix_to_face_np[valid]
    obj_ids = face_obj_ids[faces]
    sem_vals = np.full_like(obj_ids, void_val, dtype=np.int32)
    for i in range(len(obj_ids)):
        oid = int(obj_ids[i])
        if oid >= 0:
            sem_vals[i] = int(obj_to_sem_id.get(oid, void_val))
    mask[valid] = sem_vals.astype(np.uint16)
    return mask


def save_png16(path: Path, arr_uint16: np.ndarray) -> None:
    """
    Save a single-channel 16-bit PNG (preserves large ids correctly).

    Args:
        path: output file path.
        arr_uint16: (H,W) uint16 array to save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_uint16, mode="I;16").save(str(path))
