"""Image Quality Assessment (IQA) filtering using pyiqa."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_iqa_model(metric_name: str = "qualiclip", device: str = "cuda"):
    """
    Load an IQA model from pyiqa.

    Args:
        metric_name: IQA metric to use (e.g., "qualiclip", "brisque", "niqe").
        device: Device to use ("cuda" or "cpu").

    Returns:
        (model, device, higher_better) tuple.
    """
    try:
        import pyiqa
    except ImportError:
        raise ImportError(
            "pyiqa is not installed. Install it with:\n"
            "  pip install pyiqa\n"
            "or for the latest version:\n"
            "  pip install git+https://github.com/chaofengc/IQA-PyTorch.git"
        )

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading {metric_name} model on {device}...")
    model = pyiqa.create_metric(metric_name, device=device)
    higher_better = not model.lower_better
    print(f"[INFO] {metric_name} loaded. Higher score = better quality: {higher_better}")
    return model, device, higher_better


def filter_quality_images(
    color_dir: Path,
    metric_name: str,
    threshold: float,
    file_pattern: str = "*.jpg",
    device: str = "cuda",
    min_pass_count: int = 20,
    fallback_top_k: int = 50,
) -> list[str]:
    """
    Filter images by quality using pyiqa metrics (GPU-based).

    Args:
        color_dir: Directory containing image frames.
        metric_name: IQA metric to use (e.g., "qualiclip", "brisque").
        threshold: Quality threshold. Interpretation depends on metric:
                   - QualiCLIP: keep frames with score >= threshold (higher is better)
                   - BRISQUE: keep frames with score <= threshold (lower is better)
        file_pattern: Glob pattern for image files (default: "*.jpg").
        device: Device to use ("cuda" or "cpu").
        min_pass_count: Minimum desired number of threshold-passing frames.
            If fewer pass, fallback to score-ranking selection.
        fallback_top_k: Number of highest-quality frames to keep in fallback mode.

    Returns:
        Ordered list of frame ids (stems) that pass the threshold.
    """
    # Find all image files
    all_image_files = sorted(color_dir.glob(file_pattern))

    # Filter to only include color images (exclude depth and mesh textures)
    image_files = [f for f in all_image_files if ".color." in f.name]

    if not image_files:
        print(f"[WARN] No color images found in {color_dir} with pattern {file_pattern}")
        return []

    # Load IQA model
    model, device, higher_better = load_iqa_model(metric_name, device)

    # Score all images
    scores = {}
    for img_path in tqdm(image_files, desc=f"{metric_name.upper()} filtering", dynamic_ncols=True):
        try:
            score = model(str(img_path)).item()
            scores[img_path.stem] = score
        except Exception as e:
            print(f"[WARN] Failed to score {img_path.name}: {e}")
            scores[img_path.stem] = float("nan")

    valid_scored = [
        (p.stem, float(scores.get(p.stem, float("nan"))))
        for p in image_files
        if not np.isnan(scores.get(p.stem, float("nan")))
    ]
    if not valid_scored:
        print(f"[WARN] No valid IQA scores for {metric_name}.")
        return []

    # Threshold pass
    if higher_better:
        threshold_kept = [stem for stem, score in valid_scored if score >= threshold]
    else:
        threshold_kept = [stem for stem, score in valid_scored if score <= threshold]

    print(
        f"[INFO] {len(threshold_kept)}/{len(valid_scored)} images passed "
        f"(threshold={threshold}, {metric_name})"
    )

    # If too few pass, relax by distribution and keep the top-K scored frames.
    if len(threshold_kept) < max(1, int(min_pass_count)):
        ranked = sorted(valid_scored, key=lambda x: x[1], reverse=higher_better)
        k = min(max(1, int(fallback_top_k)), len(ranked))
        topk = ranked[:k]
        relaxed_threshold = topk[-1][1]
        comparator = ">=" if higher_better else "<="
        print(
            f"[WARN] Only {len(threshold_kept)} frames passed; "
            f"relaxing threshold via score distribution and keeping top {k} "
            f"(effective threshold {comparator} {relaxed_threshold:.6f})."
        )
        return [stem for stem, _ in topk]

    return threshold_kept
