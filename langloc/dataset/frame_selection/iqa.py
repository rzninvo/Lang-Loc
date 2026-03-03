"""Image Quality Assessment (IQA) filtering using pyiqa."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
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


def _load_image_tensor(img_path: Path) -> torch.Tensor:
    """Load a single image as a normalized float32 CPU tensor."""
    from PIL import Image
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
    tensor = transforms.ToTensor()(img_rgb)  # (3, H, W), float32 [0, 1]
    return tensor


def filter_quality_images(
    color_dir: Path,
    metric_name: str,
    threshold: float,
    file_pattern: str = "*.jpg",
    device: str = "cuda",
    min_pass_count: int = 50,
    fallback_top_k: int = 50,
    batch_size: int = 1,
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
        batch_size: Batch size for IQA inference (1 = legacy sequential).

    Returns:
        Ordered list of frame ids (stems) that pass the threshold.
    """
    # Find all image files
    image_files = sorted(color_dir.glob(file_pattern))

    if not image_files:
        print(f"[WARN] No color images found in {color_dir} with pattern {file_pattern}")
        return []

    # Load IQA model
    model, device, higher_better = load_iqa_model(metric_name, device)

    # Score all images (no_grad: IQA is inference-only, no backprop needed)
    scores = {}
    with torch.no_grad():
        if batch_size <= 1:
            # Legacy: one-at-a-time path scoring
            for img_path in tqdm(image_files, desc=f"{metric_name.upper()} filtering",
                                 dynamic_ncols=True):
                try:
                    score = model(str(img_path)).item()
                    scores[img_path.stem] = score
                except Exception as e:
                    print(f"[WARN] Failed to score {img_path.name}: {e}")
                    scores[img_path.stem] = float("nan")
        else:
            # Batched inference: parallel image loading + GPU batch scoring
            io_workers = min(batch_size, 4)
            for batch_start in tqdm(range(0, len(image_files), batch_size),
                                    desc=f"{metric_name.upper()} batched (bs={batch_size})",
                                    dynamic_ncols=True):
                batch_paths = image_files[batch_start:batch_start + batch_size]
                try:
                    with ThreadPoolExecutor(max_workers=io_workers) as tp:
                        tensors = list(tp.map(
                            lambda p: _load_image_tensor(p), batch_paths
                        ))
                    batch_tensor = torch.stack(tensors, dim=0).to(device)  # (B, 3, H, W)
                    batch_scores = model(batch_tensor)  # (B,) or (B, 1)
                    batch_scores = batch_scores.view(-1).cpu().tolist()
                    for path, score in zip(batch_paths, batch_scores):
                        scores[path.stem] = score
                    del tensors, batch_tensor
                except Exception as e:
                    # Fallback: score this batch sequentially (e.g. size mismatch)
                    print(f"[WARN] Batch scoring failed ({e}), falling back to sequential")
                    for path in batch_paths:
                        try:
                            score = model(str(path)).item()
                            scores[path.stem] = score
                        except Exception as e2:
                            print(f"[WARN] Failed to score {path.name}: {e2}")
                            scores[path.stem] = float("nan")

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
