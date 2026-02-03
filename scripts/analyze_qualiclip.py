#!/usr/bin/env python3
"""
QualiCLIP Score Analysis Script for EDA.

This script analyzes image quality scores using QualiCLIP (from pyiqa) on all images
in a directory. It computes quartile statistics to help determine appropriate thresholds
for image quality filtering in the NBV pipeline.

Usage:
    # Analyze a single scene's images
    python scripts/analyze_qualiclip.py /path/to/scene/color

    # Analyze with visualization
    python scripts/analyze_qualiclip.py /path/to/scene/color --plot

    # Analyze with specific file pattern
    python scripts/analyze_qualiclip.py /path/to/scene/color --pattern "*.png"

    # Use a different metric (e.g., qualiclip+)
    python scripts/analyze_qualiclip.py /path/to/scene/color --metric qualiclip+

Requirements:
    pip install pyiqa matplotlib
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_qualiclip_model(metric_name: str = "qualiclip", device: str = "cuda"):
    """Load the QualiCLIP model from pyiqa."""
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
    print(f"[INFO] {metric_name} loaded. Higher score = better quality: {not model.lower_better}")
    return model, device


def score_images(
    image_dir: Path,
    model,
    device: torch.device,
    pattern: str = "*.jpg",
) -> List[Tuple[str, float]]:
    """
    Score all images in a directory using the given IQA model.

    Returns:
        List of (filename, score) tuples.
    """
    image_files = sorted(image_dir.glob(pattern))
    if not image_files:
        # Try common image extensions
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
            image_files = sorted(image_dir.glob(ext))
            if image_files:
                print(f"[INFO] Found {len(image_files)} images with pattern {ext}")
                break

    # Filter to only include color images (exclude depth images and mesh textures)
    filtered_files = []
    for img_path in image_files:
        filename = img_path.name
        # Include only files with ".color." in the name (e.g., frame-000000.color.jpg)
        # Exclude mesh textures (e.g., mesh.refined_0.png) and depth images
        if ".color." in filename:
            filtered_files.append(img_path)

    image_files = filtered_files

    if not image_files:
        raise ValueError(f"No color images found in {image_dir} with pattern {pattern}")

    print(f"[INFO] Scoring {len(image_files)} images...")
    results = []

    for img_path in tqdm(image_files, desc="QualiCLIP scoring"):
        try:
            # pyiqa can take file paths directly
            score = model(str(img_path)).item()
            results.append((img_path.name, score))
        except Exception as e:
            print(f"[WARN] Failed to score {img_path.name}: {e}")
            results.append((img_path.name, float("nan")))

    return results


def compute_statistics(scores: List[float]) -> dict:
    """Compute comprehensive statistics from scores."""
    scores = np.array([s for s in scores if not np.isnan(s)])

    if len(scores) == 0:
        return {"error": "No valid scores"}

    stats = {
        "count": len(scores),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "Q1 (25%)": float(np.percentile(scores, 25)),
        "Q2 (50%/median)": float(np.percentile(scores, 50)),
        "Q3 (75%)": float(np.percentile(scores, 75)),
        "Q4 (max)": float(np.max(scores)),
        "P10": float(np.percentile(scores, 10)),
        "P90": float(np.percentile(scores, 90)),
        "IQR": float(np.percentile(scores, 75) - np.percentile(scores, 25)),
    }
    return stats


def plot_distribution(scores: List[float], output_path: Path = None, title: str = "QualiCLIP Score Distribution"):
    """Plot histogram and box plot of scores."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping plot.")
        return

    scores = np.array([s for s in scores if not np.isnan(s)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(scores, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(np.median(scores), color="red", linestyle="--", linewidth=2, label=f"Median: {np.median(scores):.3f}")
    ax1.axvline(np.percentile(scores, 25), color="orange", linestyle=":", linewidth=2, label=f"Q1: {np.percentile(scores, 25):.3f}")
    ax1.axvline(np.percentile(scores, 75), color="green", linestyle=":", linewidth=2, label=f"Q3: {np.percentile(scores, 75):.3f}")
    ax1.set_xlabel("QualiCLIP Score (higher = better)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(scores, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    ax2.set_ylabel("QualiCLIP Score")
    ax2.set_title("Score Distribution (Box Plot)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Plot saved to {output_path}")
    else:
        plt.show()


def find_threshold_candidates(stats: dict) -> dict:
    """Suggest threshold candidates based on statistics."""
    suggestions = {
        "aggressive (keep top 25%)": stats["Q3 (75%)"],
        "moderate (keep top 50%)": stats["Q2 (50%/median)"],
        "permissive (keep top 75%)": stats["Q1 (25%)"],
        "very_permissive (keep top 90%)": stats["P10"],
    }
    return suggestions


def main():
    parser = argparse.ArgumentParser(
        description="Analyze image quality scores using QualiCLIP for EDA."
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing images to analyze.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jpg",
        help="Glob pattern for image files (default: *.jpg).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="qualiclip",
        choices=["qualiclip", "qualiclip+", "qualiclip+-clive", "qualiclip+-flive", "qualiclip+-spaq"],
        help="QualiCLIP variant to use (default: qualiclip).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a histogram and box plot of the scores.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Save plot to this path instead of displaying.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Save per-image scores to a CSV file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu).",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    # Load model and score images
    model, device = load_qualiclip_model(args.metric, args.device)
    results = score_images(image_dir, model, device, args.pattern)

    # Extract scores
    scores = [score for _, score in results]

    # Compute and print statistics
    stats = compute_statistics(scores)

    print("\n" + "=" * 60)
    print(f"QualiCLIP ({args.metric}) Score Statistics")
    print("=" * 60)
    print(f"Directory: {image_dir}")
    print(f"Images analyzed: {stats['count']}")
    print("-" * 60)
    print(f"Min:             {stats['min']:.4f}")
    print(f"Max:             {stats['max']:.4f}")
    print(f"Mean:            {stats['mean']:.4f}")
    print(f"Std Dev:         {stats['std']:.4f}")
    print("-" * 60)
    print(f"P10 (10%):       {stats['P10']:.4f}")
    print(f"Q1 (25%):        {stats['Q1 (25%)']:.4f}")
    print(f"Q2 (50%/median): {stats['Q2 (50%/median)']:.4f}")
    print(f"Q3 (75%):        {stats['Q3 (75%)']:.4f}")
    print(f"P90 (90%):       {stats['P90']:.4f}")
    print(f"IQR:             {stats['IQR']:.4f}")
    print("=" * 60)

    # Threshold suggestions
    suggestions = find_threshold_candidates(stats)
    print("\nSuggested Thresholds (QualiCLIP: higher = better):")
    print("-" * 60)
    for name, threshold in suggestions.items():
        print(f"  {name}: >= {threshold:.4f}")
    print("=" * 60)

    # Show worst and best images
    results_sorted = sorted(results, key=lambda x: x[1] if not np.isnan(x[1]) else float("-inf"))
    print("\nWorst 5 images (lowest scores):")
    for name, score in results_sorted[:5]:
        print(f"  {name}: {score:.4f}")

    print("\nBest 5 images (highest scores):")
    for name, score in results_sorted[-5:]:
        print(f"  {name}: {score:.4f}")

    # Optional: save CSV
    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "qualiclip_score"])
            for name, score in results:
                writer.writerow([name, score])
        print(f"\n[INFO] Scores saved to {args.output_csv}")

    # Optional: plot
    if args.plot or args.save_plot:
        plot_output = Path(args.save_plot) if args.save_plot else None
        plot_distribution(scores, plot_output, f"QualiCLIP ({args.metric}) Score Distribution\n{image_dir.name}")


if __name__ == "__main__":
    main()
