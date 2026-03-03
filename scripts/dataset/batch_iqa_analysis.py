#!/usr/bin/env python3
"""
Batch IQA (Image Quality Assessment) Analysis Script.

Downloads scenes from ScanNet or 3RScan, computes IQA scores (QualiCLIP by default),
aggregates statistics across all scenes, and optionally removes scene data after processing.

Usage:
    # Analyze 10 scenes from 3RScan default source
    python scripts/batch_iqa_analysis.py --dataset 3RScan --num-scenes 10

    # Analyze all ScanScribe scenes, keep data after processing
    python scripts/batch_iqa_analysis.py --dataset 3RScan --source scanscribe --keep-data

    # Analyze 5 ScanNet scenes with qualiclip+ metric
    python scripts/batch_iqa_analysis.py --dataset scannet --num-scenes 5 --metric qualiclip+

    # Resume from a previous run (skip already processed scenes)
    python scripts/batch_iqa_analysis.py --dataset 3RScan --num-scenes 50 --resume

Requirements:
    pip install pyiqa torch pyyaml tqdm matplotlib
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from langloc.utils.config_loader import load_config
from tqdm import tqdm


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SceneStats:
    """Statistics for a single scene."""
    scene_id: str
    num_images: int
    min_score: float
    max_score: float
    mean_score: float
    std_score: float
    q1: float  # 25th percentile
    median: float  # 50th percentile
    q3: float  # 75th percentile
    p10: float  # 10th percentile
    p90: float  # 90th percentile
    iqr: float  # Interquartile range
    scores: List[float] = field(default_factory=list)  # Optional: all scores

    def to_dict(self, include_scores: bool = False) -> dict:
        d = asdict(self)
        if not include_scores:
            d.pop("scores", None)
        return d


@dataclass
class AggregateStats:
    """Aggregated statistics across all scenes."""
    total_scenes: int
    total_images: int
    global_min: float
    global_max: float
    global_mean: float
    global_std: float
    global_q1: float
    global_median: float
    global_q3: float
    global_p10: float
    global_p90: float
    # Per-scene averages
    avg_scene_mean: float
    avg_scene_std: float
    # Suggested thresholds
    suggested_aggressive: float  # Keep top 25%
    suggested_moderate: float  # Keep top 50%
    suggested_permissive: float  # Keep top 75%
    suggested_very_permissive: float  # Keep top 90%


# ============================================================================
# IQA Model
# ============================================================================

def load_iqa_model(metric_name: str, device: str = "cuda"):
    """Load the IQA model from pyiqa."""
    try:
        import pyiqa
    except ImportError:
        print("[ERROR] pyiqa is not installed. Install it with:")
        print("  pip install pyiqa")
        print("or for the latest version:")
        print("  pip install git+https://github.com/chaofengc/IQA-PyTorch.git")
        sys.exit(1)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading {metric_name} model on {device}...")
    model = pyiqa.create_metric(metric_name, device=device)
    higher_better = not model.lower_better
    print(f"[INFO] {metric_name} loaded. Higher score = better quality: {higher_better}")
    return model, device, higher_better


def score_images_in_directory(
    image_dir: Path,
    model,
    device: torch.device,
    pattern: str = "*.jpg",
) -> List[Tuple[str, float]]:
    """Score all images in a directory."""
    # Find images with various patterns
    image_files = []
    for ext in [pattern, "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        image_files.extend(image_dir.glob(ext))
    image_files = sorted(set(image_files))

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
        return []

    results = []
    for img_path in tqdm(image_files, desc=f"Scoring {image_dir.name}", leave=False):
        try:
            score = model(str(img_path)).item()
            results.append((img_path.name, score))
        except Exception as e:
            print(f"[WARN] Failed to score {img_path.name}: {e}")
            results.append((img_path.name, float("nan")))

    return results


def compute_scene_stats(scene_id: str, scores: List[float]) -> Optional[SceneStats]:
    """Compute statistics for a single scene."""
    valid_scores = [s for s in scores if not np.isnan(s)]
    if not valid_scores:
        return None

    arr = np.array(valid_scores)
    return SceneStats(
        scene_id=scene_id,
        num_images=len(valid_scores),
        min_score=float(np.min(arr)),
        max_score=float(np.max(arr)),
        mean_score=float(np.mean(arr)),
        std_score=float(np.std(arr)),
        q1=float(np.percentile(arr, 25)),
        median=float(np.percentile(arr, 50)),
        q3=float(np.percentile(arr, 75)),
        p10=float(np.percentile(arr, 10)),
        p90=float(np.percentile(arr, 90)),
        iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        scores=valid_scores,
    )


def compute_aggregate_stats(scene_stats_list: List[SceneStats]) -> AggregateStats:
    """Compute aggregate statistics across all scenes."""
    # Gather all scores
    all_scores = []
    for ss in scene_stats_list:
        all_scores.extend(ss.scores)

    arr = np.array(all_scores)

    # Per-scene averages
    scene_means = [ss.mean_score for ss in scene_stats_list]
    scene_stds = [ss.std_score for ss in scene_stats_list]

    return AggregateStats(
        total_scenes=len(scene_stats_list),
        total_images=len(all_scores),
        global_min=float(np.min(arr)),
        global_max=float(np.max(arr)),
        global_mean=float(np.mean(arr)),
        global_std=float(np.std(arr)),
        global_q1=float(np.percentile(arr, 25)),
        global_median=float(np.percentile(arr, 50)),
        global_q3=float(np.percentile(arr, 75)),
        global_p10=float(np.percentile(arr, 10)),
        global_p90=float(np.percentile(arr, 90)),
        avg_scene_mean=float(np.mean(scene_means)),
        avg_scene_std=float(np.mean(scene_stds)),
        # For higher-is-better metrics, thresholds are minimums
        suggested_aggressive=float(np.percentile(arr, 75)),  # Keep top 25%
        suggested_moderate=float(np.percentile(arr, 50)),  # Keep top 50%
        suggested_permissive=float(np.percentile(arr, 25)),  # Keep top 75%
        suggested_very_permissive=float(np.percentile(arr, 10)),  # Keep top 90%
    )


# ============================================================================
# Scene Management
# ============================================================================

def get_scene_ids(
    dataset: str,
    source: str,
    num_scenes: int | str,
    cfg: dict,
) -> List[str]:
    """Get list of scene IDs to process."""
    scene_ids = []

    if dataset == "scannet":
        if num_scenes == "all":
            # ScanNet has ~1500 scenes, default to 100 for safety
            num_scenes = 100
            print(f"[WARN] 'all' for ScanNet defaults to {num_scenes} scenes")
        for i in range(int(num_scenes)):
            scene_ids.append(f"scene{i:04d}_00")

    elif dataset == "3RScan":
        partial_file_raw = cfg["paths"].get("rscan_partial_scans", "")
        partial_file = Path(partial_file_raw) if partial_file_raw else None
        partial_ids = set()
        if partial_file is not None:
            if partial_file.exists() and partial_file.is_file():
                with open(partial_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "#" in line:
                            line = line.split("#", 1)[0].strip()
                        if line:
                            partial_ids.add(line)
            else:
                print(f"[WARN] 3RScan partial file not found: {partial_file}")

        if source == "scanscribe":
            scanscribe_file = Path(cfg["paths"].get("scanscribe_manifest", "configs/manifests/scanscribe_cleaned.json"))
            if not scanscribe_file.exists():
                print(f"[ERROR] ScanScribe file not found: {scanscribe_file}")
                sys.exit(1)
            with open(scanscribe_file) as f:
                data = json.load(f)
            all_ids = list(data.keys())
            if partial_ids:
                before = len(all_ids)
                all_ids = [sid for sid in all_ids if sid not in partial_ids]
                removed = before - len(all_ids)
                print(f"[INFO] Filtered {removed} partial scans from ScanScribe source")
            if num_scenes == "all":
                scene_ids = all_ids
            else:
                scene_ids = all_ids[:int(num_scenes)]
            print(f"[INFO] Using ScanScribe dataset: {len(scene_ids)} scenes")
        else:
            rscan_file = Path(cfg["paths"].get("rscan_release_scans", ""))
            if not rscan_file.exists():
                print(f"[ERROR] 3RScan release file not found: {rscan_file}")
                sys.exit(1)
            with open(rscan_file) as f:
                all_ids = [line.strip() for line in f if line.strip()]
            if partial_ids:
                before = len(all_ids)
                all_ids = [sid for sid in all_ids if sid not in partial_ids]
                removed = before - len(all_ids)
                print(f"[INFO] Filtered {removed} partial scans from default 3RScan list")
            if num_scenes == "all":
                scene_ids = all_ids
            else:
                scene_ids = all_ids[:int(num_scenes)]
            print(f"[INFO] Using default 3RScan release: {len(scene_ids)} scenes")

    return scene_ids


def get_image_directory(dataset: str, scene_id: str, cfg: dict) -> Path:
    """Get the image directory for a scene."""
    if dataset == "scannet":
        return Path(cfg["paths"]["scannet_root"]) / scene_id / "color"
    else:  # 3RScan
        return Path(cfg["paths"]["rscan_root"]) / scene_id


def get_image_pattern(dataset: str) -> str:
    """Get the glob pattern for images."""
    if dataset == "scannet":
        return "*.jpg"
    else:  # 3RScan
        return "frame-*.color.jpg"


def download_scene(dataset: str, scene_id: str) -> bool:
    """Download a scene using the existing download script."""
    print(f"[INFO] Downloading {scene_id}...")
    try:
        result = subprocess.run(
            ["bash", "scripts/download_subset.sh", "--dataset", dataset, scene_id],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode != 0:
            print(f"[ERROR] Download failed for {scene_id}: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Download timed out for {scene_id}")
        return False
    except Exception as e:
        print(f"[ERROR] Download failed for {scene_id}: {e}")
        return False


def delete_scene(dataset: str, scene_id: str, cfg: dict):
    """Delete a scene's data directory."""
    if dataset == "scannet":
        scene_dir = Path(cfg["paths"]["scannet_root"]) / scene_id
    else:
        scene_dir = Path(cfg["paths"]["rscan_root"]) / scene_id

    if scene_dir.exists():
        print(f"[INFO] Deleting scene data: {scene_dir}")
        shutil.rmtree(scene_dir)


def scene_exists(dataset: str, scene_id: str, cfg: dict) -> bool:
    """Check if a scene's data already exists."""
    image_dir = get_image_directory(dataset, scene_id, cfg)
    return image_dir.exists()


# ============================================================================
# Results Management
# ============================================================================

def save_results(
    output_dir: Path,
    metric_name: str,
    dataset: str,
    scene_stats_list: List[SceneStats],
    aggregate_stats: AggregateStats,
    all_scores: List[Tuple[str, str, float]],  # (scene_id, filename, score)
):
    """Save all results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Per-scene summary CSV
    scene_csv = output_dir / f"scene_stats_{metric_name}_{dataset}_{timestamp}.csv"
    with open(scene_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_id", "num_images", "min", "max", "mean", "std",
            "q1", "median", "q3", "p10", "p90", "iqr"
        ])
        for ss in scene_stats_list:
            writer.writerow([
                ss.scene_id, ss.num_images, ss.min_score, ss.max_score,
                ss.mean_score, ss.std_score, ss.q1, ss.median, ss.q3,
                ss.p10, ss.p90, ss.iqr
            ])
    print(f"[INFO] Scene statistics saved to: {scene_csv}")

    # 2. All scores CSV (for detailed analysis)
    scores_csv = output_dir / f"all_scores_{metric_name}_{dataset}_{timestamp}.csv"
    with open(scores_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "filename", "score"])
        for scene_id, filename, score in all_scores:
            writer.writerow([scene_id, filename, score])
    print(f"[INFO] All scores saved to: {scores_csv}")

    # 3. Aggregate statistics JSON
    agg_json = output_dir / f"aggregate_stats_{metric_name}_{dataset}_{timestamp}.json"
    with open(agg_json, "w") as f:
        json.dump(asdict(aggregate_stats), f, indent=2)
    print(f"[INFO] Aggregate statistics saved to: {agg_json}")

    # 4. Summary report
    report_file = output_dir / f"report_{metric_name}_{dataset}_{timestamp}.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"IQA Batch Analysis Report - {metric_name.upper()}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("GLOBAL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total scenes analyzed: {aggregate_stats.total_scenes}\n")
        f.write(f"Total images scored:   {aggregate_stats.total_images}\n")
        f.write(f"Global min score:      {aggregate_stats.global_min:.4f}\n")
        f.write(f"Global max score:      {aggregate_stats.global_max:.4f}\n")
        f.write(f"Global mean:           {aggregate_stats.global_mean:.4f}\n")
        f.write(f"Global std:            {aggregate_stats.global_std:.4f}\n")
        f.write(f"Global P10:            {aggregate_stats.global_p10:.4f}\n")
        f.write(f"Global Q1 (25%):       {aggregate_stats.global_q1:.4f}\n")
        f.write(f"Global Median (50%):   {aggregate_stats.global_median:.4f}\n")
        f.write(f"Global Q3 (75%):       {aggregate_stats.global_q3:.4f}\n")
        f.write(f"Global P90:            {aggregate_stats.global_p90:.4f}\n")
        f.write("\n")

        f.write("SUGGESTED THRESHOLDS (higher score = better quality)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Aggressive (keep top 25%):      >= {aggregate_stats.suggested_aggressive:.4f}\n")
        f.write(f"Moderate (keep top 50%):        >= {aggregate_stats.suggested_moderate:.4f}\n")
        f.write(f"Permissive (keep top 75%):      >= {aggregate_stats.suggested_permissive:.4f}\n")
        f.write(f"Very permissive (keep top 90%): >= {aggregate_stats.suggested_very_permissive:.4f}\n")
        f.write("\n")

        f.write("PER-SCENE SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average scene mean: {aggregate_stats.avg_scene_mean:.4f}\n")
        f.write(f"Average scene std:  {aggregate_stats.avg_scene_std:.4f}\n")
        f.write("\n")

        # Top 5 best and worst scenes by mean
        sorted_scenes = sorted(scene_stats_list, key=lambda x: x.mean_score)
        f.write("Worst 5 scenes (lowest mean scores):\n")
        for ss in sorted_scenes[:5]:
            f.write(f"  {ss.scene_id}: mean={ss.mean_score:.4f}, n={ss.num_images}\n")
        f.write("\n")
        f.write("Best 5 scenes (highest mean scores):\n")
        for ss in sorted_scenes[-5:]:
            f.write(f"  {ss.scene_id}: mean={ss.mean_score:.4f}, n={ss.num_images}\n")

    print(f"[INFO] Report saved to: {report_file}")

    return report_file


def load_previous_results(output_dir: Path, metric_name: str, dataset: str) -> set:
    """Load scene IDs that have already been processed."""
    processed = set()
    for csv_file in output_dir.glob(f"scene_stats_{metric_name}_{dataset}_*.csv"):
        try:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed.add(row["scene_id"])
        except Exception:
            pass
    return processed


def plot_global_distribution(
    all_scores: List[float],
    metric_name: str,
    output_path: Path,
):
    """Plot global score distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping plot.")
        return

    scores = np.array([s for s in all_scores if not np.isnan(s)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(scores, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(np.median(scores), color="red", linestyle="--", linewidth=2,
                label=f"Median: {np.median(scores):.3f}")
    ax1.axvline(np.percentile(scores, 25), color="orange", linestyle=":", linewidth=2,
                label=f"Q1: {np.percentile(scores, 25):.3f}")
    ax1.axvline(np.percentile(scores, 75), color="green", linestyle=":", linewidth=2,
                label=f"Q3: {np.percentile(scores, 75):.3f}")
    ax1.set_xlabel(f"{metric_name} Score (higher = better)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Global {metric_name} Score Distribution\n({len(scores):,} images)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(scores, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    ax2.set_ylabel(f"{metric_name} Score")
    ax2.set_title("Score Distribution (Box Plot)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch IQA analysis across multiple scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze 10 scenes from 3RScan
    python scripts/batch_iqa_analysis.py --dataset 3RScan --num-scenes 10

    # Analyze all ScanScribe scenes, keep data
    python scripts/batch_iqa_analysis.py --dataset 3RScan --source scanscribe --keep-data

    # Analyze ScanNet scenes with qualiclip+
    python scripts/batch_iqa_analysis.py --dataset scannet --num-scenes 5 --metric qualiclip+
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["scannet", "3RScan"],
        help="Dataset to analyze.",
    )
    parser.add_argument(
        "--num-scenes",
        type=str,
        default="10",
        help="Number of scenes to process, or 'all'.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="default",
        choices=["default", "scanscribe"],
        help="Source of scene IDs (for 3RScan).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="qualiclip",
        choices=["qualiclip", "qualiclip+", "qualiclip+-clive", "qualiclip+-flive", "qualiclip+-spaq", "brisque", "niqe", "musiq"],
        help="IQA metric to use.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep scene data after processing (default: delete to save space).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/iqa_analysis",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, skipping already processed scenes.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading scenes (assume data already exists).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate distribution plots.",
    )

    args = parser.parse_args()

    # Load project config from Hydra config tree
    cfg = load_config()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get scene IDs
    scene_ids = get_scene_ids(args.dataset, args.source, args.num_scenes, cfg)
    print(f"[INFO] Will process {len(scene_ids)} scenes from {args.dataset}")

    # Check for resume
    already_processed = set()
    if args.resume:
        already_processed = load_previous_results(output_dir, args.metric, args.dataset)
        if already_processed:
            print(f"[INFO] Resuming: {len(already_processed)} scenes already processed")

    # Load IQA model
    model, device, higher_better = load_iqa_model(args.metric, args.device)

    # Process scenes
    scene_stats_list: List[SceneStats] = []
    all_scores: List[Tuple[str, str, float]] = []  # (scene_id, filename, score)
    failed_scenes: List[str] = []
    pattern = get_image_pattern(args.dataset)

    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        # Skip if already processed
        if scene_id in already_processed:
            print(f"[SKIP] {scene_id} already processed")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {scene_id}")
        print(f"{'='*60}")

        # Download if needed
        if not args.skip_download:
            if not scene_exists(args.dataset, scene_id, cfg):
                success = download_scene(args.dataset, scene_id)
                if not success:
                    failed_scenes.append(scene_id)
                    continue

        # Get image directory
        image_dir = get_image_directory(args.dataset, scene_id, cfg)
        if not image_dir.exists():
            print(f"[WARN] Image directory not found: {image_dir}")
            failed_scenes.append(scene_id)
            continue

        # Score images
        results = score_images_in_directory(image_dir, model, device, pattern)
        if not results:
            print(f"[WARN] No images found in {image_dir}")
            failed_scenes.append(scene_id)
            continue

        # Compute stats
        scores = [score for _, score in results]
        stats = compute_scene_stats(scene_id, scores)
        if stats:
            scene_stats_list.append(stats)
            # Add to all_scores
            for filename, score in results:
                if not np.isnan(score):
                    all_scores.append((scene_id, filename, score))

            print(f"[OK] {scene_id}: n={stats.num_images}, mean={stats.mean_score:.4f}, "
                  f"median={stats.median:.4f}, range=[{stats.min_score:.4f}, {stats.max_score:.4f}]")
        else:
            print(f"[WARN] No valid scores for {scene_id}")
            failed_scenes.append(scene_id)

        # Delete scene data if requested
        if not args.keep_data and not args.skip_download:
            delete_scene(args.dataset, scene_id, cfg)

    # Compute aggregate stats
    if scene_stats_list:
        aggregate = compute_aggregate_stats(scene_stats_list)

        # Print summary
        print("\n" + "=" * 70)
        print(f"AGGREGATE RESULTS - {args.metric.upper()}")
        print("=" * 70)
        print(f"Scenes processed:      {aggregate.total_scenes}")
        print(f"Total images:          {aggregate.total_images:,}")
        print(f"Failed scenes:         {len(failed_scenes)}")
        print("-" * 70)
        print(f"Global min:            {aggregate.global_min:.4f}")
        print(f"Global max:            {aggregate.global_max:.4f}")
        print(f"Global mean:           {aggregate.global_mean:.4f}")
        print(f"Global std:            {aggregate.global_std:.4f}")
        print("-" * 70)
        print(f"Global P10:            {aggregate.global_p10:.4f}")
        print(f"Global Q1 (25%):       {aggregate.global_q1:.4f}")
        print(f"Global Median (50%):   {aggregate.global_median:.4f}")
        print(f"Global Q3 (75%):       {aggregate.global_q3:.4f}")
        print(f"Global P90:            {aggregate.global_p90:.4f}")
        print("=" * 70)
        print("\nSUGGESTED THRESHOLDS (higher = better):")
        print("-" * 70)
        print(f"Aggressive (keep top 25%):      >= {aggregate.suggested_aggressive:.4f}")
        print(f"Moderate (keep top 50%):        >= {aggregate.suggested_moderate:.4f}")
        print(f"Permissive (keep top 75%):      >= {aggregate.suggested_permissive:.4f}")
        print(f"Very permissive (keep top 90%): >= {aggregate.suggested_very_permissive:.4f}")
        print("=" * 70)

        # Save results
        report_path = save_results(
            output_dir, args.metric, args.dataset,
            scene_stats_list, aggregate, all_scores
        )

        # Generate plot if requested
        if args.plot:
            plot_path = output_dir / f"distribution_{args.metric}_{args.dataset}.png"
            plot_global_distribution(
                [s for _, _, s in all_scores],
                args.metric,
                plot_path,
            )

        # Print failed scenes
        if failed_scenes:
            print(f"\n[WARN] Failed scenes ({len(failed_scenes)}):")
            for sid in failed_scenes[:10]:
                print(f"  - {sid}")
            if len(failed_scenes) > 10:
                print(f"  ... and {len(failed_scenes) - 10} more")

        print(f"\n[DONE] Results saved to: {output_dir}")
    else:
        print("\n[ERROR] No scenes were successfully processed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
