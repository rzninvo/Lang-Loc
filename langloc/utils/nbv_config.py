"""
NBV Pipeline Configuration.

This module provides a dataclass and extraction function for NBV pipeline
configuration parameters, reducing boilerplate in the main pipeline scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class NBVConfig:
    """
    Configuration parameters for the NBV (Next-Best-View) pipeline.

    This dataclass consolidates all the configuration knobs used by both
    ScanNet and 3RScan processing pipelines into a single, type-safe structure.

    Attributes:
        # Rasterization settings
        image_downsample_factor: Downsampling factor for visibility pass.
        subsample_factor: Frame subsampling factor (take every Nth frame).
        faces_per_pixel: Number of faces to keep per pixel during rasterization.
        bin_size: Tiling bin size for rasterizer (None = auto).
        max_faces_per_bin: Maximum faces per bin (None = auto).
        blur_radius: Soft rasterization blur radius (0 = hard).
        limit_images: Maximum number of frames to process (None = all).

        # NBV selection
        max_best: Maximum number of best views to select (None = unlimited).
        min_gain_pixels: Minimum pixel gain to continue NBV selection.
        kmeans_n_clusters: Number of clusters for K-means pose clustering.
        iqa_metric: IQA metric to use (e.g., "qualiclip", "brisque").
        iqa_threshold: Quality threshold (interpretation depends on metric).
        iqa_device: Device for IQA model ("cuda" or "cpu").

        # Object visibility thresholds
        coverage_threshold: Minimum coverage (0-1) for object visibility.
        min_pixel_count: Minimum absolute pixel count for object visibility.
        min_obj_pixels_for_presence: Min pixels to count object as "present".

        # FOV and depth settings
        fov_depth_clip_min: Minimum depth (meters) for object visibility.
        fov_depth_clip_max: Maximum depth (meters) for object visibility.

        # NBV algorithm parameters
        nbv_alpha: Balance between coverage (1.0) and diversity (0.0).
        nbv_min_position_distance: Min distance (m) between selected views.
        nbv_min_angle_distance: Min angle (degrees) between selected views.
        nbv_enable_pose_filtering: Enable spatial diversity filtering.

        # DPP selection parameters
        dpp_enabled: Enable DPP-based view selection.
        dpp_total_views: Final output count (Stage 2 target).
        dpp_seed_size: Stage-1 semantic DPP seed count.
        dpp_clip_model: CLIP model architecture for semantic similarity.
        dpp_clip_device: Device for CLIP model inference.
        dpp_stage1_total_views: Stage-1 semantic DPP output size.
        dpp_stage2_sigma_position: Position RBF sigma (meters).
        dpp_stage2_sigma_angle: Angle RBF sigma (degrees).
        dpp_stage2_iou_gamma: Exponent on pixel IoU for Stage 2 (< 1 softens).

        # Spatial relations parameters
        spatial_max_distance: Max distance (m) for spatial relations.
        spatial_size_ratio_threshold: Max size ratio for spatial relations.
        spatial_eps: Min displacement (m) for directional relations.

        # Scene-level graph generation
        build_scene_graph: Whether to build a 3DSSG-compatible scene graph.
        scene_graph_max_distance: Max centroid distance (m) for scene graph edges.
        scene_graph_add_embeddings: Whether to add word2vec/clip embeddings.
        scene_graph_embedding_type: Embedding type ("word2vec" or "clip").
        gravity_axis: Scene gravity convention ("y_up" for ScanNet, "z_up" for 3RScan).

        # Parallelism / batching
        iqa_batch_size: Batch size for IQA model inference (1 = sequential).
        rasterization_batch_size: Batch size for PyTorch3D GPU rasterization (1 = sequential).

        # Mask export settings
        mask_downsample_factor: Downsampling factor for mask export.
        semantic_id_key: TSV column for semantic IDs (ScanNet only).
        labelmap_tsv: Path to label map TSV file (ScanNet only).

        # Output directories (relative to output_dir)
        cache_dir: Cache directory name.
        raster_out_dir: Raster output directory name.
        output_folder: Main output folder name.
    """

    # Rasterization settings
    image_downsample_factor: int = 2
    subsample_factor: int = 5
    faces_per_pixel: int = 1
    bin_size: Optional[int] = None
    max_faces_per_bin: Optional[int] = None
    blur_radius: float = 0.0
    limit_images: Optional[int] = None

    # NBV selection
    max_best: Optional[int] = None
    min_gain_pixels: int = 0
    kmeans_n_clusters: int = 10
    iqa_metric: str = "qualiclip"
    iqa_threshold: float = 0.4
    iqa_device: str = "cuda"

    # Object visibility thresholds
    coverage_threshold: float = 0.05
    min_pixel_count: int = 50
    min_obj_pixels_for_presence: int = 100

    # FOV and depth settings
    fov_depth_clip_min: float = 0.2
    fov_depth_clip_max: float = 10.0

    # Depth-aware visibility
    depth_visibility_enabled: bool = False
    depth_vis_threshold: float = 0.20
    depth_consistent_ratio_threshold: float = 0.0
    min_depth_consistent_pixels: int = 0

    # NBV algorithm parameters
    nbv_alpha: float = 0.5
    nbv_min_position_distance: float = 0.0
    nbv_min_angle_distance: float = 0.0
    nbv_enable_pose_filtering: bool = False

    # DPP selection parameters
    dpp_enabled: bool = False
    dpp_total_views: int = 10
    dpp_seed_size: int = 4
    dpp_clip_model: str = "ViT-B/32"
    dpp_clip_device: str = "cuda"
    dpp_stage1_total_views: int = 25
    dpp_stage2_sigma_position: float = 0.75
    dpp_stage2_sigma_angle: float = 20.0
    dpp_stage2_iou_gamma: float = 1.0

    # Spatial relations parameters
    spatial_max_distance: float = 2.0
    spatial_size_ratio_threshold: float = 5.0
    spatial_eps: float = 0.1
    spatial_max_surface_distance: float = 1.5

    # Scene-level graph generation
    build_scene_graph: bool = True
    scene_graph_max_distance: float = 2.0
    scene_graph_add_embeddings: bool = True
    scene_graph_embedding_type: str = "word2vec"
    gravity_axis: str = "y_up"

    # Parallelism / batching
    iqa_batch_size: int = 1
    rasterization_batch_size: int = 1

    # Mask export settings
    mask_downsample_factor: int = 1
    semantic_id_key: str = "nyu40id"
    labelmap_tsv: Path = field(default_factory=lambda: Path("data/scannetv2-labels.combined.tsv"))

    # Output directories
    cache_dir: str = "cache"
    raster_out_dir: str = "raster"
    output_folder: str = "output"

    @property
    def fov_depth_clip(self) -> tuple[float, float]:
        """Return depth clip as a tuple for convenience."""
        return (self.fov_depth_clip_min, self.fov_depth_clip_max)


def extract_nbv_config(cfg: Dict[str, Any], dataset: str = "scannetpp") -> NBVConfig:
    """
    Extract NBV configuration from a loaded YAML config dictionary.

    This function reads the dataset-specific section of the config and
    returns a typed NBVConfig dataclass with all parameters.

    Args:
        cfg: Loaded configuration dictionary (from load_config()).
        dataset: Dataset key in the config ('scannetpp' or '3rscan').

    Returns:
        NBVConfig dataclass with all extracted parameters.

    Example:
        >>> cfg = load_config('configs/dataset/default.yaml')
        >>> nbv_cfg = extract_nbv_config(cfg, dataset='scannetpp')
        >>> print(nbv_cfg.iqa_metric, nbv_cfg.iqa_threshold)
        qualiclip 0.35
    """
    section = cfg.get(dataset, {})

    return NBVConfig(
        # Rasterization settings
        image_downsample_factor=int(section.get("image_downsample_factor", 2)),
        subsample_factor=int(section.get("subsample_factor", 5)),
        faces_per_pixel=int(section.get("faces_per_pixel", 1)),
        bin_size=section.get("bin_size", None),
        max_faces_per_bin=section.get("max_faces_per_bin", None),
        blur_radius=float(section.get("blur_radius", 0.0)),
        limit_images=section.get("limit_images", None),

        # NBV selection
        max_best=section.get("max_best", None),
        min_gain_pixels=int(section.get("min_gain_pixels", 0)),
        kmeans_n_clusters=int(section.get("kmeans_n_clusters", 10)),
        iqa_metric=str(section.get("iqa_metric", "qualiclip")),
        iqa_threshold=float(section.get("iqa_threshold", 0.4)),
        iqa_device=str(section.get("iqa_device", "cuda")),

        # Object visibility thresholds
        coverage_threshold=float(section.get("coverage_threshold", 0.05)),
        min_pixel_count=int(section.get("min_pixel_count", 50)),
        min_obj_pixels_for_presence=int(section.get("min_obj_pixels_for_presence", 100)),

        # FOV and depth settings
        fov_depth_clip_min=float(section.get("fov_depth_clip_min", 0.2)),
        fov_depth_clip_max=float(section.get("fov_depth_clip_max", 10.0)),

        # Depth-aware visibility
        depth_visibility_enabled=bool(section.get("depth_visibility_enabled", False)),
        depth_vis_threshold=float(section.get("depth_vis_threshold", 0.20)),
        depth_consistent_ratio_threshold=float(section.get("depth_consistent_ratio_threshold", 0.0)),
        min_depth_consistent_pixels=int(section.get("min_depth_consistent_pixels", 0)),

        # NBV algorithm parameters
        nbv_alpha=float(section.get("nbv_alpha", 0.5)),
        nbv_min_position_distance=float(section.get("nbv_min_position_distance", 0.0)),
        nbv_min_angle_distance=float(section.get("nbv_min_angle_distance", 0.0)),
        nbv_enable_pose_filtering=bool(section.get("nbv_enable_pose_filtering", False)),

        # DPP selection parameters
        dpp_enabled=bool(section.get("dpp_enabled", False)),
        dpp_total_views=int(section.get("dpp_total_views", 10)),
        dpp_seed_size=int(section.get("dpp_seed_size", 4)),
        dpp_clip_model=str(section.get("dpp_clip_model", "ViT-B/32")),
        dpp_clip_device=str(section.get("dpp_clip_device", "cuda")),
        dpp_stage1_total_views=int(section.get("dpp_stage1_total_views", 25)),
        dpp_stage2_sigma_position=float(section.get("dpp_stage2_sigma_position", 0.75)),
        dpp_stage2_sigma_angle=float(section.get("dpp_stage2_sigma_angle", 20.0)),
        dpp_stage2_iou_gamma=float(section.get("dpp_stage2_iou_gamma", 1.0)),

        # Spatial relations parameters
        spatial_max_distance=float(section.get("spatial_max_distance", 2.0)),
        spatial_size_ratio_threshold=float(section.get("spatial_size_ratio_threshold", 5.0)),
        spatial_eps=float(section.get("spatial_eps", 0.1)),
        spatial_max_surface_distance=float(section.get("spatial_max_surface_distance", 1.5)),

        # Scene-level graph generation
        build_scene_graph=bool(section.get("build_scene_graph", True)),
        scene_graph_max_distance=float(section.get("scene_graph_max_distance", 2.0)),
        scene_graph_add_embeddings=bool(section.get("scene_graph_add_embeddings", True)),
        scene_graph_embedding_type=str(section.get("scene_graph_embedding_type", "word2vec")),
        gravity_axis=str(section.get("gravity_axis", "y_up")),

        # Parallelism / batching
        iqa_batch_size=int(section.get("iqa_batch_size", 1)),
        rasterization_batch_size=int(section.get("rasterization_batch_size", 1)),

        # Mask export settings
        mask_downsample_factor=int(section.get("mask_downsample_factor", 1)),
        semantic_id_key=str(section.get("semantic_id_key", "nyu40id")),
        labelmap_tsv=Path(section.get("labelmap_tsv", "data/scannetv2-labels.combined.tsv")),

        # Output directories
        cache_dir=str(section.get("cache_dir", "cache")),
        raster_out_dir=str(section.get("raster_out_dir", "raster")),
        output_folder=str(section.get("output_folder", "output")),
    )
