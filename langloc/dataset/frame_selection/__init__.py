"""
Frame selection and view quality pipeline for 3D scene datasets.

This package provides modular components for selecting representative
keyframes from ScanNet++ and 3RScan datasets.

Modules:
    iqa: Image Quality Assessment (IQA) filtering using pyiqa.
    visibility: Camera setup, rasterization, and object visibility computation.
    dpp: Determinantal Point Process (DPP) based view selection (3-stage).
    legacy: Greedy next-best-view selection and K-means clustering.
    masks: Instance and semantic mask rendering and export.
    scannetpp_best_views: ScanNet++ entry-point pipeline.
    3rscan_best_views: 3RScan entry-point pipeline.
"""
