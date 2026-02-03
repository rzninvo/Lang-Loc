"""
Image generation and view selection modules for 3D scene datasets.

This package provides next-best-view (NBV) selection algorithms and
rendering utilities for both ScanNet and 3RScan datasets.

Modules:
- nbv_pipeline: Shared NBV pipeline components (BRISQUE filtering, visibility,
  greedy selection, mask export, clustering)
- scannetpp_best_views: View selection and mask export for ScanNet scenes
- 3rscan_best_views: View selection and mask export for 3RScan scenes
"""
