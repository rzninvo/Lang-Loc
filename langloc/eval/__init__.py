"""Cross-cutting evaluation utilities for the LangLoc pipeline.

Consolidates evaluation code that is shared across pipeline stages
(retrieval, localization, dialogue) or that operates independently.

Stage-specific evaluation harnesses remain in their respective packages:
- ``langloc.retrieval.eval``       -- Retrieval Recall@k
- ``langloc.graph_matching.eval``  -- Graph matching top-k accuracy
- ``langloc.dialogue.eval_runner`` -- Dialogue backend evaluation
- ``langloc.localization.evaluation`` -- Dense-grid localization evaluation
"""

from langloc.eval.metrics import (
    SceneMetrics,
    compute_metrics_standard,
    compute_metrics_simple,
    compute_view_iou_error,
    build_metrics_table_standard,
    build_metrics_table_simple,
)

from langloc.eval.view_iou import (
    compute_view_iou,
    build_iou_context,
)


def import_module_from_path(py_path, module_name: str):
    """Dynamically import a Python module from a file path."""
    import importlib.util
    import sys
    from pathlib import Path

    py_path = Path(py_path)
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


__all__ = [
    "SceneMetrics",
    "compute_metrics_standard",
    "compute_metrics_simple",
    "compute_view_iou_error",
    "build_metrics_table_standard",
    "build_metrics_table_simple",
    "compute_view_iou",
    "build_iou_context",
]
