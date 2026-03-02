"""Localization subpackage for dense-grid camera localization.

Public API re-exports for convenient access::

    from langloc.localization import evaluate_scene, EvalMode, SceneMetrics
"""
from langloc.localization.evaluation import EvalMode, evaluate_scene, run_evaluation
from langloc.localization.metrics import SceneMetrics
from langloc.localization.grid import load_scene, sample_grid, first_hit_is_object
from langloc.localization.matching import topk_matched_objects
from langloc.localization.pipeline import run_loc_pipeline
from langloc.localization.prediction import select_prediction_point
from langloc.localization.frame_io import FrameSelection

__all__ = [
    "EvalMode",
    "evaluate_scene",
    "run_evaluation",
    "SceneMetrics",
    "FrameSelection",
    "load_scene",
    "sample_grid",
    "first_hit_is_object",
    "topk_matched_objects",
    "run_loc_pipeline",
    "select_prediction_point",
]
