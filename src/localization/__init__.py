"""Localization subpackage for dense-grid camera localization.

Public API re-exports for convenient access::

    from src.localization import evaluate_scene, EvalMode, SceneMetrics
"""
from src.localization.evaluation import EvalMode, evaluate_scene, run_evaluation
from src.localization.metrics import SceneMetrics
from src.localization.grid import load_scene, sample_grid, first_hit_is_object
from src.localization.matching import topk_matched_objects
from src.localization.pipeline import run_loc_pipeline
from src.localization.prediction import select_prediction_point
from src.localization.frame_io import FrameSelection

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
