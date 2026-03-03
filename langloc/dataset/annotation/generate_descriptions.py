#!/usr/bin/env python3
"""
generate_descriptions.py
------------------------
Generates natural-language descriptions for each selected keyframe
based on precomputed object visibility and spatial relations.

The descriptions are produced by querying an OpenAI GPT model (or another
LLM) and are saved in the same schema used by the human annotation platform,
including camera poses, object metadata, and timestamps.

Typical usage example:
----------------------
    python -m langloc.dataset.annotation.generate_descriptions \
        scan_id=<scene_id> dataset.target=3RScan
"""

import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import hydra
from omegaconf import DictConfig
from langloc.utils.camera_utils import load_camera_poses_json

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------

# Load the environment variables from the project root .env file.
# This file must contain an OPENAI_API_KEY entry.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    print(f"[WARN] .env file not found at {dotenv_path}")

# Initialize the OpenAI client (uses OPENAI_API_KEY from environment)
client = OpenAI()

# ---------------------------------------------------------------------
# GPT Query
# ---------------------------------------------------------------------

def call_gpt(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Query the OpenAI GPT API to generate a natural-language description
    from a text prompt summarizing object and spatial information.

    Parameters
    ----------
    prompt : str
        The text prompt describing the scene contents and relationships.
    model : str, optional
        Name of the OpenAI model to use. Default is "gpt-4o-mini".

    Returns
    -------
    str
        The model-generated description text. If the request fails,
        a placeholder string "[Error generating description]" is returned.

    Notes
    -----
    - Requires a valid `OPENAI_API_KEY` loaded from `.env` or the environment.
    - Uses `temperature=0.5` for moderate creativity while maintaining factual tone.
    - Truncates output at 500 tokens for consistency across frames.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You describe indoor camera views in natural, conversational language. "
                        "Write as if you're casually telling someone what you see, not listing facts. "
                        "Describe the scene holistically - what dominates the view, the overall layout. "
                        "Weave spatial relationships naturally into the description rather than stating them explicitly. "
                        "For example, instead of 'A chair is to the left of a table', say 'A chair sits beside a table' or simply 'The view shows a chair and table'. "
                        "If only one or two objects are visible, just describe what's there - don't mention absence of relations. "
                        "Keep it very concise: 2-3 sentences maximum. "
                        "Avoid robotic phrases like 'is positioned', 'is located', 'there is'. "
                        "No speculation about room type or purpose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,  # Slightly higher for more natural variation
            max_completion_tokens=80,  # Shorter, punchier descriptions
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT request failed: {e}")
        return "[Error generating description]"


def extract_color_hint(obj_meta: dict) -> str | None:
    """
    Extract a concise color hint from object metadata.

    Reads the literal color name from ``attributes.color`` (populated from
    objects.json during the visibility pass in 3rscan_best_views.py).
    """
    if not isinstance(obj_meta, dict):
        return None
    attrs = obj_meta.get("attributes", {})
    if not isinstance(attrs, dict):
        return None

    for key in ("color", "colour"):
        val = attrs.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    return item.strip().lower()
        elif isinstance(val, str) and val.strip():
            return val.strip().lower()

    return None


# ---------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------

# View-dependent predicate IDs whose inverses carry no new information
# for natural-language description (left↔right, front↔behind).
_INVERSE_PRED_IDS = {
    2: 3,   # left ↔ right
    3: 2,
    4: 5,   # front ↔ behind
    5: 4,
}


def _dedup_relations_for_prompt(spatial_relations: list) -> list:
    """
    Remove symmetric/inverse duplicates from spatial relations so the
    GPT prompt is concise.  Only used for prompt construction — the full
    relation list is preserved in the saved output.

    Rules:
      - Directional inverses (left↔right, front↔behind): keep whichever
        appears first; the reverse pair adds no information for a human.
      - Other predicates: keep one per unordered (subject, object, pred_id).
    """
    seen: set = set()
    deduped: list = []
    for r in spatial_relations:
        sub = r.get("subject_id", r.get("subject"))
        obj = r.get("object_id", r.get("object"))
        pred_id = r.get("predicate_id")

        if pred_id is not None and pred_id in _INVERSE_PRED_IDS:
            # Use a canonical key so left(A,B) and right(B,A) map the same
            pair = frozenset((sub, obj))
            canon = ("dir", pair)
        elif pred_id is not None:
            pair = frozenset((sub, obj))
            canon = ("rel", pair, pred_id)
        else:
            # Heuristic relations (no pred_id) — dedup by label pair + relation
            pair = frozenset((r.get("subject", ""), r.get("object", "")))
            canon = ("heur", pair, r.get("relation", ""))

        if canon in seen:
            continue
        seen.add(canon)
        deduped.append(r)
    return deduped


def build_prompt(fid: str, visible_objects: dict, spatial_relations: list) -> str:
    """
    Construct a descriptive text prompt from the cached visibility
    and spatial metadata of one keyframe.

    This structured prompt is what the GPT model receives as input.
    It lists all visible objects and describes their qualitative relations
    (e.g., "the fridge is right_of the sink").

    Parameters
    ----------
    fid : str
        The frame (image) identifier (e.g., "000123").
    visible_objects : dict
        Mapping from object ID → metadata dictionary containing at least
        a 'label' field and optionally centroid/bbox data.
    spatial_relations : list
        List of pairwise spatial relations between visible objects,
        each entry having fields {'subject', 'object', 'relation'}.

    Returns
    -------
    str
        A formatted English prompt string to be fed to the GPT model.

    """
    obj_list = [f"{v['label'] or f'object {oid}'}" for oid, v in visible_objects.items()]
    color_hints = []
    for oid, obj in visible_objects.items():
        label = (obj.get("label") or f"object {oid}").strip()
        color = extract_color_hint(obj)
        if color:
            color_hints.append(f"{label}: {color}")

    # Build a more natural prompt
    prompt_parts = [f"Describe what you see in this indoor camera view.\n"]
    prompt_parts.append(f"Visible objects: {', '.join(obj_list)}\n")
    if color_hints:
        prompt_parts.append(f"Color hints: {', '.join(color_hints[:12])}\n")

    # Only include spatial hints if there are meaningful relations AND multiple objects
    if spatial_relations and len(visible_objects) > 1:
        unique_rels = _dedup_relations_for_prompt(spatial_relations)
        relations_natural = []
        for r in unique_rels:
            subj = r['subject']
            obj = r['object']
            rel = r['relation'].replace("_", " ")
            relations_natural.append(f"{subj} {rel} {obj}")

        if relations_natural:
            prompt_parts.append(f"Layout hints: {', '.join(relations_natural)}\n")

    prompt_parts.append(
        "\nDescribe the view in 2-3 short sentences. "
        "Write naturally and conversationally, as if describing it to a friend. "
        "Focus on what's most prominent and how the space feels. "
        "Use color hints only when they sound natural and helpful."
    )

    return "".join(prompt_parts)


# ---------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------

def main(scene_id: str, dataset: str, cfg: DictConfig):
    """
    Generate GPT-based scene descriptions for all **selected keyframes** of a given scene.
    Only frames that were chosen during the NBV + clustering step (those listed in
    `output/camera_pose.json`) will be described.

    Parameters
    ----------
    scene_id : str
        Identifier of the scene (e.g., "scene0000_00" or a 3RScan UUID).
    dataset : str
        Dataset type: "scannet" or "3RScan".
    cfg : DictConfig
        Hydra DictConfig with ``dataset`` and ``paths`` groups.

    Workflow
    --------
    1. Loads per-frame visibility & relation data from cache.
    2. Loads `camera_pose.json` to get the list of selected keyframes.
    3. Builds GPT prompts for only those keyframes.
    4. Saves results in `output/descriptions/` as per-frame and combined files.
    """
    dataset_key = "scannetpp" if dataset.lower() == "scannet" else "3rscan"
    dataset_cfg = cfg.dataset[dataset_key]
    description_model = str(dataset_cfg.get("description_model", "gpt-5.2"))

    # Resolve dataset and scene path
    if dataset.lower() == "scannet":
        scene_path = Path(cfg.paths.scannet_root) / scene_id
    else:
        scene_path = Path(cfg.paths.rscan_root) / scene_id

    output_dir = scene_path / "output"
    cache_json = output_dir / "cache" / f"{scene_id}.json"
    desc_dir = output_dir / "descriptions"
    desc_dir.mkdir(exist_ok=True, parents=True)

    # --- Verify input files ---
    if not cache_json.exists():
        raise FileNotFoundError(f"[ERROR] Cache file not found: {cache_json}")
    pose_dict = load_camera_poses_json(scene_path)
    if not pose_dict:
        raise FileNotFoundError(f"[ERROR] camera_pose.json not found for {scene_id}")

    # --- Load cache data ---
    with open(cache_json, "r") as f:
        image_stats = json.load(f)
    cache_by_fid = {entry["fid"]: entry for entry in image_stats}

    selected_fids = list(pose_dict.keys())
    print(f"[INFO] Loaded {len(image_stats)} total frames from cache.")
    print(f"[INFO] Found {len(selected_fids)} selected keyframes from camera_pose.json.")

    annotations = []

    # --- Loop only over selected keyframes ---
    for fid in tqdm(selected_fids, desc="Generating GPT descriptions", dynamic_ncols=True):
        entry = cache_by_fid.get(fid)
        if entry is None:
            print(f"[WARN] Frame {fid} not found in cache. Skipping.")
            continue

        visible_objects = entry.get("visible_objects", {})
        spatial_relations = entry.get("spatial_relations", [])
        if not visible_objects:
            print(f"[WARN] No visible objects for frame {fid}. Skipping.")
            continue

        # Build GPT prompt & generate description
        prompt = build_prompt(fid, visible_objects, spatial_relations)
        description = call_gpt(prompt, model=description_model)

        pose_matrix = pose_dict.get(fid, None)
        annotation = {
            "scene_index": scene_id,
            "image_index": fid,
            "scene_pose": pose_matrix,
            "description": description,
            "visible_objects": visible_objects,
            "spatial_relations": spatial_relations,
            "timestamp": datetime.now().isoformat(),
        }
        annotations.append(annotation)

        # Save per-frame files
        (desc_dir / f"{fid}.json").write_text(json.dumps(annotation, indent=2))
        (desc_dir / f"{fid}.txt").write_text(description)

    # --- Save combined file ---
    (desc_dir / "all_descriptions.json").write_text(json.dumps(annotations, indent=2))
    print(f"[INFO] Saved {len(annotations)} GPT-generated descriptions for selected keyframes in: {desc_dir}")



# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def cli(cfg: DictConfig) -> None:
    main(
        scene_id=cfg.scan_id,
        dataset=cfg.dataset.get("target", "3RScan"),
        cfg=cfg,
    )


if __name__ == "__main__":
    cli()
