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
    python3 -m src.annotation.generate_descriptions \
        <scene_id> \
        --dataset 3RScan \
        --config config/default.yaml
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from src.utils.config_loader import load_config

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------

# Load the environment variables from the project root .env file.
# This file must contain an OPENAI_API_KEY entry.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    print(f"[WARN] .env file not found at {dotenv_path}")

# Initialize the OpenAI client (uses OPENAI_API_KEY from environment)
client = OpenAI()

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def load_camera_poses(scene_path: Path) -> dict:
    """
    Load the per-frame camera pose dictionary for a given scene.

    Each pose is stored as a 4x4 SE(3) matrix in the file:
    `output/camera_pose.json`, generated during keyframe extraction.

    Parameters
    ----------
    scene_path : Path
        Path to the scene root directory (e.g., `<dataset_root>/3RScan/<scene_id>`).

    Returns
    -------
    dict
        A dictionary mapping `image_index` (frame ID string) → 4x4 list of floats
        representing the camera-to-world transformation matrix.
        Returns an empty dictionary if the file is missing.

    """
    pose_file = scene_path / "output" / "camera_pose.json"
    if not pose_file.exists():
        return {}
    return json.loads(pose_file.read_text())


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


def filter_salient_objects_gpt(visible_objects: dict, model: str = "gpt-4o-mini") -> dict:
    """
    Use GPT to filter out non-salient objects (walls, floors, structural elements)
    and keep only objects important for scene description.

    Parameters
    ----------
    visible_objects : dict
        Mapping from object ID → metadata dictionary with 'label' and 'pixel_percent' fields.
    model : str, optional
        OpenAI model to use for filtering. Default is "gpt-4o-mini".

    Returns
    -------
    dict
        Filtered dictionary containing only salient objects.
        Returns original dict if GPT call fails or returns invalid response.
    """
    if not visible_objects:
        return visible_objects

    # Build object list with labels and pixel percentages
    obj_info_list = []
    for oid, obj in visible_objects.items():
        label = obj.get('label', f'object_{oid}')
        pixel_pct = obj.get('pixel_percent', 0.0)
        obj_info_list.append(f"{label} ({pixel_pct:.1f}%)")

    obj_info_str = ', '.join(obj_info_list)

    prompt = (
        f"Given these objects visible in an indoor camera view with their coverage percentages:\n"
        f"{obj_info_str}\n\n"
        f"Task: Identify which objects are IMPORTANT for describing the scene.\n"
        f"Guidelines:\n"
        f"- Include: furniture, appliances, fixtures, decorative items, electronics\n"
        f"- Exclude: walls, floors, ceilings, structural elements (beams, columns)\n"
        f"- Consider: Objects with higher coverage (%) are usually more important\n"
        f"- Keep objects that help understand the room's purpose and layout\n\n"
        f"Return ONLY a comma-separated list of the important object names (without percentages). "
        f"Use the exact names from the input. No explanations."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a scene analysis assistant that identifies salient objects. "
                        "Return only a comma-separated list of important object names. "
                        "Be concise and exact."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # Deterministic filtering
            max_completion_tokens=150,  # Updated parameter name for newer models
        )

        response_text = response.choices[0].message.content.strip()

        # Parse GPT response: split by comma and normalize
        important_labels = {label.strip().lower() for label in response_text.split(',')}

        # Filter visible_objects to keep only those with important labels
        filtered_objects = {
            oid: obj for oid, obj in visible_objects.items()
            if obj.get('label', '').strip().lower() in important_labels
        }

        if not filtered_objects:
            # If GPT filtered everything out (error), return original
            print(f"[WARN] GPT filtered all objects. Keeping original list.")
            return visible_objects

        print(f"[INFO] GPT filtered: {len(visible_objects)} → {len(filtered_objects)} salient objects")
        return filtered_objects

    except Exception as e:
        print(f"[ERROR] GPT object filtering failed: {e}")
        print(f"[INFO] Using all {len(visible_objects)} objects without filtering.")
        return visible_objects


# ---------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------

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

    # Build a more natural prompt
    prompt_parts = [f"Describe what you see in this indoor camera view.\n"]
    prompt_parts.append(f"Visible objects: {', '.join(obj_list)}\n")

    # Only include spatial hints if there are meaningful relations AND multiple objects
    if spatial_relations and len(visible_objects) > 1:
        # Format relations more naturally
        relations_natural = []
        for r in spatial_relations[:5]:  # Limit to top 5 most important relations
            subj = r['subject']
            obj = r['object']
            rel = r['relation']
            # Convert technical relations to natural language hints
            if rel == "left_of":
                relations_natural.append(f"{subj} left of {obj}")
            elif rel == "right_of":
                relations_natural.append(f"{subj} right of {obj}")
            elif rel == "in_front_of":
                relations_natural.append(f"{subj} in front of {obj}")
            elif rel == "behind":
                relations_natural.append(f"{subj} behind {obj}")
            elif rel == "above":
                relations_natural.append(f"{subj} above {obj}")
            elif rel == "below":
                relations_natural.append(f"{subj} below {obj}")
            else:
                relations_natural.append(f"{subj} {rel} {obj}")

        if relations_natural:
            prompt_parts.append(f"Layout hints: {', '.join(relations_natural)}\n")

    prompt_parts.append(
        "\nDescribe the view in 2-3 short sentences. "
        "Write naturally and conversationally, as if describing it to a friend. "
        "Focus on what's most prominent and how the space feels."
    )

    return "".join(prompt_parts)


# ---------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------

def main(scene_id: str, dataset: str, config_path: str):
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
    config_path : str
        Path to the YAML configuration file (defines dataset paths).

    Workflow
    --------
    1. Loads per-frame visibility & relation data from cache.
    2. Loads `camera_pose.json` to get the list of selected keyframes.
    3. Builds GPT prompts for only those keyframes.
    4. Saves results in `output/descriptions/` as per-frame and combined files.
    """
    cfg = load_config(config_path)

    # Load filtering configuration
    dataset_key = "scannetpp" if dataset.lower() == "scannet" else "3rscan"
    filter_nonsalient = bool(cfg.get(dataset_key, {}).get("filter_nonsalient_objects", False))
    description_model = str(cfg.get(dataset_key, {}).get("description_model", "gpt-4o-mini"))

    # Resolve dataset and scene path
    if dataset.lower() == "scannet":
        dataset_path = Path(cfg["paths"]["scannet_dataset_path"])
        scene_path = dataset_path / scene_id
    else:
        base_path = Path(cfg["paths"]["base_data_dir"])
        scene_path = base_path / "3RScan" / scene_id

    output_dir = scene_path / "output"
    cache_json = output_dir / "cache" / f"{scene_id}.json"
    desc_dir = output_dir / "descriptions"
    desc_dir.mkdir(exist_ok=True, parents=True)

    if filter_nonsalient:
        print(f"[INFO] GPT-based object filtering is ENABLED (model: {description_model})")
    else:
        print(f"[INFO] GPT-based object filtering is DISABLED")

    # --- Verify input files ---
    if not cache_json.exists():
        raise FileNotFoundError(f"[ERROR] Cache file not found: {cache_json}")
    pose_dict = load_camera_poses(scene_path)
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

        # --- Apply GPT-based saliency filtering (optional) ---
        if filter_nonsalient:
            visible_objects = filter_salient_objects_gpt(visible_objects, model=description_model)

            # Filter spatial relations to only include salient objects
            salient_labels = {obj.get('label', '').lower() for obj in visible_objects.values()}
            spatial_relations = [
                rel for rel in spatial_relations
                if rel.get('subject', '').lower() in salient_labels
                and rel.get('object', '').lower() in salient_labels
            ]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate GPT-based keyframe descriptions for ScanNet++ or 3RScan scenes."
    )
    parser.add_argument("scene_id", help="Scene identifier (e.g., scene0000_00 or 3RScan UUID).")
    parser.add_argument("--dataset", required=True, choices=["scannet", "3RScan"], help="Dataset type.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to configuration YAML file.")
    args = parser.parse_args()
    main(args.scene_id, args.dataset, args.config)
