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
                        "You describe indoor camera views in clear, fluent, human-style language. "
                        "Your descriptions should read naturally and coherently, not as lists. "
                        "Use simple adjectives only when they help clarity (e.g., small, large, tall). "
                        "Focus on the major objects, their layout, and the structure of the space. "
                        "Avoid storytelling or emotion. "
                        "Avoid speculative statements beyond the visible data. "
                        "Avoid speculation about the room type"
                        "Stay grounded in the provided objects and relations. "
                        "Keep it under 120 words."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT request failed: {e}")
        return "[Error generating description]"


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
    relations_text = ", ".join(
        [f"{r['subject']} is {r['relation']} {r['object']}" for r in spatial_relations]
    )

    prompt = (
        f"You are given structured metadata about an indoor camera frame.\n\n"
        f"Frame ID: {fid}\n"
        f"Visible objects: {', '.join(obj_list)}\n"
        f"Spatial hints: {relations_text if relations_text else 'none'}\n\n"
        "Write a natural 2-4 sentence description that explains the scene, the positions "
        "of the main objects and their relations to each other (if they have any). "
        "Avoid using complex adjectives unless necessary for clarity. "
        "Do not list the relations verbatim; integrate them into natural phrasing. "
        "Avoid robotic language. Avoid speculation beyond what the metadata permits."
        "Avoid sugar coating or storytelling."
    )
    return prompt


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

        # Build GPT prompt & generate description
        prompt = build_prompt(fid, visible_objects, spatial_relations)
        description = call_gpt(prompt)

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
