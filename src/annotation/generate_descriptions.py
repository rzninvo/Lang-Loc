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
from src.utils.camera_utils import load_camera_poses_json

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


def filter_salient_objects_gpt(
    visible_objects: dict,
    model: str = "gpt-4o-mini",
    debug_log: bool = True,
) -> dict:
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
        if debug_log:
            removed_items = []
            for oid, obj in visible_objects.items():
                if oid not in filtered_objects:
                    removed_items.append(_format_object_for_log(oid, obj))
            if removed_items:
                print(f"[DEBUG] GPT non-salient objects: {', '.join(removed_items)}")
        return filtered_objects

    except Exception as e:
        print(f"[ERROR] GPT object filtering failed: {e}")
        print(f"[INFO] Using all {len(visible_objects)} objects without filtering.")
        return visible_objects


def _format_object_for_log(oid, obj: dict) -> str:
    """Compact object formatter for debug logs."""
    label = str(obj.get("label", f"object_{oid}")).strip() or f"object_{oid}"
    px = obj.get("pixel_percent", 0.0)
    try:
        px = float(px)
    except (TypeError, ValueError):
        px = 0.0
    return f"{oid}:{label} ({px:.2f}%)"


def filter_salient_objects_rule_based(
    visible_objects: dict,
    exclude_labels: list[str] | None = None,
    min_pixel_percent: float = 0.0,
    max_objects: int | None = None,
    debug_log: bool = True,
) -> dict:
    """
    Deterministic saliency filtering:
    1) Remove known structural labels.
    2) Remove objects below a pixel-percent threshold.
    3) Keep top-K by pixel_percent (optional).
    """
    if not visible_objects:
        return visible_objects

    if exclude_labels is None:
        exclude_labels = ["wall", "floor", "ceiling", "beam", "column"]
    exclude_set = {str(lbl).strip().lower() for lbl in exclude_labels if str(lbl).strip()}

    candidates = []
    removed_items = []
    for oid, obj in visible_objects.items():
        label = str(obj.get("label", "")).strip().lower()
        if label in exclude_set:
            removed_items.append((_format_object_for_log(oid, obj), f"excluded_label:{label}"))
            continue

        px = obj.get("pixel_percent", 0.0)
        try:
            pixel_percent = float(px)
        except (TypeError, ValueError):
            pixel_percent = 0.0

        if pixel_percent < float(min_pixel_percent):
            removed_items.append((_format_object_for_log(oid, obj), f"low_pixel_percent:{pixel_percent:.2f}"))
            continue

        candidates.append((pixel_percent, str(oid), oid, obj))

    if not candidates:
        print("[WARN] Rule-based saliency filtered all objects. Keeping original list.")
        return visible_objects

    candidates.sort(key=lambda x: (-x[0], x[1]))
    if isinstance(max_objects, int) and max_objects > 0:
        dropped = candidates[max_objects:]
        candidates = candidates[:max_objects]
        for pixel_percent, _, oid, obj in dropped:
            removed_items.append((_format_object_for_log(oid, obj), f"top_k_prune:{pixel_percent:.2f}"))

    filtered = {oid: obj for _, _, oid, obj in candidates}
    print(f"[INFO] Rule-based filtered: {len(visible_objects)} → {len(filtered)} salient objects")
    if debug_log and removed_items:
        removed_text = ", ".join([f"{entry} [{reason}]" for entry, reason in removed_items])
        print(f"[DEBUG] Rule-based non-salient objects: {removed_text}")
    return filtered


def load_3rscan_object_attributes(scene_id: str, objects_json_path: Path) -> dict:
    """
    Load 3RScan object metadata for a scene from `objects.json`.
    Returns nested semantic attributes plus selected top-level metadata.
    """
    if not objects_json_path.exists():
        return {}

    try:
        payload = json.loads(objects_json_path.read_text())
    except Exception as exc:
        print(f"[WARN] Failed to parse {objects_json_path}: {exc}")
        return {}

    scans = payload.get("scans", [])
    scene_entry = next((scan for scan in scans if scan.get("scan") == scene_id), None)
    if scene_entry is None:
        return {}

    attributes_by_oid = {}
    for obj in scene_entry.get("objects", []):
        raw_id = obj.get("id")
        if raw_id is None:
            continue
        try:
            oid = int(raw_id)
        except (TypeError, ValueError):
            continue
        attrs_raw = obj.get("attributes", {})
        merged = dict(attrs_raw) if isinstance(attrs_raw, dict) else {}

        for key in ("ply_color", "nyu40", "eigen13", "rio27", "global_id", "id", "label"):
            value = obj.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            merged[key] = value

        affordances = obj.get("affordances")
        if isinstance(affordances, list):
            merged["affordances"] = affordances

        attributes_by_oid[oid] = merged

    return attributes_by_oid


def attach_object_attributes(visible_objects: dict, attributes_by_oid: dict) -> dict:
    """
    Attach `attributes` to each visible object when available.
    """
    if not visible_objects or not attributes_by_oid:
        return visible_objects

    enriched = {}
    for oid_raw, obj_meta in visible_objects.items():
        obj_copy = dict(obj_meta) if isinstance(obj_meta, dict) else obj_meta
        oid = None
        try:
            oid = int(oid_raw)
        except (TypeError, ValueError):
            pass

        if oid is not None and isinstance(obj_copy, dict):
            existing_attrs = obj_copy.get("attributes", {})
            if not isinstance(existing_attrs, dict):
                existing_attrs = {}
            source_attrs = attributes_by_oid.get(oid, {})
            if not isinstance(source_attrs, dict):
                source_attrs = {}
            # Start from source metadata and only override with non-empty cached values.
            merged_attrs = dict(source_attrs)
            for key, value in existing_attrs.items():
                if key not in merged_attrs:
                    merged_attrs[key] = value
                elif not _is_empty_attr_value(value):
                    merged_attrs[key] = value
            obj_copy["attributes"] = merged_attrs

        enriched[oid_raw] = obj_copy

    return enriched


def _is_empty_attr_value(value) -> bool:
    """Treat None/empty strings/empty lists/empty dicts as empty values."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _as_string_list(value) -> list[str]:
    """Normalize scalar/list attribute values into a clean list of strings."""
    if isinstance(value, str):
        txt = value.strip()
        return [txt] if txt else []
    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            txt = str(item).strip()
            if txt:
                out.append(txt)
        return out
    return []


def _hex_to_basic_color_name(hex_color: str) -> str | None:
    """
    Convert a hex color string (e.g. '#aec7e8') into a coarse color name.
    """
    if not isinstance(hex_color, str):
        return None
    hex_color = hex_color.strip()
    if len(hex_color) != 7 or not hex_color.startswith("#"):
        return None
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except ValueError:
        return None

    palette = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "red": (220, 20, 60),
        "orange": (255, 140, 0),
        "yellow": (255, 215, 0),
        "green": (34, 139, 34),
        "cyan": (0, 180, 180),
        "blue": (65, 105, 225),
        "purple": (138, 43, 226),
        "pink": (255, 105, 180),
        "brown": (139, 69, 19),
    }
    best_name = None
    best_dist = None
    for name, (pr, pg, pb) in palette.items():
        dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def extract_color_hint(obj_meta: dict) -> str | None:
    """
    Extract a concise color hint from object metadata.
    Priority: explicit `attributes.color/colour` > inferred from `attributes.ply_color`.
    """
    if not isinstance(obj_meta, dict):
        return None
    attrs = obj_meta.get("attributes", {})
    if not isinstance(attrs, dict):
        attrs = {}

    for key in ("color", "colour"):
        values = _as_string_list(attrs.get(key))
        if values:
            return values[0].lower()

    ply_color = attrs.get("ply_color")
    if isinstance(ply_color, str):
        inferred = _hex_to_basic_color_name(ply_color)
        if inferred:
            return inferred

    return None


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
        "Focus on what's most prominent and how the space feels. "
        "Use color hints only when they sound natural and helpful."
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
    dataset_cfg = cfg.get(dataset_key, {})
    description_model = str(dataset_cfg.get("description_model", "gpt-4o-mini"))

    # New mode selector with backward compatibility to `filter_nonsalient_objects`.
    # Allowed: "none", "rule_based", "gpt".
    legacy_filter_nonsalient = bool(dataset_cfg.get("filter_nonsalient_objects", False))
    saliency_mode_raw = dataset_cfg.get("saliency_filter_mode")
    if saliency_mode_raw is None:
        saliency_mode = "gpt" if legacy_filter_nonsalient else "none"
    else:
        saliency_mode = str(saliency_mode_raw).strip().lower()
    if saliency_mode not in {"none", "rule_based", "gpt"}:
        print(f"[WARN] Unknown saliency_filter_mode='{saliency_mode}'. Falling back to 'none'.")
        saliency_mode = "none"

    saliency_rule_cfg = dataset_cfg.get("saliency_rule_based", {})
    if not isinstance(saliency_rule_cfg, dict):
        saliency_rule_cfg = {}

    exclude_labels = saliency_rule_cfg.get(
        "exclude_labels",
        ["wall", "floor", "ceiling", "beam", "column"],
    )
    if not isinstance(exclude_labels, list):
        exclude_labels = ["wall", "floor", "ceiling", "beam", "column"]

    try:
        rule_min_pixel_percent = float(saliency_rule_cfg.get("min_pixel_percent", 0.0))
    except (TypeError, ValueError):
        rule_min_pixel_percent = 0.0

    raw_max_objects = saliency_rule_cfg.get("max_objects", None)
    if raw_max_objects is None:
        rule_max_objects = None
    else:
        try:
            parsed_max = int(raw_max_objects)
            rule_max_objects = parsed_max if parsed_max > 0 else None
        except (TypeError, ValueError):
            rule_max_objects = None

    saliency_debug_log = bool(dataset_cfg.get("saliency_debug_log", True))

    # Resolve dataset and scene path
    objects_json_path = None
    if dataset.lower() == "scannet":
        dataset_path = Path(cfg["paths"]["scannet_dataset_path"])
        scene_path = dataset_path / scene_id
    else:
        base_path = Path(cfg["paths"]["base_data_dir"])
        scene_path = base_path / "3RScan" / scene_id
        objects_json_path = base_path / "3RScan" / "objects.json"

    # Loaded lazily only if cache objects need enrichment.
    object_attributes_by_oid = {}

    output_dir = scene_path / "output"
    cache_json = output_dir / "cache" / f"{scene_id}.json"
    desc_dir = output_dir / "descriptions"
    desc_dir.mkdir(exist_ok=True, parents=True)

    if saliency_mode == "gpt":
        print(
            f"[INFO] Saliency filtering mode: gpt "
            f"(model: {description_model}, debug_log={saliency_debug_log})"
        )
    elif saliency_mode == "rule_based":
        print(
            "[INFO] Saliency filtering mode: rule_based "
            f"(exclude_labels={exclude_labels}, min_pixel_percent={rule_min_pixel_percent}, "
            f"max_objects={rule_max_objects}, debug_log={saliency_debug_log})"
        )
    else:
        print("[INFO] Saliency filtering mode: none")

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

    # For 3RScan, only load objects.json if cache entries are missing attributes.
    if objects_json_path is not None:
        need_enrichment = False
        for fid in selected_fids[:20]:
            entry = cache_by_fid.get(fid)
            if not entry:
                continue
            visible_objects = entry.get("visible_objects", {})
            for obj in visible_objects.values():
                attrs = obj.get("attributes")
                if not isinstance(attrs, dict):
                    need_enrichment = True
                    break
                # Older caches may only include a subset; enrich if palette color is absent.
                if "ply_color" not in attrs:
                    need_enrichment = True
                    break
            if need_enrichment:
                break

        if need_enrichment:
            object_attributes_by_oid = load_3rscan_object_attributes(scene_id, objects_json_path)
            if object_attributes_by_oid:
                print(
                    f"[INFO] Loaded attributes for {len(object_attributes_by_oid)} objects "
                    f"from {objects_json_path} (cache enrichment)."
                )
        else:
            print("[INFO] Cache already contains object attributes; skipping objects.json reload.")

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

        # Ensure attributes are present in outputs even for older caches.
        visible_objects = attach_object_attributes(visible_objects, object_attributes_by_oid)

        # --- Apply saliency filtering (optional) ---
        if saliency_mode == "gpt":
            visible_objects = filter_salient_objects_gpt(
                visible_objects,
                model=description_model,
                debug_log=saliency_debug_log,
            )
        elif saliency_mode == "rule_based":
            visible_objects = filter_salient_objects_rule_based(
                visible_objects,
                exclude_labels=exclude_labels,
                min_pixel_percent=rule_min_pixel_percent,
                max_objects=rule_max_objects,
                debug_log=saliency_debug_log,
            )

        if saliency_mode != "none":
            # Keep only relations between retained salient object labels.
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
