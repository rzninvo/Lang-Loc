"""Shared utilities for scene graph construction scripts.

Provides spatial descriptors, PLY/semseg loading, geometric feature
computation, relation helpers, and scene-level CLIP embedding functions
used by the ``build_scene_*`` family of scripts.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData


# ── Thresholds for spatial relations ────────────────────────────────────────

DIR_THRESHOLD = 0.20
HEIGHT_THRESHOLD = 0.15
NEAR_THRESHOLD = 0.80


# ── Spatial descriptors ────────────────────────────────────────────────────

def get_spatial_descriptor(centroid, bbox_size, scene_bounds):
    """Create spatial descriptor: e.g., 'north_upper', 'south_lower'"""
    x, y, z = centroid
    min_x, max_x, min_y, max_y, min_z, max_z = scene_bounds

    x_range = max_x - min_x
    z_range = max_z - min_z

    x_norm = (x - min_x) / x_range if x_range > 0 else 0.5
    z_norm = (z - min_z) / z_range if z_range > 0 else 0.5

    if abs(x_norm - 0.5) > 0.3:
        horizontal = "east" if x_norm > 0.5 else "west"
    elif abs(z_norm - 0.5) > 0.3:
        horizontal = "north" if z_norm > 0.5 else "south"
    else:
        horizontal = "center"

    y_range = max_y - min_y
    y_norm = (y - min_y) / y_range if y_range > 0 else 0.5

    if y_norm > 0.66:
        vertical = "upper"
    elif y_norm < 0.33:
        vertical = "lower"
    else:
        vertical = "middle"

    return f"{horizontal}_{vertical}"


def get_size_descriptor(bbox_size):
    """Classify object by size."""
    max_dim = max(bbox_size)
    if max_dim > 2.0:
        return "large"
    elif max_dim > 0.8:
        return "medium"
    else:
        return "small"


def get_color_descriptor(mean_color):
    """Get dominant color name."""
    r, g, b = mean_color

    if max(r, g, b) - min(r, g, b) < 30:
        if sum([r, g, b]) / 3 < 80:
            return "dark"
        elif sum([r, g, b]) / 3 > 180:
            return "white"
        else:
            return "gray"

    if r > g and r > b:
        if r > 180:
            return "red"
        else:
            return "brown"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    else:
        return "mixed"


def create_unique_label(base_label, centroid, bbox_size, mean_color,
                        scene_bounds, label_counts, node_id):
    """Create unique label with spatial and node ID info."""
    if base_label.startswith("obj") or base_label == "object" or base_label == "unknown":
        size = get_size_descriptor(bbox_size)
        color = get_color_descriptor(mean_color)
        spatial = get_spatial_descriptor(centroid, bbox_size, scene_bounds)
        unique_label = f"{size}_{color}_object_{node_id}_{spatial}"
    else:
        spatial = get_spatial_descriptor(centroid, bbox_size, scene_bounds)
        base_with_spatial = f"{base_label}_{spatial}"

        if base_with_spatial in label_counts:
            count = label_counts[base_with_spatial]
            label_counts[base_with_spatial] += 1
            unique_label = f"{base_with_spatial}_{count}"
        else:
            label_counts[base_with_spatial] = 1
            unique_label = base_with_spatial

    return unique_label


# ── PLY and semseg loading ─────────────────────────────────────────────────

def load_ply_instances(ply_path):
    """Load point cloud and group by instance ID.

    Returns ``(instance_points, all_xyz)`` where *instance_points* maps
    ``str(objectId)`` to ``(xyz, rgb)`` arrays.  Background (objectId 0)
    is excluded.
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T
    inst = v["objectId"]

    instance_ids = np.unique(inst)
    instance_ids = instance_ids[instance_ids > 0]

    instance_points = {}
    for iid in instance_ids:
        mask = inst == iid
        pts = xyz[mask]
        colors = rgb[mask]
        instance_points[str(int(iid))] = (pts, colors)

    return instance_points, xyz


def load_semseg(semseg_path):
    """Load semantic segmentation labels.

    Returns mapping from ``str(objectId)`` to label string.
    """
    with open(semseg_path, "r") as f:
        data = json.load(f)

    id_to_label = {}
    for group in data["segGroups"]:
        object_id = str(group["objectId"])
        label = group["label"].lower()
        id_to_label[object_id] = label

    return id_to_label


# ── Geometric features ────────────────────────────────────────────────────

def compute_node_attributes(points, colors):
    """Compute centroid, mean color, and radius."""
    centroid = points.mean(axis=0)
    d = np.linalg.norm(points - centroid, axis=1)
    radius = float(min(d.max(), 0.40))
    mean_color = colors.mean(axis=0).tolist()
    return centroid.tolist(), mean_color, radius


def compute_geometric_features(points):
    """Compute geometric features for the object."""
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    eigenvalues = np.sort(eigenvalues)[::-1]

    l1, l2, l3 = eigenvalues
    linearity = (l1 - l2) / l1 if l1 > 0 else 0
    planarity = (l2 - l3) / l1 if l1 > 0 else 0
    sphericity = l3 / l1 if l1 > 0 else 0

    std_dev = np.sqrt(eigenvalues).tolist()
    extent = (2 * np.sqrt(eigenvalues)).tolist()

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = (bbox_max - bbox_min).tolist()

    return {
        "std_dev": std_dev,
        "std_color": [0, 0, 0],
        "extent": extent,
        "linearity": float(linearity),
        "planarity": float(planarity),
        "sphericity": float(sphericity),
        "bbox_size": bbox_size,
        "num_points": int(len(points))
    }


# ── Spatial relations ──────────────────────────────────────────────────────

def directional_rel(ci, cj):
    """Compute directional relationships."""
    dx, dy, dz = cj - ci
    out = []
    if abs(dx) > DIR_THRESHOLD:
        out.append("right_of" if dx > 0 else "left_of")
    if abs(dz) > DIR_THRESHOLD:
        out.append("in_front_of" if dz > 0 else "behind")
    if abs(dy) > HEIGHT_THRESHOLD:
        out.append("above" if dy > 0 else "below")
    return out


def distance_rel(ci, cj, ri, rj):
    """Compute distance relationships."""
    d = np.linalg.norm(cj - ci)
    if d <= (ri + rj) * 0.65:
        return ["touching"]
    if d < NEAR_THRESHOLD:
        return ["near"]
    return []


def symmetric_rel(sub, obj, rel):
    """Get symmetric relationship."""
    inverses = {
        "left_of": "right_of",
        "right_of": "left_of",
        "in_front_of": "behind",
        "behind": "in_front_of",
        "above": "below",
        "below": "above",
        "near": "near"
    }
    if rel in inverses:
        return obj, sub, inverses[rel]
    return None


# ── Scene-level CLIP helpers ───────────────────────────────────────────────

def create_scene_description(nodes, edges, max_objects=10, max_relations=5):
    """Create a natural language description of the scene.

    Example output::

        "A room with desk, chair, monitor, lamp, bookshelf. desk near chair;
        monitor on desk; lamp on desk"
    """
    object_labels = []
    node_id_to_label = {}

    for node_id, node_data in nodes.items():
        base_label = node_data['base_label']
        object_labels.append(base_label)
        node_id_to_label[node_id] = base_label

    object_counts = {}
    for label in object_labels:
        object_counts[label] = object_counts.get(label, 0) + 1

    sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
    top_objects = [obj for obj, count in sorted_objects[:max_objects]]
    object_str = ", ".join(top_objects)

    relation_strs = []
    subject_counts = {}
    exclude_from_relations = {'floor', 'wall', 'ceiling', 'walls'}

    for edge in edges:
        if len(relation_strs) >= max_relations:
            break

        subj_id = edge['subject']
        obj_id = edge['object']
        rel = edge['relation']

        if subj_id in node_id_to_label and obj_id in node_id_to_label:
            subj_label = node_id_to_label[subj_id]
            obj_label = node_id_to_label[obj_id]

            if subj_label in exclude_from_relations or obj_label in exclude_from_relations:
                continue
            if subject_counts.get(subj_label, 0) >= 2:
                continue

            relation_strs.append(f"{subj_label} {rel} {obj_label}")
            subject_counts[subj_label] = subject_counts.get(subj_label, 0) + 1

    if relation_strs:
        description = f"A room with {object_str}. {'; '.join(relation_strs)}"
    else:
        description = f"A room with {object_str}"

    return description


def get_scene_clip_embedding(description, clip_model, device):
    """Get CLIP embedding for a scene description string."""
    import clip
    import torch
    with torch.no_grad():
        tokens = clip.tokenize([description]).to(device)
        clip_emb = clip_model.encode_text(tokens)[0]
        clip_emb = clip_emb / clip_emb.norm()
    return clip_emb.cpu().tolist()
