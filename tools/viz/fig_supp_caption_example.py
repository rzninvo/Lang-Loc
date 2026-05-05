#!/usr/bin/env python3
"""Generate D4 dataset slide figure: description-generation quadrant example.

Composes a 2x2 figure for one frame:
  top-left:     RGB keyframe
  top-right:    structured prompt (visible objects + spatial relations)
  bottom-left:  scene-graph render
  bottom-right: GPT-4o-mini caption (pink quote box)

Usage::

    python -m tools.viz.fig_supp_caption_example \\
        --root ./data/scans \\
        --scan-id scene0002_00 \\
        --frame-id 000838 \\
        --scene-graph-img ./teaser_output/scene0002_00_scene_graph.png \\
        --output docs/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
import textwrap


_GENERIC_LABELS = {"floor", "wall", "ceiling"}

# Inverse-predicate dedup for spatial relations (mirrors the real prompt
# builder so the figure shows verbatim what GPT-4o-mini actually receives).
_INVERSE_PRED_IDS = {2: 3, 3: 2, 4: 5, 5: 4}


def _extract_color_hint(obj_meta: dict) -> str | None:
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


def _dedup_relations_for_prompt(spatial_relations: list) -> list:
    seen: set = set()
    deduped: list = []
    for r in spatial_relations:
        sub = r.get("subject_id", r.get("subject"))
        obj = r.get("object_id", r.get("object"))
        pred_id = r.get("predicate_id")
        if pred_id is not None and pred_id in _INVERSE_PRED_IDS:
            pair = frozenset((sub, obj))
            canon = ("dir", pair)
        elif pred_id is not None:
            pair = frozenset((sub, obj))
            canon = ("rel", pair, pred_id)
        else:
            pair = frozenset((r.get("subject", ""), r.get("object", "")))
            canon = ("heur", pair, r.get("relation", ""))
        if canon in seen:
            continue
        seen.add(canon)
        deduped.append(r)
    return deduped


def _build_actual_prompt(visible_objects: dict, spatial_relations: list,
                         max_relations: int | None = None) -> str:
    """Mirrors ``langloc.dataset.annotation.generate_descriptions.build_prompt``
    so the figure shows the exact text that the GPT-4o-mini call receives.

    For slide use, pass ``max_relations`` to truncate the (often very long)
    Layout-hints line to the first ``N`` relations and append a
    ``"… (+ K more)"`` marker — the API call still sees the full list.
    """
    obj_list = [(v.get('label') or f"object {oid}")
                for oid, v in visible_objects.items()]
    color_hints = []
    for oid, obj in visible_objects.items():
        label = (obj.get("label") or f"object {oid}").strip()
        color = _extract_color_hint(obj)
        if color:
            color_hints.append(f"{label}: {color}")

    parts = ["Describe what you see in this indoor camera view.\n"]
    parts.append(f"Visible objects: {', '.join(obj_list)}\n")
    if color_hints:
        parts.append(f"Color hints: {', '.join(color_hints[:12])}\n")
    if spatial_relations and len(visible_objects) > 1:
        unique_rels = _dedup_relations_for_prompt(spatial_relations)
        relations_natural = []
        for r in unique_rels:
            subj = r['subject']
            obj = r['object']
            rel = r['relation'].replace("_", " ")
            relations_natural.append(f"{subj} {rel} {obj}")
        if relations_natural:
            shown = relations_natural
            if max_relations is not None and len(relations_natural) > max_relations:
                extra = len(relations_natural) - max_relations
                shown = relations_natural[:max_relations] + [f"… (+ {extra} more)"]
            parts.append(f"Layout hints: {', '.join(shown)}\n")
    parts.append(
        "\nDescribe the view in 2-3 short sentences. "
        "Write naturally and conversationally, as if describing it to a "
        "friend. Focus on what's most prominent and how the space feels. "
        "Use color hints only when they sound natural and helpful."
    )
    return "".join(parts)


def _set_pres_rc():
    """Configure matplotlib to match the Aptos Display presentation style.

    On a system with Aptos Display installed, that font is used directly. On
    other systems, matplotlib falls back to the next available family in the
    list (Inter / Helvetica Neue / DejaVu Sans), all of which are humanist
    geometric sans-serifs that render close to Aptos Display.
    """
    plt.rcParams.update({
        "font.family": ["Aptos Display", "Aptos", "Inter",
                        "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
    })


def _load_thumb(path: Path, target_w: int = 640) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = target_w / w
        im = im.resize((target_w, int(h * scale)), Image.LANCZOS)
        return np.asarray(im)


def _fit_to_canvas(path: Path,
                   target_w: int = 1280,
                   target_h: int = 800,
                   bg_color=(248, 248, 246)) -> np.ndarray:
    """Resize an image to fit ``target_w x target_h`` (preserving aspect),
    then centre-pad with ``bg_color`` to fill the full canvas.

    Two images returned by this function with the same target size always
    display at the same physical size in matplotlib — regardless of their
    original aspect ratios.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        im = im.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), bg_color)
        canvas.paste(im, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        return np.asarray(canvas)


def _format_visible_objects(visible_objects: dict, n_show: int = 8) -> str:
    """Build a monospace block of visible objects sorted by distance."""
    rows = []
    for oid, obj in visible_objects.items():
        if not isinstance(obj, dict):
            continue
        rows.append((
            obj.get("label", "?"),
            float(obj.get("pixel_percent", 0)),
            float(obj.get("distance_from_camera", 0.0)),
        ))
    # Sort by camera distance (closest first)
    rows.sort(key=lambda x: x[2])
    rows = rows[:n_show]
    out = ["Visible objects (camera-frame):"]
    for lab, cov, d in rows:
        out.append(f"  {lab:<11s}  cov={cov:5.1f}%  d={d:4.2f} m")
    return "\n".join(out)


def _format_spatial_relations(spatial_relations: list, n_show: int = 6) -> str:
    """Compact, deduplicated spatial-relation block."""
    seen = set()
    items = []
    for r in spatial_relations:
        if not isinstance(r, dict):
            continue
        s, rel, o = r.get("subject", ""), r.get("relation", ""), r.get("object", "")
        # Skip generic-label edges and "bigger/smaller than" / "lower than" filler;
        # keep only directional + on/near for clarity in the figure.
        if rel not in {"left", "right", "front", "behind", "on", "in",
                       "close_by", "near"}:
            continue
        if s in _GENERIC_LABELS or o in _GENERIC_LABELS:
            continue
        key = tuple(sorted([s, o]) + [rel])
        if key in seen:
            continue
        seen.add(key)
        items.append(f"  {s} {rel} {o}")
        if len(items) >= n_show:
            break
    if not items:
        return "Spatial relations:\n  (none kept after filtering)"
    return "Spatial relations:\n" + "\n".join(items)


def _draw_step_label(ax, n: int, title: str, *,
                     color="#1d2b53") -> None:
    """Minimal numbered step header above a panel.

    Layout:
        ``01  │  Title in regular weight``

    The number is rendered in a heavier weight + accent colour, separated from
    the title by a thin vertical rule. No filled chips, no circles — keeps the
    look clean and presentation-grade.
    """
    # Numeric prefix (large, accent-coloured, light-weight display style)
    ax.text(0.0, 1.060, f"{n:02d}", transform=ax.transAxes,
            fontsize=20, fontweight="bold", color=color,
            ha="left", va="center", zorder=10, clip_on=False)
    # Thin vertical separator
    ax.plot([0.058, 0.058], [1.018, 1.105],
            transform=ax.transAxes,
            color="#bbbbbb", linewidth=0.8, clip_on=False, zorder=10)
    # Title
    ax.text(0.075, 1.060, title, transform=ax.transAxes,
            fontsize=12.5, fontweight="bold", color="#1c1c1c",
            ha="left", va="center", zorder=10, clip_on=False)


def render_caption_example(rgb_path: Path,
                           sg_img_path: Path | None,
                           visible_objects: dict,
                           spatial_relations: list,
                           description: str,
                           output_path: Path,
                           attribution: str = "",
                           dpi: int = 300) -> None:
    _set_pres_rc()

    # Common canvas: both image panels render at this exact pixel size, so they
    # appear identically sized regardless of original aspect ratio.
    CANVAS_W, CANVAS_H = 1600, 1000

    fig = plt.figure(figsize=(14, 7.6), dpi=dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0],
                          height_ratios=[1.0, 1.0],
                          wspace=0.06, hspace=0.22,
                          left=0.025, right=0.985, top=0.92, bottom=0.04)

    accent_blue = "#1d2b53"
    pink_fg = "#b3415a"
    pink_bg = "#fdeaef"
    pink_edge = "#e69aa7"

    # ---- Top-left: RGB keyframe ----
    ax_rgb = fig.add_subplot(gs[0, 0])
    rgb = _load_thumb(rgb_path, target_w=1600)
    # aspect="auto" lets the axes fill the cell exactly (no aspect-driven
    # shrinkage), so transAxes coordinates are consistent between image cells
    # and text cells — that's what keeps "01"/"03" aligned with "02"/"04".
    ax_rgb.imshow(rgb, aspect="auto")
    ax_rgb.set_axis_off()
    _draw_step_label(ax_rgb, 1, "Keyframe (selected by 2-stage DPP)",
                     color=accent_blue)

    # ---- Top-right: structured prompt block ----
    ax_prompt = fig.add_subplot(gs[0, 1])
    ax_prompt.set_axis_off()
    ax_prompt.set_xlim(0, 1)
    ax_prompt.set_ylim(0, 1)
    _draw_step_label(ax_prompt, 2, "Structured prompt to GPT-4o-mini",
                     color=accent_blue)

    # Code-block-style background (subtle border + soft fill)
    box = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.0,rounding_size=0.020",
        linewidth=0.9, edgecolor="#d6d6d2", facecolor="#f7f7f5",
        transform=ax_prompt.transAxes, zorder=1)
    ax_prompt.add_patch(box)

    # Build the actual prompt text that gets sent to GPT-4o-mini, then word-
    # wrap each section for the slide. The structure (header / Visible objects
    # line / Layout hints line / instruction) mirrors `build_prompt` exactly.
    # We truncate the Layout-hints line for the figure only — the API call
    # still receives the full list at generation time.
    raw_prompt = _build_actual_prompt(visible_objects, spatial_relations,
                                      max_relations=8)
    wrap_width = 60
    wrapped_blocks = []
    for line in raw_prompt.split("\n"):
        if not line.strip():
            wrapped_blocks.append("")
            continue
        # Preserve "Header: payload" framing while wrapping the payload
        if ":" in line and line.split(":", 1)[0] in {
                "Visible objects", "Color hints", "Layout hints"}:
            head, payload = line.split(":", 1)
            wrapped_payload = textwrap.fill(
                payload.strip(), width=wrap_width - len(head) - 2,
                subsequent_indent=" " * (len(head) + 2))
            wrapped_blocks.append(f"{head}: {wrapped_payload}")
        else:
            wrapped_blocks.append(textwrap.fill(line, width=wrap_width))
    prompt_text = "\n".join(wrapped_blocks)

    ax_prompt.text(0.045, 0.95, prompt_text, transform=ax_prompt.transAxes,
                   fontsize=9.5, family="monospace",
                   verticalalignment="top", horizontalalignment="left",
                   color="#1c1c1c", zorder=2)

    # ---- Bottom-left: scene graph render ----
    ax_sg = fig.add_subplot(gs[1, 0])
    if sg_img_path is not None and sg_img_path.exists():
        sg = _load_thumb(sg_img_path, target_w=1600)
        ax_sg.imshow(sg, aspect="auto")
    else:
        ax_sg.text(0.5, 0.5, "(scene graph unavailable)",
                   ha="center", va="center", style="italic", color="#888888")
    ax_sg.set_axis_off()
    _draw_step_label(ax_sg, 3, "Object-level scene graph",
                     color=accent_blue)

    # ---- Bottom-right: GPT-4o-mini caption (modern blockquote) ----
    ax_cap = fig.add_subplot(gs[1, 1])
    ax_cap.set_axis_off()
    ax_cap.set_xlim(0, 1)
    ax_cap.set_ylim(0, 1)
    _draw_step_label(ax_cap, 4, "GPT-4o-mini caption",
                     color=pink_fg)

    # Soft rounded card
    card = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.0,rounding_size=0.020",
        linewidth=1.0, edgecolor=pink_edge, facecolor=pink_bg,
        transform=ax_cap.transAxes, zorder=1)
    ax_cap.add_patch(card)

    # Vertical accent bar (modern blockquote indicator)
    bar = Rectangle((0.04, 0.12), 0.012, 0.76,
                    facecolor=pink_fg, edgecolor="none",
                    transform=ax_cap.transAxes, zorder=2)
    ax_cap.add_patch(bar)

    # Subtle decorative opening quote (smaller, inside the bar's column)
    ax_cap.text(0.115, 0.88, "“", transform=ax_cap.transAxes,
                fontsize=34, color=pink_fg, fontweight="bold",
                family="DejaVu Serif",
                zorder=3, va="top", ha="left", alpha=0.7)

    # Wrap caption explicitly so it stays inside the card
    wrapped = textwrap.fill(description, width=58, break_long_words=False)

    ax_cap.text(0.115, 0.50, wrapped, transform=ax_cap.transAxes,
                fontsize=12.5, style="italic", color="#3a1d24",
                ha="left", va="center", zorder=4,
                linespacing=1.35)

    # Source attribution line
    if attribution:
        ax_cap.text(0.97, 0.07, attribution,
                    transform=ax_cap.transAxes,
                    fontsize=8.5, color=pink_fg, style="italic",
                    ha="right", va="bottom", zorder=4, alpha=0.85)

    fig.savefig(output_path, format="pdf", dpi=dpi,
                bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path.with_suffix(".png"), format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="D4 caption-example quadrant figure.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--scan-id", "--scan_id", dest="scan_id", required=True)
    ap.add_argument("--frame-id", "--frame_id", dest="frame_id", required=True)
    ap.add_argument("--scene-graph-img", "--scene_graph_img",
                    dest="scene_graph_img", type=Path, default=None,
                    help="Path to a pre-rendered scene-graph image "
                         "(PDF or PNG). Default: teaser_output/{scan_id}_scene_graph.png.")
    ap.add_argument("--output", type=Path, default=Path("docs/figures"))
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scan_dir = args.root / args.scan_id
    desc_path = scan_dir / "output" / "descriptions" / f"{args.frame_id}.json"
    desc = json.loads(desc_path.read_text())
    print(f"Loaded description: {desc_path}")

    rgb_path = scan_dir / "color" / f"{args.frame_id}.jpg"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB frame not found: {rgb_path}")

    sg_img = args.scene_graph_img
    if sg_img is None:
        cand = (Path("teaser_output") / f"{args.scan_id}_scene_graph.png")
        if cand.exists():
            sg_img = cand
        else:
            print(f"  [warn] No scene graph image found; pass --scene-graph-img")
    if sg_img is not None and sg_img.suffix.lower() == ".pdf":
        print(f"  [warn] scene-graph-img is PDF; please provide a PNG/JPG "
              "(matplotlib's imshow needs raster). Skipping the SG panel.")
        sg_img = None

    out_path = args.output / f"caption_example_{args.scan_id}_{args.frame_id}.pdf"
    render_caption_example(
        rgb_path=rgb_path,
        sg_img_path=sg_img,
        visible_objects=desc.get("visible_objects", {}),
        spatial_relations=desc.get("spatial_relations", []),
        description=desc.get("description", ""),
        output_path=out_path,
        attribution=f"— GPT-4o-mini · {args.scan_id} / frame {args.frame_id}",
    )
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
