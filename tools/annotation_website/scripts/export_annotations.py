#!/usr/bin/env python3
"""Export collected human descriptions to JSONL for the localization
pipeline.

The localization pipeline expects a JSONL where each line carries a
``scene_id``, ``frame_id``, and ``description`` field at minimum (mirrors
the per-frame JSON inside ``output/descriptions/``). Optionally we also
emit a per-scene aggregated JSON in the same shape as
``all_descriptions.json`` so downstream code that reads that path can
operate unchanged.

Usage:

    python scripts/export_annotations.py \\
        --out ../../eval/human_descriptions.jsonl \\
        [--per-scene-json ../../eval/human_descriptions_by_scene/]
        [--include-flagged]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select

# allow running from inside tools/annotation_website
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.db import get_session_factory, init_db  # noqa: E402
from server.models import Annotator, Description, Keyframe  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-scene-json", type=Path, default=None)
    p.add_argument("--include-flagged", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    init_db()
    factory = get_session_factory()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_flagged = 0
    by_scene: dict[str, list[dict]] = defaultdict(list)

    with factory() as sess, args.out.open("w", encoding="utf-8") as fh:
        rows = sess.execute(
            select(Description, Annotator.nickname)
            .join(Annotator, Annotator.id == Description.annotator_id)
            .order_by(Description.scene_id, Description.frame_id, Description.submitted_at)
        ).all()

        for desc, nickname in rows:
            if desc.flagged and not args.include_flagged:
                n_flagged += 1
                continue
            entry = {
                "scene_id": desc.scene_id,
                "frame_id": desc.frame_id,
                "image_index": desc.frame_id,  # same key the pipeline uses
                "description": desc.text,
                "annotator_id": desc.annotator_id,
                "annotator_nickname": nickname,
                "word_count": desc.word_count,
                "duration_ms": desc.duration_ms,
                "flagged": bool(desc.flagged),
                "flag_reason": desc.flag_reason,
                "submitted_at": desc.submitted_at.isoformat() if desc.submitted_at else None,
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            by_scene[desc.scene_id].append(entry)
            n_total += 1

    print(f"wrote {n_total} descriptions to {args.out}")
    if n_flagged and not args.include_flagged:
        print(f"  ({n_flagged} flagged descriptions omitted; pass --include-flagged to keep them)")

    if args.per_scene_json is not None:
        args.per_scene_json.mkdir(parents=True, exist_ok=True)
        for sid, entries in by_scene.items():
            (args.per_scene_json / f"{sid}.json").write_text(
                json.dumps(entries, indent=2, ensure_ascii=False)
            )
        print(f"  also wrote per-scene JSONs to {args.per_scene_json}")


if __name__ == "__main__":
    main()
