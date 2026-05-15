#!/usr/bin/env python3
"""Build a copy of Abu's ``scenes_for_human_annotation/`` tree with the
ChatGPT-generated descriptions replaced by the human-written ones from
the LangLoc annotation site DB.

Output layout mirrors Abu's exactly. For each scene:

  ``output/descriptions/<frame>.json``
      replaced if a human description exists for that frame
      (preserves Abu's ``scene_index`` / ``timestamp`` / etc.; keeps
      the GT-derived ``visible_objects`` and ``spatial_relations``
      from the source JSON, swaps only ``description`` and adds
      ``_describer``, ``_word_count``, ``_submitted_at``)
  everything else under each scene
      symlinked to the source tree (mesh, depth, color, intrinsic,
      cache, .aggregation.json, .txt, .ply, .segs.json, ...)

Use ``--symlink`` (default) to symlink non-description files;
``--copy`` to actually copy them (heavy: meshes are big).

Pick + skip-flagged semantics match
``tools/baselines/human/extract_descriptions.py``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--src-tree", type=Path, required=True,
                   help="Abu's scenes_for_human_annotation/ root")
    p.add_argument("--out-tree", type=Path, required=True,
                   help="parallel tree to write")
    p.add_argument("--pick", choices=["longest", "earliest"], default="longest")
    p.add_argument("--skip-flagged", action="store_true",
                   help="exclude server-flagged descriptions (e.g. 'very "
                        "short time on task'); falls back to all if a frame "
                        "has nothing else")
    p.add_argument("--symlink", action="store_true", default=True,
                   help="symlink non-description files (default)")
    p.add_argument("--copy", dest="symlink", action="store_false",
                   help="copy non-description files instead of symlinking")
    p.add_argument("--keep-abu-on-missing", action="store_true",
                   help="if a frame has no human description, copy Abu's "
                        "original ChatGPT description instead of skipping")
    return p.parse_args()


def _pick_description(rows: list[tuple], pick: str) -> tuple | None:
    """rows: (annotator_id, text, word_count, submitted_at, flagged, flag_reason).
    Returns the chosen row or None."""
    if not rows:
        return None
    if pick == "earliest":
        return min(rows, key=lambda r: r[3])
    return max(rows, key=lambda r: r[2])


def _human_descriptions_for(
    conn: sqlite3.Connection, scene_id: str, frame_id: str, skip_flagged: bool
) -> list[tuple]:
    sql = """
        SELECT annotator_id, text, word_count, submitted_at, flagged, flag_reason
        FROM descriptions
        WHERE scene_id = ? AND frame_id = ?
    """
    if skip_flagged:
        sql += " AND flagged = 0"
    return conn.execute(sql, (scene_id, frame_id)).fetchall()


def _link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()

    if not args.src_tree.is_dir():
        raise SystemExit(f"--src-tree not a dir: {args.src_tree}")
    args.out_tree.mkdir(parents=True, exist_ok=True)

    n_scenes = 0
    n_humanized = 0
    n_kept_abu = 0
    n_skipped = 0

    with sqlite3.connect(args.db) as conn:
        for scene_dir in sorted(args.src_tree.iterdir()):
            if not scene_dir.is_dir():
                continue
            sid = scene_dir.name
            n_scenes += 1
            out_scene = args.out_tree / sid
            out_scene.mkdir(parents=True, exist_ok=True)

            for child in scene_dir.iterdir():
                if child.name == "output":
                    continue
                _link_or_copy(child, out_scene / child.name, args.symlink)

            src_output = scene_dir / "output"
            out_output = out_scene / "output"
            out_output.mkdir(exist_ok=True)
            for child in src_output.iterdir():
                if child.name == "descriptions":
                    continue
                _link_or_copy(child, out_output / child.name, args.symlink)

            src_desc = src_output / "descriptions"
            out_desc = out_output / "descriptions"
            out_desc.mkdir(exist_ok=True)

            # Pass 1: per-frame <frame>.json files
            picked_per_frame: dict[str, tuple] = {}  # frame_id -> pick row
            for desc_file in sorted(src_desc.iterdir()):
                if desc_file.suffix not in (".json", ".txt"):
                    continue
                if desc_file.suffix == ".txt":
                    _link_or_copy(desc_file, out_desc / desc_file.name, args.symlink)
                    continue
                if desc_file.name == "all_descriptions.json":
                    continue  # handled in pass 2
                if desc_file.name.endswith("_parsed.json"):
                    continue

                fid = desc_file.stem
                rows = _human_descriptions_for(conn, sid, fid, args.skip_flagged)
                if (not rows) and args.skip_flagged:
                    rows = _human_descriptions_for(conn, sid, fid, skip_flagged=False)
                pick = _pick_description(rows, args.pick)

                if pick is None:
                    if args.keep_abu_on_missing:
                        _link_or_copy(desc_file, out_desc / desc_file.name, args.symlink)
                        n_kept_abu += 1
                    else:
                        n_skipped += 1
                    continue

                annotator_id, text, word_count, submitted_at, flagged, flag_reason = pick
                picked_per_frame[fid] = pick

                src = json.loads(desc_file.read_text())
                src["description"] = text
                src["_describer"] = f"human:{annotator_id[:8]}"
                src["_word_count"] = word_count
                src["_submitted_at"] = submitted_at
                if flagged:
                    src["_flagged"] = True
                    src["_flag_reason"] = flag_reason

                (out_desc / desc_file.name).write_text(json.dumps(src, indent=2))
                n_humanized += 1

            # Pass 2: all_descriptions.json (the master list the pipeline reads)
            src_all = src_desc / "all_descriptions.json"
            if src_all.exists():
                entries = json.loads(src_all.read_text())
                for entry in entries:
                    fid = str(entry.get("image_index") or entry.get("frame_id") or "")
                    if fid in picked_per_frame:
                        ann, text, wc, ts, flagged, reason = picked_per_frame[fid]
                        entry["description"] = text
                        entry["_describer"] = f"human:{ann[:8]}"
                        entry["_word_count"] = wc
                        entry["_submitted_at"] = ts
                        if flagged:
                            entry["_flagged"] = True
                            entry["_flag_reason"] = reason
                    elif args.keep_abu_on_missing:
                        # leave entry untouched (Abu's original description)
                        pass
                    # else: entry stays as Abu's original — Abu can drop frames
                    # himself if he wants strict humans-only coverage
                (out_desc / "all_descriptions.json").write_text(json.dumps(entries, indent=2))

    print(
        f"[build_abu_tree] scenes={n_scenes}, frames humanized={n_humanized}, "
        f"frames kept-as-Abu={n_kept_abu}, frames skipped (no human desc)={n_skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
