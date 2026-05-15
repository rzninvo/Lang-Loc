#!/usr/bin/env python3
"""Generate first-person scene descriptions from keyframe images using
the GPT-5.5 vision API.

The describer sees only the image — no GT-frame metadata, no
visible-object list, no spatial-relation triples. That's the whole
point: this is the closed-loop-broken half of the rebuttal experiment.

The downstream localization pipeline still grounds the parsed scene
graph against the GT frame's ``visible_objects`` (this is how every
paper baseline works — the limitation is documented in
docs/rebuttal_report/rebuttal_report.tex §7.2). To stay compatible
with that pipeline, we copy ``scene_pose``, ``image_index``,
``visible_objects`` and ``spatial_relations`` from the existing
per-frame metadata into the new per-frame JSON; the only field we
replace is ``description``.

Output layout (the tree the localization pipeline reads when invoked
with ``paths.query_root=<out-root>``):

    <out-root>/<scene_id>/output/descriptions/<frame_id>.json

Each output JSON has, at minimum: ``description``, ``image_index``,
``scene_pose``, ``visible_objects``, ``spatial_relations``,
``_describer`` (model + seed + run-id metadata).

Run manifest with prompt + model + seed + git SHA is written to
``<out-root>/run_manifest.json`` for reproducibility per CLAUDE.md §0.

Resume-by-default: existing valid ``<frame_id>.json`` files are
skipped. Pass ``--overwrite`` to redo. Per-frame failures are
recorded as ``<frame_id>.error.json`` so retry sees them and skips
unless ``--retry-errors`` is set.

Cost cap: ``--max-cost-usd`` (default 35) and
``LANGLOC_GPT_MAX_USD`` env var, whichever is lower. Both are checked
on every successful API response; the script aborts cleanly when
cumulative cost would exceed.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Matched-stimulus prompt: identical to the website's instruction to humans.
# The website asks first-person spatial descriptions because that's the
# template the localization pipeline's parser converts cleanly. Both the
# VLM and humans get the same prompt; the comparison measures describer
# accuracy under a fixed instruction (NOT natural-distribution diff).
PROMPT_VLM_FIRST_PERSON = (
    "You're standing where the camera is, looking at this room. "
    "In your own words, describe what you see, in first person, in 2-4 "
    "sentences. Mention what's directly in front of you and what's on "
    "your left and right. Mention specific distinctive objects (a "
    "particular cabinet, a coffee maker, a window) and how they relate "
    "to you in space. Don't list — describe naturally as if telling a "
    "friend on the phone where you are."
)

# OpenAI vision pricing as of 2026-05-10: $8/M input tokens (text+image),
# $30/M output tokens. We compute cost from response.usage so the cap is
# accurate even if the model is upgraded.
PRICE_INPUT_PER_M = 8.0
PRICE_OUTPUT_PER_M = 30.0

ERROR_PLACEHOLDER = "[Error generating description]"


@dataclass
class CostTracker:
    spent_usd: float = 0.0
    n_calls: int = 0
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    cap_usd: float = 35.0

    def add(self, usage) -> None:
        # CompletionUsage: prompt_tokens, completion_tokens
        in_t = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_t = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = (in_t * PRICE_INPUT_PER_M + out_t * PRICE_OUTPUT_PER_M) / 1_000_000.0
        self.spent_usd += cost
        self.n_input_tokens += in_t
        self.n_output_tokens += out_t
        self.n_calls += 1

    def would_exceed(self, est_call_cost: float = 0.02) -> bool:
        return (self.spent_usd + est_call_cost) >= self.cap_usd

    def summary(self) -> dict:
        return {
            "spent_usd": round(self.spent_usd, 4),
            "calls": self.n_calls,
            "input_tokens": self.n_input_tokens,
            "output_tokens": self.n_output_tokens,
            "cap_usd": self.cap_usd,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["3rscan", "scannet"])
    p.add_argument("--manifest", type=Path, required=True,
                   help="text file with one scene_id per line (in order)")
    p.add_argument("--data-root", type=Path, required=True,
                   help="root containing <scene_id>/output/{color,descriptions}/")
    p.add_argument("--out-root", type=Path, required=True,
                   help="parallel tree to write <scene>/output/descriptions/<frame>.json")
    p.add_argument("--num-scenes", type=int, default=100)
    p.add_argument("--num-frames-per-scene", type=int, default=0,
                   help="0 = all keyframes referenced in all_descriptions.json")
    p.add_argument("--ordered", action="store_true",
                   help="take scenes in manifest order (default behaviour anyway)")
    p.add_argument("--model", default="gpt-5.5",
                   help="OpenAI vision model id (pinned for reproducibility)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-completion-tokens", type=int, default=512)
    p.add_argument("--prompt-file", type=Path, default=None,
                   help="path to a text file overriding the default prompt")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--max-cost-usd", type=float,
                   default=float(os.environ.get("LANGLOC_GPT_MAX_USD", "35")),
                   help="hard cost cap; aborts cleanly when reached")
    p.add_argument("--overwrite", action="store_true",
                   help="re-call GPT even if a valid <frame>.json already exists")
    p.add_argument("--retry-errors", action="store_true",
                   help="re-call GPT on frames whose previous run wrote a *.error.json")
    p.add_argument("--smoke-test", action="store_true",
                   help="hard-cap concurrency=2 and num-scenes/frames small for the smoke test")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O HELPERS
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_scene_ids(manifest: Path, num: int) -> list[str]:
    ids = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
    return ids[:num]


def _load_keyframes_meta(scene_dir: Path) -> dict[str, dict]:
    """scene_dir/output/descriptions/all_descriptions.json -> {frame_id: meta}."""
    p = scene_dir / "output" / "descriptions" / "all_descriptions.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out: dict[str, dict] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        fid = str(entry.get("image_index") or entry.get("frame_id") or "")
        if not fid:
            continue
        out[fid] = entry
    return out


def _frame_color_path(scene_dir: Path, frame_id: str, dataset: str) -> Path:
    return scene_dir / "output" / "color" / f"{frame_id}.jpg"


def _existing_is_valid(p: Path) -> bool:
    """A valid prior result: parseable JSON, non-empty description, and
    not the error placeholder."""
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        desc = (d.get("description") or "").strip()
        if not desc or desc.startswith("[Error"):
            return False
        if not d.get("scene_pose"):
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GPT CALL with backoff + Retry-After + jitter
# ---------------------------------------------------------------------------
async def _call_gpt_vision(
    client,
    *,
    image_bytes: bytes,
    prompt: str,
    model: str,
    seed: int,
    max_tokens: int,
    max_retries: int = 5,
) -> tuple[str, object, str]:
    """Returns (description, usage, model_fingerprint).
    Raises on terminal failure (after retries)."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                seed=seed,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_completion_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            fp = str(getattr(resp, "system_fingerprint", "") or "")
            return text, resp.usage, fp
        except Exception as e:
            last_exc = e
            # honour Retry-After if present (RateLimitError / APIError both expose .response sometimes)
            retry_after = None
            r = getattr(e, "response", None)
            if r is not None:
                hdr = getattr(r, "headers", None) or {}
                ra = hdr.get("retry-after") if hasattr(hdr, "get") else None
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        retry_after = None
            if attempt >= max_retries:
                raise
            base = retry_after if retry_after is not None else 2.0 * (2 ** attempt)
            jitter = random.uniform(0, min(base, 5.0))
            sleep = min(60.0, base + jitter)
            print(f"[WARN] call failed: {type(e).__name__}: {e!s:.180s} ; "
                  f"sleeping {sleep:.1f}s before retry {attempt+1}/{max_retries}",
                  flush=True)
            await asyncio.sleep(sleep)
    raise RuntimeError(f"unreachable after retries; last={last_exc}")


# ---------------------------------------------------------------------------
# WORKER
# ---------------------------------------------------------------------------
async def _process_frame(
    *,
    sem: asyncio.Semaphore,
    cost: CostTracker,
    cost_lock: asyncio.Lock,
    aborted: asyncio.Event,
    client,
    scene_id: str,
    frame_id: str,
    src_meta: dict,
    src_image: Path,
    out_path: Path,
    err_path: Path,
    prompt: str,
    model: str,
    seed: int,
    max_tokens: int,
) -> dict:
    """Returns a status dict. Writes the per-frame JSON or error JSON."""
    if aborted.is_set():
        return {"frame_id": frame_id, "scene_id": scene_id, "status": "aborted"}
    if not src_image.exists():
        err = {"error": "source image missing", "src_image": str(src_image)}
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(json.dumps(err, indent=2))
        return {"frame_id": frame_id, "scene_id": scene_id, "status": "no_image"}

    # quick budget check before reading 200 KB and waking up the GPU
    async with cost_lock:
        if cost.would_exceed():
            aborted.set()
            return {"frame_id": frame_id, "scene_id": scene_id, "status": "cost_cap"}

    image_bytes = src_image.read_bytes()

    async with sem:
        if aborted.is_set():
            return {"frame_id": frame_id, "scene_id": scene_id, "status": "aborted"}
        try:
            t0 = time.time()
            text, usage, fp = await _call_gpt_vision(
                client,
                image_bytes=image_bytes,
                prompt=prompt,
                model=model,
                seed=seed,
                max_tokens=max_tokens,
            )
            dt = time.time() - t0
        except Exception as e:
            err = {"error": f"{type(e).__name__}: {e}", "frame_id": frame_id, "scene_id": scene_id}
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(json.dumps(err, indent=2))
            return {"frame_id": frame_id, "scene_id": scene_id, "status": "error", "error": str(e)}

    # cost bookkeeping
    async with cost_lock:
        cost.add(usage)
        if cost.would_exceed(0.0):
            # we just crossed the cap on this call; allow this one but signal abort
            aborted.set()

    # build the per-frame JSON (carry scene_pose + visible_objects + spatial_relations)
    out_obj = {
        "scene_id": scene_id,
        "image_index": frame_id,
        "scene_pose": src_meta.get("scene_pose"),
        "visible_objects": src_meta.get("visible_objects", {}),
        "spatial_relations": src_meta.get("spatial_relations", []),
        "description": text or ERROR_PLACEHOLDER,
        "_describer": {
            "kind": "gpt_vision_image_only",
            "model": model,
            "model_fingerprint": fp,
            "seed": seed,
            "max_completion_tokens": max_tokens,
            "prompt_id": "first_person_v1",
            "wall_seconds": round(dt, 3),
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False))
    # if we wrote a successful result, clear any prior error sentinel
    if err_path.exists():
        try:
            err_path.unlink()
        except OSError:
            pass
    return {"frame_id": frame_id, "scene_id": scene_id, "status": "ok",
            "wall_seconds": round(dt, 3), "spent_usd": round(cost.spent_usd, 4)}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
async def amain(args: argparse.Namespace) -> int:
    # late import so --help works without httpx/openai
    try:
        from openai import AsyncOpenAI
    except Exception as e:
        print(f"[ERROR] could not import openai async client: {e}", file=sys.stderr)
        return 2

    if args.smoke_test:
        args.concurrency = min(args.concurrency, 2)

    # seed everything we can
    random.seed(args.seed)

    prompt = PROMPT_VLM_FIRST_PERSON
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text().strip()
        if not prompt:
            print("[ERROR] --prompt-file is empty", file=sys.stderr)
            return 2

    scene_ids = _load_scene_ids(args.manifest, args.num_scenes)
    if not scene_ids:
        print(f"[ERROR] no scenes in manifest {args.manifest}", file=sys.stderr)
        return 2
    print(f"[run] dataset={args.dataset}  scenes={len(scene_ids)}  "
          f"model={args.model}  seed={args.seed}  concurrency={args.concurrency}  "
          f"cap=${args.max_cost_usd:.2f}", flush=True)

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # write run manifest first so it's preserved even on early abort
    manifest_obj = {
        "dataset": args.dataset,
        "scene_ids": scene_ids,
        "model": args.model,
        "seed": args.seed,
        "max_completion_tokens": args.max_completion_tokens,
        "prompt_id": "first_person_v1",
        "prompt": prompt,
        "git_sha": _git_sha(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_root": str(args.data_root),
        "out_root": str(out_root),
        "max_cost_usd": args.max_cost_usd,
        "concurrency": args.concurrency,
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest_obj, indent=2))

    cost = CostTracker(cap_usd=args.max_cost_usd)
    cost_lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    aborted = asyncio.Event()
    client = AsyncOpenAI()

    tasks = []
    n_scheduled = 0
    n_skipped_existing = 0
    n_skipped_error = 0

    for sid in scene_ids:
        scene_dir = args.data_root / sid
        if not scene_dir.exists():
            print(f"[WARN] scene dir missing: expected={scene_dir}", flush=True)
            continue
        keyframes_meta = _load_keyframes_meta(scene_dir)
        if not keyframes_meta:
            print(f"[WARN] no all_descriptions.json for {sid}, skipping scene", flush=True)
            continue
        out_scene_dir = out_root / sid / "output" / "descriptions"
        for fid, meta in list(keyframes_meta.items())[: args.num_frames_per_scene or len(keyframes_meta)]:
            out_path = out_scene_dir / f"{fid}.json"
            err_path = out_scene_dir / f"{fid}.error.json"
            if not args.overwrite and _existing_is_valid(out_path):
                n_skipped_existing += 1
                continue
            if not args.retry_errors and err_path.exists():
                n_skipped_error += 1
                continue
            src_image = _frame_color_path(scene_dir, fid, args.dataset)
            tasks.append(_process_frame(
                sem=sem, cost=cost, cost_lock=cost_lock, aborted=aborted,
                client=client,
                scene_id=sid, frame_id=fid,
                src_meta=meta, src_image=src_image,
                out_path=out_path, err_path=err_path,
                prompt=prompt,
                model=args.model, seed=args.seed,
                max_tokens=args.max_completion_tokens,
            ))
            n_scheduled += 1
    print(f"[run] scheduled={n_scheduled}  skipped_existing={n_skipped_existing}  "
          f"skipped_prior_error={n_skipped_error}", flush=True)
    if not tasks:
        print("[run] nothing to do.")
        return 0

    # run with progress prints every N frames
    n_done = 0
    n_ok = 0
    n_err = 0
    n_aborted = 0
    t_start = time.time()
    chunk = 25
    for fut in asyncio.as_completed(tasks):
        result = await fut
        n_done += 1
        st = result.get("status")
        if st == "ok":
            n_ok += 1
        elif st == "error":
            n_err += 1
        elif st in ("aborted", "cost_cap"):
            n_aborted += 1
        if n_done % chunk == 0 or st in ("error", "cost_cap"):
            elapsed = time.time() - t_start
            print(f"[run] progress {n_done}/{len(tasks)}  ok={n_ok}  err={n_err}  "
                  f"aborted={n_aborted}  spent=${cost.spent_usd:.4f}  "
                  f"elapsed={elapsed:.1f}s", flush=True)

    final = {
        **manifest_obj,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_scheduled": n_scheduled,
        "n_skipped_existing": n_skipped_existing,
        "n_skipped_prior_error": n_skipped_error,
        "n_ok": n_ok,
        "n_err": n_err,
        "n_aborted_for_cost": n_aborted,
        "cost": cost.summary(),
    }
    (out_root / "run_manifest.json").write_text(json.dumps(final, indent=2))
    print("[run] DONE", json.dumps(final["cost"]), flush=True)
    return 0 if n_err == 0 and n_aborted == 0 else 1


def main() -> None:
    args = parse_args()
    code = asyncio.run(amain(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
