#!/usr/bin/env python3
"""LLM-as-answerer evaluation for the dialogue disambiguation pipeline.

Replaces the oracle answerer with a local causal LM (e.g. Qwen2.5-1.5B-Instruct)
that answers yes/no/uncertain questions based on per-frame scene descriptions.

Usage::

    python -m langloc.eval.llm_answerer \
        --dialogue_script  path/to/dialogue_entry.py \
        --candidates_json  path/to/candidates.json \
        --dataset_root     path/to/scans \
        --output_json      results.json \
        --model_name       Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import builtins
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional



# ── Qwen model (loaded once globally) ────────────────────────────────────────
_tokenizer = None
_model = None
_device = "cpu"


def load_model(model_name: str, load_in_4bit: bool = False) -> None:
    """Load a HuggingFace causal LM and tokenizer into module globals.

    Args:
        model_name: HuggingFace model name or local path.
        load_in_4bit: If ``True``, quantize to 4-bit via bitsandbytes.
    """
    global _tokenizer, _model, _device
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading {model_name} …")
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs: dict = dict(trust_remote_code=True)
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16 if _device == "cuda" else torch.float32
        kwargs["device_map"] = "auto"

    _model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    _model.eval()
    print(f"Model ready on {_device}.")


def ask_llm(prompt: str, max_new_tokens: int = 8) -> str:
    """Generate a short answer from the loaded LLM.

    Args:
        prompt: The user prompt to send to the model.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The decoded, lowercased, stripped model output.
    """
    import torch
    messages = [{"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer([text], return_tensors="pt")
    if _device != "cpu":
        inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()


def parse_llm_answer(raw: str) -> str:
    """Parse raw LLM output into a single-character answer code.

    Args:
        raw: Raw text output from the LLM.

    Returns:
        ``"y"`` for yes, ``"n"`` for no, or ``"u"`` for uncertain/unknown.
    """
    r = raw.lower().strip()
    if r.startswith("yes"): return "y"
    if r.startswith("no"):  return "n"
    if r.startswith("y"):   return "y"
    if r.startswith("n"):   return "n"
    return "u"


# ── Scene description helpers ────────────────────────────────────────────────

def load_scene_descriptions(scene_dir: Path) -> List[Dict[str, Any]]:
    """Load all frame description JSONs from a scene directory.

    Args:
        scene_dir: Scene root directory containing ``output/descriptions/``.

    Returns:
        List of frame description dicts, one per JSON file entry.
    """
    desc_dir = scene_dir / "output" / "descriptions"
    frames = []
    for p in sorted(desc_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                frames.append(data)
            elif isinstance(data, list):
                frames.extend(item for item in data if isinstance(item, dict))
        except Exception:
            pass
    return frames


def build_scene_context(frames: List[Dict[str, Any]]) -> str:
    """Build a compact context string from all scene frames.

    Used as a fallback when the specific reference frame is unavailable.

    Args:
        frames: List of frame description dicts.

    Returns:
        A multi-line context string summarising descriptions, objects,
        and spatial relations across all frames.
    """
    if not frames:
        return "No scene descriptions available."

    label_scores: Dict[str, float] = {}
    for fr in frames:
        vo = fr.get("visible_objects", {})
        obj_iter = vo.values() if isinstance(vo, dict) else vo
        for obj in obj_iter:
            if not isinstance(obj, dict):
                continue
            lab = obj.get("label", "")
            pct = float(obj.get("pixel_percent", 0))
            if lab:
                label_scores[lab] = max(label_scores.get(lab, 0), pct)
    top_labels = sorted(label_scores, key=lambda l: -label_scores[l])

    rel_set: set = set()
    for fr in frames:
        for r in fr.get("spatial_relations", []):
            s, rel, o = r.get("subject", ""), r.get("relation", ""), r.get("object", "")
            if s and rel and o:
                rel_set.add(f"{s} is {rel} {o}")

    seen_desc: set = set()
    descs: List[str] = []
    for fr in frames:
        d = fr.get("description", "").strip()
        if d and d not in seen_desc:
            seen_desc.add(d)
            descs.append(d)

    parts = []
    if descs:
        parts.append("Scene descriptions:\n" + "\n".join(f"- {d}" for d in descs))
    if top_labels:
        parts.append("Visible objects: " + ", ".join(top_labels[:30]))
    if rel_set:
        parts.append("Spatial relations: " + "; ".join(sorted(rel_set)[:40]))

    return "\n\n".join(parts)


def load_reference_frame(scene_dir: Path, frame_id: str) -> Optional[Dict[str, Any]]:
    """Load a specific reference frame JSON by frame ID.

    First tries a direct filename match, then falls back to scanning all
    description JSONs for a matching ``image_index`` field.

    Args:
        scene_dir: Scene root directory containing ``output/descriptions/``.
        frame_id: Frame identifier to look up.

    Returns:
        The frame description dict, or ``None`` if not found.
    """
    desc_dir = scene_dir / "output" / "descriptions"
    candidate = desc_dir / f"{frame_id}.json"
    if candidate.exists():
        data = json.loads(candidate.read_text())
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data:
            return data[0]
    for p in sorted(desc_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            items = [data] if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                if str(item.get("image_index", "")) == str(frame_id):
                    return item
        except Exception:
            pass
    return None


def build_frame_context(frame: Dict[str, Any]) -> str:
    """Build a compact context string from a single reference frame.

    Args:
        frame: A frame description dict with optional ``description``,
            ``visible_objects``, and ``spatial_relations`` keys.

    Returns:
        A multi-line context string for LLM prompting.
    """
    parts = []

    desc = frame.get("description", "").strip()
    if desc:
        parts.append(f"Scene description: {desc}")

    vo = frame.get("visible_objects", {})
    obj_iter = vo.values() if isinstance(vo, dict) else (vo if isinstance(vo, list) else [])
    labels = sorted({
        obj.get("label", "")
        for obj in obj_iter
        if isinstance(obj, dict) and obj.get("label", "")
    })
    if labels:
        parts.append("Visible objects: " + ", ".join(labels))

    rels = frame.get("spatial_relations", [])
    rel_lines = [
        f"- {r.get('subject', '')} is {r.get('relation', '')} {r.get('object', '')}"
        for r in rels if isinstance(r, dict)
    ]
    if rel_lines:
        parts.append("Spatial relations:\n" + "\n".join(rel_lines[:40]))

    return "\n\n".join(parts) if parts else "No scene data available."


def build_llm_prompt(question_text: str, context: str) -> str:
    """Build the yes/no/uncertain prompt for the LLM.

    Args:
        question_text: The disambiguating question to pose.
        context: Scene context string (objects, relations, descriptions).

    Returns:
        The full prompt string ready for LLM inference.
    """
    return (
        "You are looking at a single camera view inside a room.\n\n"
        f"{context}\n\n"
        f"Question: {question_text}\n\n"
        "Answer with exactly one word — 'yes', 'no', or 'uncertain' — "
        "based only on the description and object list above. Do not explain."
    )


# ── Import helper ────────────────────────────────────────────────────────────

from langloc.eval import import_module_from_path


# ── Question text parsers ────────────────────────────────────────────────────
RE_LABEL = re.compile(r"Ask\[label\]:\s+Do you see \*\*(.+?)\*\*", re.IGNORECASE)
RE_REL = re.compile(r"Ask\[rel\s*\]:\s+Is \*\*(.+?)\*\*\s+(.+?)\s+\*\*(.+?)\*\*", re.IGNORECASE)


# ── Shared mutable state (used by hooks) ─────────────────────────────────────
_state: Dict[str, Any] = {
    "scene_id": None,
    "context": "",
    "frame_id": None,
    "output_lines": [],
    "answer_log": [],
    "last_question_line": "",
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the LLM answerer CLI.

    Returns:
        Parsed argument namespace.
    """
    ap = argparse.ArgumentParser(
        description="Replace oracle with a local LLM for dialogue question answering."
    )
    ap.add_argument("--dialogue_script", type=Path, required=True,
                    help="Path to the dialogue entry-point script.")
    ap.add_argument("--candidates_json", type=Path, required=True,
                    help="Path to the evaluation candidates JSON.")
    ap.add_argument("--dataset_root", type=Path, required=True,
                    help="Root directory containing scene scan folders.")
    ap.add_argument("--output_json", type=Path, default=Path("llm_results.json"),
                    help="Output path for per-scene results JSON.")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HuggingFace model name/path for the answerer LLM.")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Load model in 4-bit quantization (requires bitsandbytes).")
    ap.add_argument("--max_pool_frames", type=int, default=30)
    ap.add_argument("--eval_mode", default="sequential")
    ap.add_argument("--question_strategy", default="ig")
    ap.add_argument("--auto_relax", action="store_true", default=False)
    ap.add_argument("--rel_min_answerable", type=float, default=0.1)
    ap.add_argument("--include_predicted_pose", action="store_true", default=True)
    ap.add_argument("--show_gt_debug", action="store_true", default=True)
    return ap.parse_args()


def main() -> None:
    """Run the LLM answerer over all scenes and save per-scene results."""
    args = parse_args()

    load_model(args.model_name, load_in_4bit=args.load_in_4bit)

    raw = json.loads(args.candidates_json.read_text())
    scene_ids: List[str] = []
    scene_frame_id: Dict[str, str] = {}
    seen: set = set()
    for e in raw.get("scenes", []):
        if not isinstance(e, dict):
            continue
        sid = e.get("scene_id")
        fid = str(e.get("frame_id", ""))
        if sid and sid not in seen:
            seen.add(sid)
            scene_ids.append(sid)
            scene_frame_id[sid] = fid
    print(f"Found {len(scene_ids)} scenes.\n")

    # 3. Import dialogue module
    dlg = import_module_from_path(args.dialogue_script, "dlg_mod_llm")

    # 4. Install print/input hooks
    orig_print = builtins.print
    orig_input = builtins.input

    def hooked_print(*a: Any, **kw: Any) -> None:
        """Intercept print calls to capture output and detect question lines."""
        orig_print(*a, **kw)
        sep = kw.get("sep", " ")
        end = kw.get("end", "\n")
        line = sep.join(str(x) for x in a) + end
        _state["output_lines"].append(line)
        stripped = line.strip()
        if stripped.startswith("Ask[label]") or stripped.startswith("Ask[rel"):
            _state["last_question_line"] = stripped

    def hooked_input(prompt: str = "") -> str:
        """Intercept input calls, query the LLM, and return the parsed answer."""
        orig_print(prompt, end="", flush=True)
        question_line = _state["last_question_line"]

        m_label = RE_LABEL.search(question_line)
        m_rel = RE_REL.search(question_line)
        if m_label:
            q_str = f"Do you see a {m_label.group(1)} in this scene?"
        elif m_rel:
            q_str = f"Is the {m_rel.group(1)} {m_rel.group(2)} the {m_rel.group(3)}?"
        else:
            q_str = question_line

        llm_prompt = build_llm_prompt(q_str, _state["context"])
        raw_ans = ask_llm(llm_prompt)
        ans = parse_llm_answer(raw_ans)

        orig_print(f"[llm -> '{raw_ans}' -> {ans}]")
        _state["output_lines"].append(f"[llm -> '{raw_ans}' -> {ans}]\n")
        _state["answer_log"].append({
            "question": q_str,
            "raw_line": question_line,
            "llm_raw": raw_ans,
            "answer": ans,
        })
        return ans

    builtins.print = hooked_print
    builtins.input = hooked_input

    # 5. Wrap run_entry to reset state per scene
    orig_run_entry = dlg.run_entry

    def run_entry_wrapped(entry: Dict[str, Any], dargs: Any) -> Any:
        """Reset per-scene state and load context before delegating to the original entry."""
        sid = entry.get("scene_id", "")
        fid = scene_frame_id.get(sid, "")
        _state["scene_id"] = sid
        _state["frame_id"] = fid
        _state["output_lines"] = []
        _state["answer_log"] = []
        _state["last_question_line"] = ""

        scene_dir = Path(str(dargs.dataset_root)) / sid
        ref_frame = load_reference_frame(scene_dir, fid)
        if ref_frame:
            _state["context"] = build_frame_context(ref_frame)
            orig_print(f"  [llm context] frame_id={fid} | "
                       f"objects={len(ref_frame.get('visible_objects', {}))} | "
                       f"relations={len(ref_frame.get('spatial_relations', []))}")
        else:
            frames = load_scene_descriptions(scene_dir)
            _state["context"] = build_scene_context(frames)
            orig_print(f"  [llm context] frame_id={fid} not found, using all {len(frames)} frames")

        return orig_run_entry(entry, dargs)

    dlg.run_entry = run_entry_wrapped

    # 6. Run all scenes
    results: List[Dict] = []

    for i, sid in enumerate(scene_ids, 1):
        orig_print(f"\n{'=' * 60}")
        orig_print(f"[{i:3d}/{len(scene_ids)}] {sid}")
        orig_print("=" * 60)

        _state["scene_id"] = sid
        _state["output_lines"] = []
        _state["answer_log"] = []
        _state["last_question_line"] = ""

        returncode = 0
        error_text = ""
        saved_argv = sys.argv[:]
        try:
            sys.argv = [
                str(args.dialogue_script),
                "--candidates_json", str(args.candidates_json),
                "--dataset_root", str(args.dataset_root),
                "--only_scene_id", sid,
                "--limit", "1",
                "--eval_mode", args.eval_mode,
                "--question_strategy", args.question_strategy,
                "--answer_mode", "interactive",
                "--max_pool_frames", str(args.max_pool_frames),
                "--rel_min_answerable", str(args.rel_min_answerable),
                *(["--auto_relax"] if args.auto_relax else []),
                *(["--include_predicted_pose"] if args.include_predicted_pose else []),
                *(["--show_gt_debug"] if args.show_gt_debug else []),
            ]
            dlg.main()
        except SystemExit as e:
            returncode = int(e.code) if e.code is not None else 0
        except Exception:
            returncode = 1
            error_text = traceback.format_exc()
            orig_print(error_text)
        finally:
            sys.argv = saved_argv
            builtins.print = hooked_print
            builtins.input = hooked_input

        results.append({
            "scene_id": sid,
            "returncode": returncode,
            "error": error_text,
            "stdout": "".join(_state["output_lines"]),
            "qa_log": [
                {
                    "question": qa["question"],
                    "raw_line": qa["raw_line"],
                    "llm_raw": qa["llm_raw"],
                    "answer": qa["answer"],
                }
                for qa in _state["answer_log"]
            ],
        })

        args.output_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False)
        )
        ok_so_far = sum(1 for r in results if r["returncode"] == 0)
        orig_print(f"  -> saved {len(results)} scenes ({ok_so_far} OK) to {args.output_json}")

    # 7. Restore and final summary
    builtins.print = orig_print
    builtins.input = orig_input

    ok = sum(1 for r in results if r["returncode"] == 0)
    err = len(results) - ok
    print(f"\nFinished. {ok}/{len(results)} OK, {err} errors.")
    print(f"Results saved -> {args.output_json}")


if __name__ == "__main__":
    main()
