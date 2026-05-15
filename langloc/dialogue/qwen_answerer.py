"""Qwen-based dialog answerer (Master report appendix A.6 protocol).

For ScanNet evaluation the original LangLoc paper uses Qwen2.5-1.5B-Instruct
to answer dialog yes/no questions from ground-truth reference-frame metadata
(description, visible objects, spatial relations).  This module implements
that protocol so the rebuttal can re-derive the published Tab. 4(b) numbers.

Reference: Master report Appendix A.6 (Qwen-based Answering Protocol).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from langloc.dialogue.question_pool import Question
from langloc.dialogue.semantics import rel_item_to_tuple, relation_phrase

_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class QwenAnswerer:
    """Loaded Qwen model + tokenizer with the protocol prompt template."""

    tokenizer: Any
    model: Any
    device: str

    @classmethod
    def load(cls, device: Optional[str] = None) -> "QwenAnswerer":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(_MODEL_ID)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        mdl = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=dtype, device_map=device
        )
        mdl.eval()
        return cls(tokenizer=tok, model=mdl, device=device)


@dataclass
class QwenFrameContext:
    """Per-scene GT-frame metadata used to construct the prompt."""

    description: str
    visible_labels: List[str]
    rel_triples: List[Tuple[str, str, str]]


_SYSTEM_PROMPT = (
    "You are answering yes/no questions about a single 3D scene viewpoint. "
    "You will be given (1) a short description of what the viewpoint sees, "
    "(2) the list of objects visible from that viewpoint, and (3) the spatial "
    "relations between those objects. Answer EACH question with exactly one "
    "of: yes, no, uncertain. Answer based ONLY on the provided context. If "
    "the question concerns an object or relation that is not mentioned in "
    "the context, answer 'uncertain'."
)


def _format_relations(rel_triples: List[Tuple[str, str, str]]) -> str:
    if not rel_triples:
        return "(none recorded)"
    parts = []
    for s, r, o in rel_triples:
        parts.append(f"  - {s} {relation_phrase(r)} {o}")
    return "\n".join(parts)


def _build_prompt(
    q: Question,
    label_pool: List[str],
    rel_pool: List[Any],
    ctx: QwenFrameContext,
) -> str:
    if q.qtype == "label":
        lab = label_pool[q.idx]
        question_text = f"Do you see a {lab} from this viewpoint?"
    else:
        s, r, o = rel_item_to_tuple(rel_pool[q.idx])
        question_text = f"Is the {s} {relation_phrase(r)} the {o} from this viewpoint?"

    visible_str = ", ".join(sorted(ctx.visible_labels)) if ctx.visible_labels else "(none recorded)"
    relations_str = _format_relations(ctx.rel_triples)

    user_msg = (
        f"Description: {ctx.description.strip() or '(no description provided)'}\n"
        f"Visible objects: {visible_str}\n"
        f"Spatial relations:\n{relations_str}\n\n"
        f"Question: {question_text}\n"
        f"Answer with exactly one word: yes, no, or uncertain."
    )
    return user_msg


def qwen_answer(
    q: Question,
    label_pool: List[str],
    rel_pool: List[Any],
    ctx: QwenFrameContext,
    answerer: QwenAnswerer,
    max_new_tokens: int = 8,
) -> str:
    """Return ``"y"``, ``"n"`` or ``"u"`` for question ``q``.

    Builds the Master Appendix A.6 prompt from the GT-frame context, runs a
    greedy decode on the loaded Qwen model, and maps the first
    yes/no/uncertain token in the reply to the dialog symbol set.
    """
    user_msg = _build_prompt(q, label_pool, rel_pool, ctx)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    prompt = answerer.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = answerer.tokenizer(prompt, return_tensors="pt").to(answerer.device)
    with torch.inference_mode():
        out_ids = answerer.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=answerer.tokenizer.eos_token_id,
        )
    new_ids = out_ids[0, inputs.input_ids.shape[1]:]
    text = answerer.tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()

    # Parse: first token wins; "uncertain" / "unsure" / "unknown" → u; default u
    head = text.split()[0] if text.split() else ""
    head = head.strip(".,!?:;\"'")
    if head.startswith("y"):
        return "y"
    if head.startswith("n"):
        return "n"
    if head.startswith("u") or head in ("idk", "maybe", "unclear"):
        return "u"
    return "u"


def build_frame_context(
    description: str,
    visible_labels: Set[str],
    rel_triples: Set[Tuple[str, str, str]],
) -> QwenFrameContext:
    return QwenFrameContext(
        description=description or "",
        visible_labels=sorted(visible_labels),
        rel_triples=sorted(rel_triples),
    )
