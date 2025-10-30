#!/usr/bin/env python3
"""
Room View Narrower
------------------
Interactively narrow down the correct room view from a fixed set of candidate views by asking
the fewest, most-informative questions. After each user input:
  1) Prints the CURRENT set of possible views (dictionary: view -> items)
  2) Asks a yes/no question "Do you see a <item>?" choosing the rarest (most unique) item
     among remaining candidates. If multiple tie, chooses randomly.
Keep answering until one view remains.

DATA INPUT
- If a CSV named "views.csv" is present in the working directory, it is used.
  CSV format example (first column is the view name, then 10 items):
    view,item1,item2,...,item10
    View 1,sofa,coffee table,tv,window,curtain,bookshelf,rug,armchair,lamp,plant
- Otherwise, a built-in default dataset of 8 views x 10 items is used.

USAGE
  python room_view_narrower.py
"""

from __future__ import annotations
import csv
import os
import random
import sys
import textwrap
from typing import Dict, List, Set, Tuple
import json



random.seed()  # non-deterministic by default
  
# --------- Import views ---------
with open("data/pose_visible_objects/visible_objects_pose_run1.json", "r") as f:
    pose_data = json.load(f)
POSE_VIEWS: Dict[str, List[str]] = {}
for i,entry in enumerate(pose_data):
    pose_idx = i+1
    visible_objs = [obj["object"] for obj in entry["visible_objects"]]
    view_name = f"Pose {pose_idx}"
    POSE_VIEWS[view_name] = visible_objs
    print(POSE_VIEWS)

# --------- Default dataset (used if views.csv not found) ---------
DEFAULT_VIEWS: Dict[str, List[str]] = {
    "View 1 - TV Wall": [
        "tv", "media console", "soundbar", "remote", "wall art",
        "floor lamp", "rug", "coffee table", "sofa", "plant",
    ],
    "View 2 - Sofa Corner": [
        "sofa", "throw pillow", "blanket", "side table", "floor lamp",
        "rug", "coffee table", "tv", "plant", "window",
    ],
    "View 3 - Window Nook": [
        "window", "curtain", "radiator", "armchair", "side table",
        "floor lamp", "plant", "bookshelf", "rug", "wall art",
    ],
    "View 4 - Workspace": [
        "desk", "chair", "monitor", "keyboard", "mouse",
        "bookshelf", "desk lamp", "notebook", "trash bin", "rug",
    ],
    "View 5 - Dining Area": [
        "dining table", "chair", "pendant light", "placemat", "vase",
        "sideboard", "wall art", "plant", "window", "rug",
    ],
    "View 6 - Entryway": [
        "door", "shoe rack", "coat rack", "mirror", "bench",
        "umbrella stand", "doormat", "plant", "wall hooks", "console table",
    ],
    "View 7 - Kitchenette": [
        "counter", "sink", "stove", "microwave", "fridge",
        "kettle", "cutting board", "fruit bowl", "trash bin", "bar stool",
    ],
    "View 8 - Reading Nook": [
        "armchair", "floor lamp", "bookshelf", "side table", "throw pillow",
        "blanket", "rug", "plant", "magazine rack", "window",
    ],
}
def load_views(): 
    if POSE_VIEWS:
        print("Using POSE_VIEWS dataset with", len(POSE_VIEWS), "views.")
        return POSE_VIEWS
    else: 
        print("Using DEFAULT_VIEWS dataset with", len(DEFAULT_VIEWS), "views.")
        return DEFAULT_VIEWS
    
# def load_views_from_csv(path: str) -> Dict[str, List[str]]:
#     views: Dict[str, List[str]] = {}
#     with open(path, newline="", encoding="utf-8") as f:
#         reader = csv.reader(f)
#         header = next(reader, None)
#         if not header or len(header) < 2:
#             print("[WARN] CSV header should be like: view,item1,item2,...", file=sys.stderr)
#         for row in reader:
#             if not row:
#                 continue
#             view_name = row[0].strip()
#             items = [c.strip() for c in row[1:] if c.strip()]
#             if not view_name or not items:
#                 continue
#             views[view_name] = items
#     if not views:
#         print("[WARN] CSV produced no valid rows. Falling back to default dataset.", file=sys.stderr)
#         return DEFAULT_VIEWS.copy()
#     return views


# ---------- Utility functions ----------

def normalize_token(s: str) -> str:
    return " ".join(s.lower().strip().split())


def normalize_items(items: List[str]) -> List[str]:
    return [normalize_token(x) for x in items if normalize_token(x)]


def as_sets(views: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    return {k: set(normalize_items(v)) for k, v in views.items()}


def _token(s: str) -> str:
    return " ".join(s.lower().strip().split())

def _matches(term: str, obj: str) -> bool:
    """
    'chair' matches 'chair', 'chair_1', 'chair_2', ...
    Works with multi-word bases too, e.g. 'trash can' -> 'trash can_3'
    """
    t = _token(term)
    o = _token(obj)
    return o == t or o.startswith(t + "_")

def _any_match(term: str, items) -> bool:
    return any(_matches(term, it) for it in items)

def filter_candidates_by_observed(candidates, present, absent):
    filtered = {}
    for pose, objs in candidates.items():
        # must contain all present items (by base-name match)
        if not all(_any_match(p, objs) for p in present):
            continue
        # must contain none of the absent items (by base-name match)
        if any(_any_match(a, objs) for a in absent):
            continue
        filtered[pose] = objs
    return filtered

def rarest_item(candidates: Dict[str, Set[str]],
                asked_or_known: Set[str]) -> Tuple[str, int]:
    if not candidates:
        return ("", 0)

    # Count per-item frequency among remaining candidates, excluding already asked/known
    freq: Dict[str, int] = {}
    for vitems in candidates.values():
        for it in vitems:
            if it in asked_or_known:
                continue
            freq[it] = freq.get(it, 0) + 1

    if not freq:
        return ("", 0)

    n = len(candidates)
    # Keep only items that actually split the set (appear in some but not all)
    split_items = [(it, c) for it, c in freq.items() if 0 < c < n]
    if not split_items:
        # no discriminative question left
        return ("", 0)

    min_freq = min(c for _, c in split_items)
    rare = [it for it, c in split_items if c == min_freq]
    choice = random.choice(rare)
    return (choice, min_freq)



def pretty_dict(d: Dict[str, Set[str]]) -> str:
    if not d:
        return "{}"
    lines = ["{"]
    for k, v in d.items():
        items_sorted = ", ".join(sorted(v))
        lines.append(f'  "{k}": [{items_sorted}]')
    lines.append("}")
    return "\n".join(lines)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def main():
    views_raw = load_views()
    views = as_sets(views_raw)

    # Initial prompt
    print_header("Room View Narrower")
    print("Type what you currently see (comma-separated). Example: TV, window, sofa")
    user = input("> What do you see? ").strip()
    seen_now = normalize_items([x for x in user.split(",") if x.strip()])

    observed_present: Set[str] = set(seen_now)
    observed_absent: Set[str] = set()
    asked_items: Set[str] = set(seen_now)

    # Initial pruning
    candidates = filter_candidates_by_observed(views, observed_present, observed_absent)

    # If no candidates, relax strategy: find closest matches by coverage
    if not candidates:
        print("\n[!] No exact matches. Relaxing to closest coverage of your items...\n")
        # Score by number of observed present items found (descending), break ties by fewest extra items
        scored = []
        for vname, vitems in views.items():
            coverage = len(observed_present & vitems)
            extras = len(vitems - observed_present)
            scored.append((coverage, -extras, vname, vitems))
        scored.sort(reverse=True)
        # Keep top K by coverage
        top_cov = scored[0][0] if scored else 0
        candidates = {vn: vi for cov, _ne, vn, vi in scored if cov == top_cov and cov > 0}
        if not candidates:
            # Nothing in common at all; keep all and proceed to query
            candidates = views.copy()

    # Interaction loop
    step = 1
    while True:
        print_header(f"Step {step}: Current possible views ({len(candidates)})")
        print(pretty_dict(candidates))

        if len(candidates) <= 1:
            if len(candidates) == 1:
                winner = next(iter(candidates))
                print("\n Identified view:", winner)
            else:
                print("\n[!] No candidates remain. The answers may be inconsistent with the dataset.")
            break

        # Pick the rarest (most unique) next item to ask about
        item, freq = rarest_item(candidates, asked_items | observed_absent | observed_present)
        if not item:
            print("\n[!] No further distinguishing items to ask. Consider revising answers.")
            break

        q = f'Do you see a "{item}"? (y/n/unsure): '
        ans = input(q).strip().lower()
        while ans not in {"y", "n", "u", "unsure", "yes", "no"}:
            ans = input('Please answer "y", "n", or "unsure": ').strip().lower()

        # asked_items.add(item)

        if ans in {"y", "yes"}:
            observed_present.add(item)
        elif ans in {"n", "no"}:
            observed_absent.add(item)
        else:
            # 'unsure' -> skip updating knowledge but avoid re-asking the same item
            pass

        new_candidates = filter_candidates_by_observed(candidates, observed_present, observed_absent)
        print("Filtered candidates after answer:", new_candidates)
        # If saying "no" nuked everything, roll back that last "absent"
        if not new_candidates and ans in {"n", "no"}:
            observed_absent.discard(item)
            new_candidates = filter_candidates_by_observed(candidates, observed_present, observed_absent)

        # If filtering made no progress (same set), just continue; item is already in asked_items
        if new_candidates and len(new_candidates) == len(candidates) and new_candidates.keys() == candidates.keys():
            # no change — carry on to next question
            pass
        elif new_candidates:
            candidates = new_candidates
        # else: keep current candidates (either rollback case or nothing changed)

        step += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Bye!")
        sys.exit(0)
