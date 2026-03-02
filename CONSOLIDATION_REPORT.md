# LangLoc ECCV Repo Consolidation Report

> **Purpose:** Context document for future sessions. Describes what was done, what came from where, and what still needs attention.

---

## 1. Overview

Three separate repositories were consolidated into a single **Lang-Loc** release repo for the ECCV 2026 paper "LangLoc: Tell Me What You See" (Paper ID #3307). The paper describes a three-stage pipeline:

1. **Scene Retrieval** (Sec 3.2) — dual-branch GATv2 encoder with CLIP features
2. **Fine Localization** (Sec 3.3) — visibility-based floor-grid scoring with ray-casting
3. **Dialog Disambiguation** (Sec 3.4) — Bayesian yes/no question system

### Source Repositories

| Repo | Path | Role | Config System |
|---|---|---|---|
| **Lang-Loc** | `/home/rohamzn/UZH Uni/Master Project/Lang-Loc/` | Dataset creation + dialogue system | Plain YAML + argparse |
| **whereami-text2sgm** | `/home/rohamzn/UZH Uni/Master Project/whereami-text2sgm/` | Text2SGM models + fine localization (13 files) | Hydra + OmegaConf |
| **VLSG-clean** | `/home/rohamzn/UZH Uni/Master Project/VLSG-clean/` | Colleague's DualSceneAligner for scene retrieval (16 files) | argparse + sys.path hacks |

### Decisions Made

- **Hydra + OmegaConf** chosen as the unified config system
- **`dsf` dependency** in the dialogue module is temporary; left as-is for now
- **Two separate SceneGraph classes** kept intentionally:
  - `src/data_processing/scene_graph.py` (from whereami) — used by localization/models, supports word2vec/ada/clip embeddings, has `to_pyg()`
  - `src/retrieval/scene_graph.py` (from VLSG-clean) — used by retrieval eval, supports 3DSSG/ScanScribe graph types, different node/edge extraction

---

## 2. VLSG-clean Preparation

Before consolidation, a clean version of the colleague's retrieval code was prepared.

### Source

- **Upstream:** `https://github.com/y9miao/VLSG.git` cloned to `/home/rohamzn/UZH Uni/Master Project/VLSG-clean/`
- **Colleague's fork:** `VLSG-TEXT/Documents/SCHOOL/FALL2025/MASTER-PROJECT/VLSG-TEXT/`

### Files Selected

15 essential files were cherry-picked from ~100 files in the colleague's messy repo. ~34 iterative drafts and ~16 one-off utilities were excluded.

| Original Name | Clean Name | Location |
|---|---|---|
| `dual_scene_graph_dataset_518_v2.py` | `dual_scene_graph_dataset.py` | `src/datasets/` |
| `scanscribetoclip_dataset.py` | (unchanged) | `src/datasets/` |
| `train_with_scene_clip_518_v2.py` | `train_dual_scene.py` | `src/trainval/` |
| `eval_518_multitask.py` | `eval_dual_scene.py` | `src/eval/` |
| `eval_518_multitask_original_table1.py` | `eval_dual_scene_table1.py` | `src/eval/` |
| `helper.py` | (unchanged) | `src/eval/` |
| `scene_graph.py` | (unchanged) | `src/eval/` |
| `scene_graph_utils.py` | (unchanged) | `src/eval/` |
| `clip_utils.py` | (unchanged) | `src/eval/` |
| `dual_scene_aligner.py` | (unchanged) | `src/models/sgaligner/src/aligner/` |
| `dual_scene_aligner_wrapper.py` | (unchanged) | `src/models/sgaligner/src/aligner/` |
| `edge_gat.py` | (unchanged) | `src/models/sgaligner/src/aligner/networks/` |
| `build_scene_graph_unique_labels.py` | (unchanged) | `preprocessing/` |
| `build_scene_clip.py` | (unchanged) | `preprocessing/` |
| `add_relationship_clip.py` | (unchanged) | `preprocessing/` |

### Cleanup Applied to VLSG-clean

**14 hardcoded paths fixed across 5 files:**

- **`helper.py`**: Removed 2x `/home/julia/...` sys.path.insert lines; removed dead code (`calculate_overlap`, `np_cosine_sim`, `nodes_features_similar`, `load_scene_graphs`, `load_text_graphs`); removed unused imports (torch, spacy, copy, norm)
- **`scene_graph.py`**: Replaced 2x `/home/julia/...` paths in `__main__` with argparse
- **`eval_dual_scene.py`**: Replaced 3x `/content/drive/...` paths with `--graphs_3dssg` and `--graphs_scanscribe_test` argparse args; removed sys.path hacks; guarded `torch.cuda.empty_cache()`
- **`eval_dual_scene_table1.py`**: Replaced 3x `/content/drive/...` paths with argparse args (`--graphs_3dssg`, `--graphs_scanscribe_test`, `--graphs_scanscribe_pool`)
- **`scanscribetoclip_dataset.py`**: Replaced 4x `/Users/shirley/...` paths with argparse

**`scene_graph_utils.py`** was discovered as a missing dependency during audit (imported by `scene_graph.py` for `check_valid_graph`) and added from colleague's repo.

---

## 3. Consolidation Plan (8 Phases)

### Phase 1: Infrastructure (conf/ + pyproject.toml)

Copied all Hydra config groups from `whereami-text2sgm/conf/` into `Lang-Loc/conf/`:

| Config | Source | Notes |
|---|---|---|
| `conf/config.yaml` | whereami | Root defaults list — added `retrieval` and `dialogue` groups |
| `conf/paths/default.yaml` | whereami | All data/model paths |
| `conf/model/default.yaml` | whereami | BigGNN: N, heads, embed_dim |
| `conf/graph/default.yaml` | whereami | embedding_type, max_dist, use_attributes |
| `conf/train/default.yaml` | whereami | Training hyperparams |
| `conf/eval/default.yaml` | whereami | Eval settings |
| `conf/localization/default.yaml` | whereami | Grid step, eye height, FOV, hit radii |
| `conf/baseline/default.yaml` | whereami | Midpoint baseline params |
| `conf/retrieval/default.yaml` | **NEW** | DualSceneAligner: node_input_dim=518, hidden_dim=256, temperature=0.15, etc. |
| `conf/dialogue/default.yaml` | **NEW** | Translated from DialogueConfig dataclass (40+ params) |

Created `pyproject.toml` — package name "langloc", Python >=3.10, dependencies merged from all three repos.

### Phase 2: Shared Data Processing Layer

Copied from `whereami-text2sgm/whereami/data_processing/` → `Lang-Loc/src/data_processing/`:

| File | Description |
|---|---|
| `scene_graph.py` | Canonical SceneGraph class (word2vec/CLIP/ada, `to_pyg()`) |
| `scene_graph_utils.py` | `check_valid_graph()` |
| `create_text_embeddings.py` | Word2vec / CLIP / ada embedding creation |
| `graph_loader_3dssg.py` | 3DSSG graph loader |
| `graph_loader_scanscribe.py` | ScanScribe graph loader |
| `graph_loader_human.py` | Human annotation graph loader |
| `graph_loader_utils.py` | Shared loader utilities |
| `__init__.py` | Package init |

Also copied `whereami/utils/utils.py` → `src/utils/utils.py` (spaCy helpers).

### Phase 3: Localization Module (Stage 2)

Copied all 13 files from `whereami-text2sgm/whereami/localization/` → `Lang-Loc/src/localization/`:

`cli.py`, `evaluation.py`, `grid.py`, `matching.py`, `frame_io.py`, `prediction.py`, `metrics.py`, `coarse_search.py`, `visualization.py`, `pipeline.py`, `baseline_midpoint.py`, `sort_eval_table.py`, `__init__.py`

Also copied 6 visualization files from `whereami/visualization/` → `src/visualization/`:

`visualize_3rscan_segments.py`, `visualization_minimal.py`, `visualization_graph_object.py`, `visualize_loc_from_query.py`, `visualize_loc_prob.py`, `__init__.py`

### Phase 4: Text2SGM Models

Copied from `whereami-text2sgm/whereami/models/` → `Lang-Loc/src/models/`:

`model_graph2graph.py`, `train.py`, `eval.py`, `inference.py`, `single_inference.py`, `train_utils.py`, `timing.py`, `__init__.py`

Copied from `whereami-text2sgm/whereami/analysis/` → `Lang-Loc/src/analysis/`:

`helper.py` (contains `get_matching_subgraph`), `__init__.py`

**Note:** `args.py` from whereami/models/ was intentionally excluded (superseded by Hydra).

### Phase 5: Retrieval Module (Stage 1)

Copied 16 files from `VLSG-clean/` → `Lang-Loc/src/retrieval/` with full import rewriting:

```
src/retrieval/
├── models/
│   ├── dual_scene_aligner.py       ← from VLSG-clean/src/models/sgaligner/src/aligner/
│   ├── dual_scene_aligner_wrapper.py
│   ├── networks/
│   │   └── edge_gat.py             ← from .../networks/
│   └── __init__.py
├── datasets/
│   ├── dual_scene_graph_dataset.py  ← from VLSG-clean/src/datasets/
│   ├── scanscribetoclip_dataset.py
│   └── __init__.py
├── eval_dual_scene.py               ← from VLSG-clean/src/eval/
├── eval_dual_scene_table1.py
├── helper.py
├── scene_graph.py                   ← VLSG-clean's SceneGraph (separate from data_processing/)
├── scene_graph_utils.py
├── clip_utils.py
├── train_dual_scene.py              ← from VLSG-clean/src/trainval/
└── __init__.py
```

Preprocessing scripts moved to `scripts/retrieval/`:
- `build_scene_graph_unique_labels.py`
- `build_scene_clip.py`
- `add_relationship_clip.py`

### Phase 6: Reorganize Existing Lang-Loc

- Moved `src/frame_selection/` → `src/dataset_creation/frame_selection/`
- Moved `src/annotation/` → `src/dataset_creation/annotation/`

### Phase 7: Scripts & Entry Points

Shell scripts from whereami copied to `scripts/localization/`:
- `run_eval.sh`
- `baseline_midpoint.sh`
- `visualize_eval_loc.sh`

Existing scripts organized under `scripts/dataset/`.

Created `__init__.py` in all new directories.

### Phase 8: Verification

Import grep checks confirmed:
- **Zero** remaining `whereami.*` imports
- **Zero** remaining `sys.path` hacks in retrieval
- **Zero** remaining `sgaligner` path references
- **Zero** bare `from scene_graph import` statements

---

## 4. Import Rewriting Reference

| Old Import Prefix | New Import Prefix |
|---|---|
| `whereami.localization.` | `src.localization.` |
| `whereami.data_processing.` | `src.data_processing.` |
| `whereami.utils.` | `src.utils.` |
| `whereami.visualization.` | `src.visualization.` |
| `whereami.models.` | `src.models.` |
| `whereami.analysis.` | `src.analysis.` |
| `src.models.sgaligner.src.aligner.` | `src.retrieval.models.` |
| `src.models.sgaligner.src.aligner.networks.` | `src.retrieval.models.networks.` |
| `src.datasets.` (VLSG) | `src.retrieval.datasets.` |
| bare `from scene_graph import` | `from src.retrieval.scene_graph import` |
| bare `from helper import` | `from src.retrieval.helper import` |
| `from scene_graph_utils import` | `from src.retrieval.scene_graph_utils import` |

---

## 5. Final Directory Layout

```
Lang-Loc/
├── conf/                                     # 10 Hydra YAML configs
│   ├── config.yaml                           # Root defaults
│   ├── paths/default.yaml
│   ├── model/default.yaml
│   ├── graph/default.yaml
│   ├── train/default.yaml
│   ├── eval/default.yaml
│   ├── localization/default.yaml
│   ├── baseline/default.yaml
│   ├── retrieval/default.yaml                # NEW
│   └── dialogue/default.yaml                 # NEW
│
├── src/                                      # 93 Python files
│   ├── data_processing/ (8 files)            # From whereami — shared SceneGraph + embeddings
│   ├── retrieval/ (16 files)                 # From VLSG-clean — Stage 1 scene retrieval
│   ├── localization/ (13 files)              # From whereami — Stage 2 fine localization
│   ├── dialogue/ (12 files)                  # Existing — Stage 3 dialog disambiguation
│   ├── models/ (8 files)                     # From whereami — Text2SGM model
│   ├── visualization/ (6 files)              # From whereami
│   ├── dataset_creation/ (10 files)          # Existing, reorganized
│   │   ├── frame_selection/ (7 files)        #   Was src/frame_selection/
│   │   └── annotation/ (1 file)              #   Was src/annotation/
│   ├── analysis/ (2 files)                   # From whereami — helper functions
│   ├── utils/ (10 files)                     # Merged utilities
│   ├── navigation/ (1 file)                  # Existing (Streamlit app support)
│   ├── state/ (1 file)                       # Existing (Streamlit app support)
│   └── ui/ (5 files)                         # Existing (Streamlit app support)
│
├── scripts/                                  # 13 scripts total
│   ├── dataset/                              # Existing (setup_sample_data.sh, etc.)
│   ├── localization/                         # From whereami (run_eval.sh, etc.)
│   └── retrieval/                            # From VLSG-clean (build_scene_graph_unique_labels.py, etc.)
│
├── app/app.py                                # Streamlit annotation tool (existing)
├── config/                                   # Static data files (scene lists, etc.)
├── pyproject.toml                            # NEW
└── requirements.txt                          # Existing (not yet updated)
```

---

## 6. Known Issues & Pending Work

The repo is consolidated structurally but **has NOT been smoke-tested with actual Python imports**. The following issues are known or likely:

### Import & Runtime Issues

1. **No Python import smoke tests were run.** Verification was grep-based only. Some imports may fail at runtime.
2. **`src/models/train.py` and `src/models/eval.py`** use `@hydra.main(config_path="../../conf")` — relative path needs verification from their new location in Lang-Loc.
3. **`src/retrieval/eval_dual_scene.py`** has module-level code (CLIP loading, device detection) that runs at import time — may need guarding.
4. **`src/dialogue/eval_runner.py`** depends on external `pose_level_dialogue_semantic_fallback` (dsf) package — this is temporary and will be refactored.

### Orphaned / Unaddressed Modules

5. **`src/navigation/`, `src/state/`, `src/ui/`** — these Streamlit app support modules were NOT moved into `dataset_creation/`. They remain at the top level of `src/`.
6. **`app/app.py`** — the Streamlit annotation app may have broken imports after `src/frame_selection/` was moved to `src/dataset_creation/frame_selection/`.

### Config Issues

7. **`conf/paths/default.yaml`** references `${oc.env:WHEREAMI_DATA_ROOT}` — should be updated to a LangLoc-specific env var (e.g., `LANGLOC_DATA_ROOT`).
8. **`requirements.txt`** was NOT updated — still has old deps, doesn't include torch/hydra. It should be synced with `pyproject.toml`.

### Shell Scripts

9. **Scripts in `scripts/localization/`** may reference old `whereami` module paths (e.g., `python -m whereami.X`) that were only partially fixed during consolidation.

### Two SceneGraph Classes

10. There are **two separate `SceneGraph` classes** with different interfaces:
    - `src/data_processing/scene_graph.py` — used by localization + models
    - `src/retrieval/scene_graph.py` — used by retrieval eval
    - These should eventually be unified, but for now they serve different roles.

---

## 7. Suggested Next Steps

When resuming work on this repo:

1. **Run Python import smoke tests:**
   ```bash
   cd "/home/rohamzn/UZH Uni/Master Project/Lang-Loc"
   pip install -e .
   python -c "from src.data_processing.scene_graph import SceneGraph; print('OK')"
   python -c "from src.localization.evaluation import run_evaluation; print('OK')"
   python -c "from src.retrieval.models.dual_scene_aligner import DualSceneAligner; print('OK')"
   python -c "from src.dialogue.eval_runner import run_entry; print('OK')"
   python -c "from src.models.train import run_training; print('OK')"
   ```
2. Fix any broken imports discovered
3. Address orphaned modules (`src/navigation/`, `src/state/`, `src/ui/`)
4. Verify Hydra config paths work from new file locations
5. Update `conf/paths/default.yaml` env var name
6. Sync `requirements.txt` with `pyproject.toml`
7. Fix Streamlit app imports if needed
8. Update shell scripts to use `src.` module paths

---

## 8. Critical Files Reference

| File | Why It Matters |
|---|---|
| `src/localization/evaluation.py` | Heaviest cross-module imports — primary file to verify |
| `src/retrieval/eval_dual_scene.py` | Most rewritten file — had the most sys.path hacks |
| `src/data_processing/scene_graph.py` | Canonical SceneGraph, shared by localization and models |
| `src/retrieval/scene_graph.py` | Retrieval's own SceneGraph, different interface |
| `conf/config.yaml` | Root Hydra config — entry point for all configs |
| `src/dialogue/eval_runner.py` | Has temporary dsf dependency |
| `pyproject.toml` | Package definition — deps from all three repos |
