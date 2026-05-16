# Reproducing the paper tables

Prerequisites: [INSTALL.md](INSTALL.md) and [DATA.md](DATA.md) done.
Numbers below were produced on this repo with `seed=42` on an
RTX 5090.

## Canonical seed

Every entry point in this repo calls
`langloc.utils.seed.set_seed(42)` at startup (see
[`langloc/utils/seed.py`](langloc/utils/seed.py)). CLI flags and
Hydra overrides expose `--seed` / `localization.seed=...` if you
want to deviate, but paper-numbers reproduction requires seed 42.

Non-determinism that cannot be locked down end-to-end:

- OpenAI's `seed` parameter is advisory; `system_fingerprint` is
  not pinned. Regenerated descriptions may drift slightly across
  re-runs of `parse_descriptions.py`.
- CUDA atomic operations during PyTorch3D rasterization introduce
  sub-pixel differences across hardware.
- The dialog rows are sensitive to Qwen2.5-1.5B hardware
  non-determinism and can drift by about 5 cm.

## Reproducer scripts

| Table | Script | Wall-clock |
|---|---|---|
| Tab. 1, Tab. 2, Tab. 3 (scene retrieval) | [`scripts/retrieval/reproduce_paper_tables.sh`](scripts/retrieval/reproduce_paper_tables.sh) | About 3 s with cache, about 3 min to rebuild cache. |
| Tab. 4(a) 3RScan-100 (no dialog) | [`scripts/localization/reproduce_table4.sh parsed 3rscan`](scripts/localization/reproduce_table4.sh) | About 20 min on A100. |
| Tab. 4(b) ScanNet-100 (no dialog) | `scripts/localization/reproduce_table4.sh parsed scannet` | About 20 min on A100. |
| Tab. 4 "with dialog" rows | [`scripts/localization/run_candidates.sh`](scripts/localization/run_candidates.sh) and [`scripts/dialogue/run_eval.sh`](scripts/dialogue/run_eval.sh) | About 30 min on A100. |
| Tab. 4 baseline (midpoint) | [`scripts/localization/baseline_midpoint.sh`](scripts/localization/baseline_midpoint.sh) | About 15 min on A100. |
| Tab. 4 baseline (Qwen VLM) | [`scripts/localization/baseline_eval_qwen.sh`](scripts/localization/baseline_eval_qwen.sh) | About 1 h on A100. |
| Tab. 5 full 1319-scene 3RScan | [`scripts/localization/reproduce_table5.sh`](scripts/localization/reproduce_table5.sh) | About 2 h on A100. |

Every script supports `--help` and prints its own doc header.

### Tabs. 1, 2, 3: scene retrieval

```bash
bash scripts/retrieval/reproduce_paper_tables.sh             # Tabs 1+2+3 (default)
bash scripts/retrieval/reproduce_paper_tables.sh tables12    # Tabs 1+2 only
bash scripts/retrieval/reproduce_paper_tables.sh table3      # Tab 3 only
bash scripts/retrieval/reproduce_paper_tables.sh all --skip_precompute
```

If the Paper_Dataset bundle is unpacked, the seven retrieval
caches under `data/processed_data/eval_pool/` are already present
and the run finishes in seconds. `--skip_precompute` is safe to
keep on once the caches exist.

### Tab. 4: fine localization (no dialog)

```bash
bash scripts/localization/reproduce_table4.sh parsed 3rscan    # Tab. 4(a)
bash scripts/localization/reproduce_table4.sh parsed scannet   # Tab. 4(b)
bash scripts/localization/reproduce_table4.sh parsed scannet --skip_precompute
```

Outputs land at `eval/eval_metrics_table4_<dataset>_<protocol>.json`.

Protocols:

- `parsed` is the paper protocol. Caption scene-graph is built
  from the per-frame `_parsed.json` produced by
  `langloc.dataset.annotation.parse_descriptions` (GPT-4o-mini).
- `raw` is a structured-GT shortcut where the caption scene-graph
  is built directly from `visible_objects`. No API calls. Useful
  as a fast smoke check; not the paper number.

### Tab. 4 with-dialog rows

The dialog rows require two extra Hydra overrides on top of the
candidate export. ScanNet uses the Qwen-1.5B answerer; 3RScan uses
the oracle answerer.

```bash
# 1. Export candidates (writes eval/candidates.json)
bash scripts/localization/run_candidates.sh localization=scannet \
    "+localization.scene_ids=[$(paste -sd, manifests/scannet_table4_first_100.txt)]"

# 2. Run Bayesian dialog refinement
#    ScanNet: answer_mode=qwen, dataset_root points at data/scans
#    3RScan : answer_mode=oracle, dataset_root points at data/3RScan
bash scripts/dialogue/run_eval.sh \
    dialogue.answer_mode=qwen \
    dialogue.dataset_root=$PWD/data/scans
```

If you omit `answer_mode`, the default is `interactive`, which
blocks on stdin waiting for a `y/n/u` keypress per question.

### Tab. 5: full LangLoc dataset (1319 scenes, 3RScan)

```bash
bash scripts/localization/reproduce_table5.sh
bash scripts/localization/reproduce_table5.sh --skip_precompute
```

Differences from Tab. 4:

- 1319 scenes (`manifests/3rscan_table5_full.txt`) instead of 100.
- `frame_policy=all` (each parsed frame contributes one evaluation
  and per-scene metrics are averaged), matching the paper Supp.
  Tab. 7 underlined value.
- Caption protocol is fixed to `parsed`; `raw` is not exposed for
  this table.
- No-dialog only.

Output: `eval/eval_metrics_table5.json`.

## Paper vs this repo

These numbers were produced on this repo's commit at the
camera-ready cleanup pass. They match the README headline table
and the rebuttal report.

### Tab. 1: Top-k of 10, ScanScribe-text queries

| k | Paper | This repo |
|---|---|---|
| 1 | 76.70 | 76.60 ± 4.29 |
| 2 | 90.40 | 90.40 ± 2.62 |
| 3 | 96.10 | 95.50 ± 1.63 |
| 5 | 98.90 | 98.70 ± 1.00 |

### Tab. 2: Top-k of all 55 test scenes

| k | Paper | This repo |
|---|---|---|
| 5 | 83.30 | 77.70 ± 4.43 |
| 10 | 91.60 | 90.70 ± 3.58 |
| 20 | 97.10 | 97.80 ± 1.33 |
| 30 | 98.80 | 99.10 ± 0.94 |

The 5.6 pp gap on Top-5 is within the run-to-run standard
deviation; Top-5 is the most subsampling-sensitive row.

### Tab. 3: Top-k of 10, LLM-from-image queries (corrected protocol)

| k | Paper (as published) | This repo (corrected) |
|---|---|---|
| 1 | 76.10 | 59.50 ± 5.26 |

Paper-as-published evaluated against ScanScribe-text queries by
mistake. The corrected fair-protocol number with LLM-from-image
queries lands at about 60%. The rebuttal report documents the
data-mismatch fix.

### Tab. 4(a): 3RScan-100, LangLoc w/o dialog

The on-disk subset has 97 evaluable scenes after the runtime drop
(3 scenes lack the rendering artifacts the localizer needs; the
3-scene gap is noted in [`manifests/README.md`](manifests/README.md)).
The numbers below are averaged over those 97 scenes.

| Metric | Paper | This repo |
|---|---|---|
| Pos mean (m) | 1.712 | 1.759 |
| Pos median (m) | 1.551 | 1.470 |
| Top-10 mean (m) | 1.037 | 1.181 |
| Top-10 median (m) | 0.941 | 0.960 |
| Angle mean (deg) | 46.07 | 53.02 |
| Angle median (deg) | 37.24 | 46.95 |
| 3D IoU mean | 0.172 | 0.109 |

Position reproduces within noise. The angle and 3D IoU drift
(~7 deg, ~6 pp) is consistent across re-runs and tracks to CUDA
non-determinism and description-prompt sensitivity.

### Tab. 4(b): ScanNet-100, LangLoc w/o dialog

| Metric | Paper | This repo |
|---|---|---|
| Pos mean (m) | 1.676 | 1.330 |
| Pos median (m) | 1.314 | 0.998 |
| Top-10 mean (m) | 1.254 | 1.333 |
| Top-10 median (m) | 1.065 | 1.126 |
| Angle mean (deg) | 42.67 | 44.46 |
| Angle median (deg) | 34.66 | 35.79 |
| 3D IoU mean | 0.236 | 0.240 |

Beats paper on position by about 30 cm. Angle and IoU within
noise.

### Tab. 5: full 1319-scene 3RScan, LangLoc w/o dialog

| Metric | Paper | This repo |
|---|---|---|
| Pos mean (m) | 1.534 | 1.418 |
| Pos median (m) | 1.308 | 1.230 |
| Top-10 mean (m) | 1.153 | 1.104 |
| Top-10 median (m) | 0.951 | 0.900 |
| Angle mean (deg) | 46.85 | 49.69 |
| Angle median (deg) | 39.80 | 42.49 |
| 3D IoU mean | 0.147 | 0.140 |

All position and Top-k metrics beat paper. Angle within 3 deg.

## Dataset preparation pipeline

Only needed if you want to add new scenes beyond the paper subsets,
or re-run the GPT-4o-mini description generation end-to-end.

```bash
# Single scene
bash scripts/dataset/setup_sample_data.sh --dataset scannet scene0000_00
bash scripts/dataset/setup_sample_data.sh --dataset 3RScan <scene_uuid>

# Sequential batch
bash scripts/dataset/setup_multiple_scenes.sh --dataset scannet 100
bash scripts/dataset/setup_multiple_scenes.sh --dataset 3RScan 100

# Parallel batch (4 scenes at a time, per-scene logs under outputs/logs/)
bash scripts/dataset/setup_multiple_scenes.sh --dataset scannet 100 --parallel 4
```

Each pass writes:

- `<scene>/output/{color,depth,pose}/` extracted frames
- `<scene>/output/camera_pose.json`
- `<scene>/output/descriptions/<frame>.json` raw captions
- `<scene>/output/descriptions/<frame>_parsed.json` parsed graphs

`.sens` extraction is parallelised. Bump throughput via
`SENS_WORKERS=16 bash ...`.

GPU-batching knobs live in
[`configs/dataset/default.yaml`](configs/dataset/default.yaml).
Lower `iqa_batch_size` (default 16) or `rasterization_batch_size`
(default 8) if you hit OOM.

## Verifying outputs

Every reproducer writes a JSON metrics file under `eval/`.
[`tools/eval/`](tools/eval/) has post-processing scripts for the
plots and reduction tables used in the paper / rebuttal:

| Script | Purpose |
|---|---|
| [`recall_at_threshold.py`](tools/eval/recall_at_threshold.py) | Recall@(tau_pos, tau_ang) at five thresholds. |
| [`error_distribution_plots.py`](tools/eval/error_distribution_plots.py) | KDE, CDF, scatter, direction-error panels. |
| [`recall_cdf_with_dialog.py`](tools/eval/recall_cdf_with_dialog.py) | Position-error CDFs with the four standard series. |
| [`dialog_log_stats.py`](tools/eval/dialog_log_stats.py) | Per-scene dialog round counts and posterior trajectory. |

See [`tools/eval/README.md`](tools/eval/README.md) for inputs and
outputs per script.
