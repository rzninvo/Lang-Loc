# Configuration, project structure, and runtime notes

## Hydra config tree

All configuration is [Hydra](https://hydra.cc/) under
[`configs/`](configs/). Group-style overlays let you swap defaults
per-dataset or per-experiment.

```text
configs/
├── config.yaml                  # Root defaults list
├── paths/default.yaml           # data_root, scannet_root, rscan_root, eval_output_dir
├── dataset/default.yaml         # Keyframe selection (Next-Best-View + DPP) parameters
├── retrieval/default.yaml       # Scene-retrieval evaluation
├── localization/default.yaml    # Fine localization (paper Supp. Tab. 7)
├── localization/scannet.yaml    # ScanNet overlay (Field of View 58.30 x 45.33 deg)
├── localization/3rscan.yaml     # 3RScan overlay (Field of View 39.31 x 64.76 deg)
├── dialogue/default.yaml        # Bayesian-clarification backend
├── model/default.yaml           # BigGNN architecture
├── graph/default.yaml           # Graph construction
├── train/default.yaml           # Training hyperparameters
├── eval/default.yaml            # Evaluation settings (3D View IoU)
├── baseline/default.yaml        # Midpoint + Qwen VLM baselines
└── manifests/                   # Dataset-release manifest text files
```

Paper-table subset manifests live at the repo root under
[`manifests/`](manifests/) (see [`manifests/README.md`](manifests/README.md))
and pin which scene IDs belong to which table.

### Override any key via CLI

Hydra supports both `key=value` (override existing) and
`+key=value` (add new) syntax:

```bash
# Override an existing key
python -m langloc.localization.cli paths.data_root=/mnt/data

# Add a new key (initial value must already be null in default.yaml)
python -m langloc.localization.cli "+localization.scene_ids=[scene0000_00,scene0001_00]"

# Stack multiple overrides
python -m langloc.localization.cli \
    localization=scannet \
    localization.seed=42 \
    localization.frame_policy=all
```

The reproducer scripts under `scripts/` already pass the right
overrides for the paper tables; see
[REPRODUCE.md](REPRODUCE.md) for full invocations.

## Project structure

```text
Lang-Loc/
├── configs/             # Hydra configs (see above)
├── data/                # Datasets, gitignored. Download into here.
├── eval/                # Eval output JSONs, gitignored
├── langloc/             # Main Python package
│   ├── dataset/         #   Sec. 3.1: keyframe selection + description gen
│   ├── graphs/          #   Scene-graph types and loaders
│   ├── graph_matching/  #   BigGNN dual encoder
│   ├── retrieval/       #   Sec. 3.2: scene retrieval (Tabs. 1, 2, 3)
│   ├── localization/    #   Sec. 3.3: fine localization (Tab. 4 no-dialog)
│   ├── dialogue/        #   Sec. 3.4: with-dialog refinement (Tab. 4 dialog rows)
│   └── utils/           #   Shared (seed, geometry helpers, etc.)
├── manifests/           # Paper-table subset lists, tracked
├── scripts/             # Per-table reproduction shell scripts
└── tools/               # Standalone sub-projects (see README.md)
    ├── annotation_website/
    ├── baselines/
    ├── eval/
    └── download_scannet.py
```

## Performance notes

| Step | Throughput / cost | Notes |
|---|---|---|
| `.sens` extraction (ScanNet) | ~30 s/scan with `SENS_WORKERS=16` | I/O bound; SSD helps. |
| Keyframe selection (NBV + DPP) | ~90 s/scene on A100 | Rasterization-bound. |
| Description generation (GPT-4o-mini, all keyframes) | ~$0.01 / scene | Depends on number of keyframes. |
| Fine localization (Tab. 4 row, 100 scenes) | ~20 min on A100, ~25 min on RTX 5090 | Grid step 0.25 m. |
| Tab. 5 (1319-scene full pool) | ~2 h on RTX 5090 | Most time is per-scene mesh I/O + raycasting. |
| Description parsing (GPT-4o-mini) | ~$0.001 / frame, ~2 frames/sec | Bottleneck is OpenAI rate-limit. |
| GPT-5.5 vision baseline (1915 calls, both datasets) | ~$31, ~22 min | Concurrency 8. |
| Annotation site | <100 ms/request | Scales to hundreds of concurrent annotators on free-tier hosting. |

## Troubleshooting

For install-time errors (PyTorch3D build, spaCy model, OpenAI key),
see [INSTALL.md](INSTALL.md#troubleshooting).

**`FileNotFoundError` on 3RScan poses.**
Each scene's `sequence.zip` must be extracted before the localizer
reads its frames. The single-scene setup script extracts the
sequence as a side effect:

```bash
bash scripts/dataset/setup_sample_data.sh --dataset 3RScan <scene_uuid>
```

**`Hydra config error: scene_ids`.**
The `+localization.scene_ids=[...]` syntax requires the key to be
declared as `null` in `configs/localization/default.yaml`. If you
get an "override key not found" error, double-check the `+` prefix
and that the key path matches your config tree.

**Mesh index mismatch warning.**
Invalid faces are filtered in the loader; non-fatal.

**`No frames after IQA filtering`.**
Lower `dataset.scannet.iqa_threshold` or
`dataset.3rscan.iqa_threshold` (both default `0.5` in
[`configs/dataset/default.yaml`](configs/dataset/default.yaml))
via Hydra override. The pipeline falls back to the top-50 frames
by quality score if fewer than 50 survive.

**Annotation site cannot load mesh.**
Check that the full-res or decimated `.ply` exists at
`data/scans/<scene>/<scene>_vh_clean*.ply`. See
[`tools/annotation_website/README.md`](tools/annotation_website/README.md)
for the lookup order.

**Hydra-generated `outputs/` directory growing.**
`outputs/` and `eval/` are both gitignored. Clean them whenever you
want; the reproducer scripts re-create them on the next run.

**Reproducer prints `[WARN] Qwen baseline metrics missing`.**
[`tools/eval/recall_cdf_with_dialog.py`](tools/eval/recall_cdf_with_dialog.py)
skips the Qwen series in CDF plots if its cache is absent. To
populate it, run
[`bash scripts/localization/baseline_eval_qwen.sh`](scripts/localization/baseline_eval_qwen.sh)
first.
