<div align="center">
<table>
  <tr>
    <td align="center" valign="middle">
      <img src="media/figures/logos/cvg_logo_colour-white.png" height="40"/>
    </td>
    <td align="center" valign="middle">
      <img src="media/figures/logos/eth_logo_kurz_neg.png" height="80"/>
    </td>
    <td align="center" valign="middle">
      <img src="media/figures/logos/uzh-logo-white.png" height="70"/>
    </td>
  </tr>
</table>
</div>

# LangLoc — Language-Based 3D Indoor Localization

End-to-end pipeline for language-based localization in 3D indoor scenes using ScanNet and 3RScan datasets.

**Pipeline stages:**

1. **Dataset Creation** (Sec 3.1) — Download scenes, select diverse keyframes (NBV + DPP), generate text descriptions
2. **Scene Retrieval** (Sec 3.2) — Graph-based scene retrieval with BigGNN
3. **Fine Localization** (Sec 3.3) — Camera pose estimation from language queries
4. **Dialogue System** (Sec 3.4) — Interactive clarification dialogue

---

## Requirements

* Python >= 3.10
* CUDA-capable GPU (for PyTorch3D rasterization and CLIP)
* Linux recommended
* Dataset credentials: [ScanNet](http://www.scan-net.org/) and/or [3RScan](http://campar.in.tum.de/public_datasets/3RScan/)

### Installation

```bash
# 1. Create conda environment
conda create -n langloc python=3.10 -y
conda activate langloc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch with CUDA support (adjust for your CUDA version)
# For CUDA 12.6:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 4. Install PyTorch3D from source
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . --no-build-isolation && cd ..
```

---

## Configuration

All configuration is managed via [Hydra](https://hydra.cc/) under `configs/`:

```text
configs/
├── config.yaml              # Root config (defaults list)
├── paths/default.yaml        # All file paths (data_root, scannet_root, rscan_root, ...)
├── dataset/default.yaml      # Frame selection & NBV/DPP parameters
├── retrieval/default.yaml    # Scene retrieval settings
├── localization/default.yaml # Fine localization settings
├── dialogue/default.yaml     # Dialogue system settings
├── model/default.yaml        # BigGNN model architecture
├── graph/default.yaml        # Graph construction
├── train/default.yaml        # Training hyperparameters
├── eval/default.yaml         # Evaluation settings
└── manifests/                # Static scene lists
    ├── 3RScan_release_scans.txt
    ├── 3RScan_partial.txt
    ├── scannetv2_all.txt
    └── scanscribe_cleaned.json
```

Key paths to configure in `configs/paths/default.yaml`:

| Key | Default | Description |
| --- | ------- | ----------- |
| `data_root` | `./data` | Root directory for all dataset files |
| `scannet_root` | `${paths.data_root}/scans` | ScanNet scenes directory |
| `rscan_root` | `${paths.data_root}/3RScan` | 3RScan scenes directory |

Override any config value via CLI: `paths.data_root=/mnt/data`

---

## Dataset Creation

### Single Scene

Downloads, extracts, runs keyframe selection, and generates descriptions for one scene.

**ScanNet:**

```bash
bash scripts/dataset/setup_sample_data.sh --dataset scannet scene0000_00
```

**3RScan:**

```bash
bash scripts/dataset/setup_sample_data.sh --dataset 3RScan <scene-uuid>
```

### Multiple Scenes

Processes multiple scenes in batch. Skips already-processed scenes.

**Sequential (default):**

```bash
# Process 20 ScanNet scenes one at a time
bash scripts/dataset/setup_multiple_scenes.sh --dataset scannet 20

# Process 10 3RScan scenes one at a time
bash scripts/dataset/setup_multiple_scenes.sh --dataset 3RScan 10
```

**Parallel (recommended for multi-core servers):**

Use `--parallel N` to process N scenes concurrently. Per-scene logs are saved to `outputs/logs/`.

```bash
# Process 20 ScanNet scenes, 4 at a time
bash scripts/dataset/setup_multiple_scenes.sh --dataset scannet 20 --parallel 4

# Process all 3RScan scenes from ScanScribe manifest, 8 at a time
bash scripts/dataset/setup_multiple_scenes.sh --dataset 3RScan --source scanscribe --parallel 8
```

**Other examples:**

```bash
# Process all scenes (default source), sequential
bash scripts/dataset/setup_multiple_scenes.sh --dataset 3RScan
```

### Download Only (no keyframe selection)

The `.sens` extraction step supports parallel frame decompression via the `SENS_WORKERS` environment variable (default: auto-detected, up to 16 cores).

```bash
bash scripts/dataset/download_subset.sh --dataset scannet scene0000_00

# Speed up .sens extraction with more workers
SENS_WORKERS=16 bash scripts/dataset/download_subset.sh --dataset scannet scene0000_00

bash scripts/dataset/download_subset.sh --dataset 3RScan <scene-uuid>
```

### Python Entry Points (Hydra)

You can also run each pipeline stage directly with Hydra overrides:

```bash
# Keyframe selection — ScanNet
python -m langloc.dataset.frame_selection.scannetpp_best_views scan_id=scene0000_00

# Keyframe selection — 3RScan
python -m langloc.dataset.frame_selection.3rscan_best_views scan_id=<scene-uuid>

# Description generation
python -m langloc.dataset.annotation.generate_descriptions scan_id=<scene-id> dataset.target=3RScan
```

Common Hydra overrides:

| Override | Description |
| -------- | ----------- |
| `scan_id=<id>` | Scene ID to process |
| `dataset.debug=true` | Enable debug output |
| `dataset.auto_clean=true` | Delete raw files after processing |
| `dataset.save_semantic_masks=true` | Export 16-bit semantic masks |
| `dataset.save_instance_masks=true` | Export 16-bit instance masks |
| `dataset.target=3RScan` | Target dataset (for description generation) |

### Performance Tuning

GPU batching for IQA scoring and rasterization is configured in `configs/dataset/default.yaml` under each dataset section:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `iqa_batch_size` | `16` | Batch size for IQA quality scoring (GPU) |
| `rasterization_batch_size` | `8` | Batch size for PyTorch3D visibility rasterization (GPU) |

Increase batch sizes for faster processing if you have enough GPU memory. Set to `1` to disable batching (sequential mode).

---

## Per-Scene Outputs

After processing, each scene directory contains:

```text
<scene_id>/output/
├── color/            — Selected RGB keyframes
├── depth/            — Depth maps
├── pose/             — Camera-to-world 4x4 matrices
├── camera_pose.json  — Consolidated pose file
├── instance/         — Instance masks (if enabled)
└── semantic/         — Semantic masks (if enabled)
```

Cached intermediate results (visibility maps, NBV order) are stored in `cache/` (ScanNet) or `cache_rscan/` (3RScan).

---

## Evaluation

All evaluation results are saved to `eval/`. Create it if it doesn't exist: `mkdir -p eval`.

### 1. Scene Retrieval (Paper Tables 1-3)

Evaluates Recall@k for text-to-scene matching. Requires a trained BigGNN checkpoint in `data/model_checkpoints/graph2graph/` and processed graph `.pt` files in `data/processed_data/`.

```bash
# Table 1: 10-scene pool (Recall@1/2/3/5)
bash scripts/retrieval/run_eval.sh

# Or run directly with custom parameters:
python -m langloc.graph_matching.eval eval.model_name=<checkpoint_name> eval.eval_iters=10
```

For the DualSceneAligner retrieval evaluation (Tables 1-3):

```bash
# Table 1: 10-scene pool
python -m langloc.retrieval.eval retrieval.eval.protocol=table1 retrieval.checkpoint=<path>

# Table 2: full 55-scene pool
python -m langloc.retrieval.eval retrieval.eval.protocol=table2 retrieval.checkpoint=<path>

# Table 3: LLM-generated queries (requires precomputed cache)
python -m langloc.retrieval.eval retrieval.eval.protocol=table3 retrieval.cache_dir=<path>
```

### 2. Fine Localization (Paper Table 4)

Evaluates position error, angular error, Hit@r, mass-radius percentiles, and 3D View IoU. Requires scene meshes and frame description JSONs.

```bash
# Standard evaluation (all metrics)
bash scripts/localization/run_eval.sh

# ScanNet instead of 3RScan:
bash scripts/localization/run_eval.sh localization.dataset=scannet paths.rscan_root=./data/scans

# With visualization:
bash scripts/localization/visualize_eval_loc.sh
```

Key parameters (override via CLI):

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `localization.dataset` | `3rscan` | Dataset type (`3rscan` or `scannet`) |
| `localization.grid_step` | `0.25` | Grid spacing in metres |
| `localization.top_k` | `25` | Number of matched objects |
| `localization.matching_strategy` | `global_topk` | `global_topk`, `per_node`, or `relation_aware` |
| `localization.prediction_strategy` | `weighted` | `argmax`, `random`, or `weighted` (mean-shift) |

### 3. Localization Baselines (Paper Table 4)

**Midpoint baseline** (floor centroid, random heading):

```bash
bash scripts/localization/baseline_midpoint.sh
```

**VLM baseline** (Qwen2.5-VL on top-down renders):

```bash
# Step 1: Render top-down images (only needed once)
python -m langloc.baselines.topdown_3rscan --root ./data/3RScan --all-scans

# Step 2: Run VLM evaluation
bash scripts/localization/baseline_eval_qwen.sh

# For ScanNet:
DATASET=scannet SCENE_ROOT=./data/scans bash scripts/localization/baseline_eval_qwen.sh
```

### 4. Dialogue Disambiguation (Paper Table 4, "w/ dialog" rows)

Runs iterative Bayesian pose refinement via yes/no questions. Requires candidate poses from localization.

```bash
# Step 1: Export candidate poses
bash scripts/localization/run_candidates.sh

# Step 2: Run dialogue evaluation
bash scripts/dialogue/run_eval.sh
```

### Outputs

All evaluation scripts write results to `eval/`:

```text
eval/
├── eval_metrics.json                    — Fine localization metrics (per-scene)
├── eval_loc_summary.log                 — Fine localization summary table
├── candidates.json                      — Candidate poses for dialogue
├── baseline_eval_metrics_mid_point.json — Midpoint baseline metrics
├── baseline_eval_metrics_qwen_*.json    — VLM baseline metrics
└── baseline_eval_metrics_qwen_*.log     — VLM baseline summary
```

---

## Project Structure

```text
Lang-Loc/
├── configs/                    # Hydra configuration
├── data/                       # Dataset files (downloaded here)
├── langloc/                    # Main Python package
│   ├── dataset/                #   Dataset creation (Sec 3.1)
│   │   ├── frame_selection/    #     NBV + DPP keyframe selection
│   │   └── annotation/        #     GPT-based description generation
│   ├── graphs/                 #   Scene graph classes & loaders
│   ├── graph_matching/         #   BigGNN text-to-scene-graph matching
│   ├── retrieval/              #   Scene retrieval (Sec 3.2)
│   ├── localization/           #   Fine localization (Sec 3.3)
│   ├── dialogue/               #   Dialogue system (Sec 3.4)
│   └── utils/                  #   Shared utilities
├── scripts/                    # Shell scripts for batch processing
├── tools/                      # Standalone tools (downloaders, visualization)
└── media/                      # Figures and logos
```

---

## Troubleshooting

* **FileNotFound (3RScan poses)** — ensure `sequence.zip` was extracted
* **Mesh index mismatch** — invalid faces are filtered automatically
* **No frames after IQA filtering** — lower `dataset.scannetpp.iqa_threshold` or `dataset.3rscan.iqa_threshold`
* **PyTorch3D errors** — ensure matching `torch` + `pytorch3d` versions
