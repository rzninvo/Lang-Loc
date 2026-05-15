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

End-to-end pipeline for language-based localization in 3D indoor
scenes, accompanying our ECCV 2026 paper. Evaluated on
[ScanNet](http://www.scan-net.org/) and
[3RScan](http://campar.in.tum.de/public_datasets/3RScan/).

**Pipeline stages:**

1. **Dataset Creation** (Sec 3.1) — Download scenes, select diverse keyframes (NBV + DPP), generate text descriptions.
2. **Scene Retrieval** (Sec 3.2) — Graph-based scene retrieval with the BigGNN dual encoder.
3. **Fine Localization** (Sec 3.3) — Camera pose estimation from a natural-language query against a 3-D scene graph.
4. **Dialogue System** (Sec 3.4) — Optional Bayesian-clarification refinement.

---

## Table of contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset download](#dataset-download)
4. [Environment variables](#environment-variables)
5. [Reproducibility & seed](#reproducibility--seed)
6. [Reproducing the paper tables](#reproducing-the-paper-tables)
7. [Dataset preparation](#dataset-preparation-keyframes--descriptions)
8. [Configuration](#configuration)
9. [Project structure](#project-structure)
10. [Rebuttal extensions](#rebuttal-extensions)
11. [Performance notes](#performance-notes)
12. [Troubleshooting](#troubleshooting)

---

## Requirements

- **Python ≥ 3.10**
- **CUDA-capable GPU** (PyTorch3D rasterization + CLIP). 8 GB+ VRAM is comfortable; we used an A100 (40 GB) and an RTX 5090.
- **Disk:** ~60 GB for the ScanNet + 3RScan subsets used in the paper. Full ScanNet release is ~1.2 TB.
- **Linux** (tested on Ubuntu 22.04/24.04). macOS works for the analysis tools but not the rasterizer.
- An **OpenAI API key** for description generation and parsing (see [Environment variables](#environment-variables)).

## Installation

```bash
# 1. Conda environment
conda create -n langloc python=3.10 -y
conda activate langloc

# 2. Repo
git clone https://github.com/<your-org>/langloc.git
cd langloc
pip install -r requirements.txt
pip install -e .

# 3. PyTorch with CUDA (adjust for your driver)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 4. PyTorch3D from source (CUDA bindings are version-sensitive)
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . --no-build-isolation && cd ..
python -c "from pytorch3d.structures import Meshes; print('pytorch3d ok')"

# 5. spaCy word2vec embeddings (used by the parsed-caption grounder)
python -m spacy download en_core_web_lg
```

The annotation website (`tools/annotation_website/`) has its own
isolated dependencies — install them only if you plan to run the
crowdsourcing UI:

```bash
pip install -r tools/annotation_website/requirements.txt
```

## Dataset download

Both datasets require accepting their respective terms of use; we
cannot bundle them in this repo.

### ScanNet

1. Sign the [ScanNet Terms of Use](http://www.scan-net.org/) — fill out
   the Google form linked from the homepage. You'll be emailed a
   personalised `download-scannet.py` script.
2. We ship a slightly adapted copy of the same script at
   [`tools/download_scannet.py`](tools/download_scannet.py) (it
   doesn't include their access token; you must add yours).
3. For paper reproduction you only need the 100-scene subsets:

   ```bash
   # ~16 GB on disk; the full release is 1.2 TB
   python tools/download_scannet.py --out_dir data \
       --type _vh_clean_2.ply              # decimated meshes
   python tools/download_scannet.py --out_dir data \
       --type _vh_clean.ply                # full-res meshes (optional;
                                           # used by the human-localizer
                                           # for nicer-looking renders)
   python tools/download_scannet.py --out_dir data --type .sens
   # repeat for: .txt, .aggregation.json, _vh_clean_2.0.010000.segs.json
   ```

   The downloader writes to `data/scans/<scene_id>/...` — which is
   the default `paths.scannet_root` (see [`configs/paths/default.yaml`](configs/paths/default.yaml)).

### 3RScan

1. Register at the [3RScan project page](http://campar.in.tum.de/public_datasets/3RScan/) and accept their licence.
2. Place the scans at `data/3RScan/<scene_uuid>/...`. The
   pipeline expects:
   - `labels.instances.annotated.v2.ply` (mesh + per-vertex semantic labels)
   - `sequence.zip` (extracted on first use to get RGB frames + poses)
   - `semseg.v2.json` (instance segmentation)

### ScanScribe + 3DSSG

For Tabs. 1–2 (scene retrieval) you also need the
[3DSSG](https://3dssg.github.io/) scene graphs and the
[ScanScribe](https://kywind.github.io/scanscribe) caption dataset.
We ship pre-processed `.pt` files at the paths listed in
`configs/paths/default.yaml` (`graphs_3dssg`,
`scanscribe_{train,test}`, …). If you don't have them, see
[`docs/`](docs/) (gitignored working notes) for the preprocessing
recipes.

## Environment variables

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY at minimum.
```

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | GPT-4o-mini description parsing + GPT-5.5 vision baselines. Required by `langloc.dataset.annotation.parse_descriptions`, `tools/baselines/gpt_vlm/`. |
| `LANGLOC_GPT_MAX_USD` | Optional cost cap for vision-baseline runs. |
| `LANGLOC_COOKIE_SECRET`, `LANGLOC_ADMIN_TOKEN` | Annotation website only — see `tools/annotation_website/.env.example`. |

## Reproducibility & seed

**Canonical seed: 42.** Every script with stochastic computation
calls `langloc.utils.seed.set_seed(42)` at startup. CLI flags and
Hydra overrides expose `--seed` / `localization.seed=` if you need
to deviate, but **paper-numbers reproduction requires seed 42**.
DataLoader workers also use `langloc.utils.seed.worker_init_fn` so
shuffles are reproducible.

Non-determinism that we can't lock down:
- OpenAI `seed` parameter is advisory; `system_fingerprint` is not
  pinned. Description regeneration may drift slightly across
  re-runs of `parse_descriptions.py`.
- CUDA atomic ops on PyTorch3D rasterization; differences are
  sub-pixel.

## Reproducing the paper tables

Once datasets are in `data/` and `.env` is configured:

| Table | Script | Notes |
|---|---|---|
| **Tab. 1, Tab. 2** (scene retrieval) | [`scripts/retrieval/reproduce_paper_tables.sh`](scripts/retrieval/reproduce_paper_tables.sh) | ~3 s with cache; ~3 min to rebuild cache. Recall@k vs Text2SGM / Text2Pos / CLIP2CLIP. |
| **Tab. 3** (LLM-from-image queries) | same script, `--mode table3` arm | Requires precomputed query cache; same script builds it on demand. |
| **Tab. 4(a)** 3RScan-100 | [`scripts/localization/reproduce_table4.sh parsed 3rscan`](scripts/localization/reproduce_table4.sh) | Uses [`manifests/3rscan_table4_subset_100.txt`](manifests/). ~20 min on A100. |
| **Tab. 4(b)** ScanNet-100 | `scripts/localization/reproduce_table4.sh parsed scannet` | Uses [`manifests/scannet_table4_first_100.txt`](manifests/). ~20 min on A100. |
| **Tab. 4 "w/ dialog" rows** | [`scripts/localization/run_candidates.sh`](scripts/localization/run_candidates.sh) → [`scripts/dialogue/run_eval.sh`](scripts/dialogue/run_eval.sh) | Bayesian A3 backend + Qwen2.5-1.5B answerer (ScanNet) / oracle (3RScan). |
| **Tab. 4 baselines** | [`scripts/localization/baseline_midpoint.sh`](scripts/localization/baseline_midpoint.sh), [`scripts/localization/baseline_eval_qwen.sh`](scripts/localization/baseline_eval_qwen.sh) | Midpoint + Qwen2.5-VL top-down. |
| **Tab. 5** (full 1,319-scan) | [`scripts/localization/reproduce_table5.sh`](scripts/localization/reproduce_table5.sh) | Uses [`manifests/3rscan_table5_full.txt`](manifests/). No-dialog only. |

Outputs land in `eval/eval_metrics_*.json` and `eval/eval_loc_summary.log`
(both `eval/` and `outputs/` are gitignored).

### Sanity check: numbers we got

| Table | Pos. mean / median (paper) | Pos. mean / median (this repo, seed=42) |
|---|---|---|
| Tab. 4(b) ScanNet, "LangLoc w/o dialog" | 1.676 / 1.314 m | 1.330 / 0.998 m |
| Tab. 4(a) 3RScan, "LangLoc w/o dialog" | 1.712 / 1.551 m | 1.759 / 1.470 m |
| Tab. 5 (full 3RScan release, no dialog) | 1.534 / 1.308 m | 1.418 / 1.230 m |

Small paper-vs-port gaps come from CUDA non-determinism and
prompt-sensitivity in the description-generation step. The dialog
rows are sensitive to Qwen-1.5B's hardware-non-determinism and can
drift by ~5 cm.

## Dataset preparation (keyframes + descriptions)

If you want to process additional scenes beyond the paper subsets:

```bash
# One scene
bash scripts/dataset/setup_sample_data.sh --dataset scannet scene0000_00
bash scripts/dataset/setup_sample_data.sh --dataset 3RScan <uuid>

# Batch (sequential)
bash scripts/dataset/setup_multiple_scenes.sh --dataset scannet 100
bash scripts/dataset/setup_multiple_scenes.sh --dataset 3RScan 100

# Batch (parallel — 4 scenes at a time, per-scene logs in outputs/logs/)
bash scripts/dataset/setup_multiple_scenes.sh --dataset scannet 100 --parallel 4
```

Each pipeline pass produces `<scene>/output/{color,depth,pose}/`,
`<scene>/output/camera_pose.json`, and per-frame description JSONs
under `<scene>/output/descriptions/`.

`.sens` extraction is parallelised — bump throughput via
`SENS_WORKERS=16 bash …`.

GPU-batching knobs (configs/dataset/default.yaml): `iqa_batch_size`,
`rasterization_batch_size`. Both default to 16/8; lower them if you
hit OOM.

## Configuration

All config is [Hydra](https://hydra.cc/) under [`configs/`](configs/):

```text
configs/
├── config.yaml              # Root defaults list
├── paths/default.yaml        # data_root, scannet_root, rscan_root, …
├── dataset/default.yaml      # Frame-selection (NBV/DPP) parameters
├── retrieval/default.yaml    # Scene-retrieval evaluation
├── localization/default.yaml # Fine-localization (paper supp Tab. 7)
├── localization/scannet.yaml # ScanNet overlay (FoV 58.30°×45.33°)
├── localization/3rscan.yaml  # 3RScan overlay (FoV 39.31°×64.76°)
├── dialogue/default.yaml     # Bayesian-clarification backend
├── model/default.yaml        # BigGNN architecture
├── graph/default.yaml        # Graph construction
├── train/default.yaml        # Training hyperparameters
├── eval/default.yaml         # Evaluation settings
└── manifests/                # Dataset-release manifests (text files)
```

Subset manifests for paper-table reproduction live at the repo
root in [`manifests/`](manifests/README.md) — see that file for
which scene list maps to which table.

Override any key via CLI: `paths.data_root=/mnt/data localization.seed=0`.

## Project structure

```text
Lang-Loc/
├── configs/             # Hydra configs
├── data/                # Datasets (gitignored — download into here)
├── eval/                # Eval output JSONs (gitignored)
├── langloc/             # Main package
│   ├── dataset/         #   Sec 3.1: keyframe selection + description gen
│   ├── graphs/          #   Scene-graph types + loaders
│   ├── graph_matching/  #   BigGNN dual encoder
│   ├── retrieval/       #   Sec 3.2: scene retrieval (Tabs 1-3)
│   ├── localization/    #   Sec 3.3: fine localization (Tab 4 no-dialog)
│   ├── dialogue/        #   Sec 3.4: with-dialog refinement
│   └── utils/           #   Shared (seed, geometry helpers, ...)
├── manifests/           # Paper-table subset lists (tracked)
├── scripts/             # Per-table reproduction shell scripts
└── tools/               # Standalone sub-projects
    ├── annotation_website/  # Crowdsourced human description + localization site
    ├── baselines/           # Alternative describers (GPT-5.5 vision, humans)
    ├── eval/                # Plot generators, recall-sweep utilities
    └── download_scannet.py  # ScanNet bulk downloader (BYO ToS key)
```

## Rebuttal extensions

These were added for the ECCV 2026 rebuttal / camera-ready and are
self-contained sub-projects with their own READMEs:

| Sub-project | What it does | README |
|---|---|---|
| [`tools/annotation_website/`](tools/annotation_website/README.md) | FastAPI site for crowdsourcing human descriptions + first-person localizations. Cloudflare-tunnelled by default. | full deployment + admin guide |
| [`tools/baselines/gpt_vlm/`](tools/baselines/gpt_vlm/README.md) | OpenAI **GPT-5.5 vision** describer baseline (one image → 2–4-sentence first-person caption). Reviewer-WxoL closed-loop-bias defense. | usage + cost-cap notes |
| [`tools/baselines/human/`](tools/baselines/human/README.md) | Extracts human descriptions from the annotation site's SQLite DB into pipeline-ready JSONs. Reviewer-WxoL human-description ask. | flow + caveats |
| [`tools/eval/`](tools/eval/) | Recall-at-threshold sweep, error-distribution plots, dialog log statistics. Reviewer-EEmB-Q3 + Q4 figures. | scripts read cached eval JSONs |

## Performance notes

| Step | Throughput / cost | Notes |
|---|---|---|
| `.sens` extraction (ScanNet) | ~30 s/scan with `SENS_WORKERS=16` | I/O bound; SSD helps |
| Keyframe selection (NBV + DPP) | ~90 s/scene on A100 | rasterization-bound |
| Description generation (GPT-4o-mini, all keyframes) | ~$0.01 / scene | depends on # of keyframes |
| Fine localization (Tab. 4 row, 100 scenes) | ~20 min on A100, ~30 min on RTX 5090 | grid-step 0.25 m |
| Tab. 5 (1,319-scan full pool) | ~5 h on A100 | most time is per-scene mesh I/O + raycasting |
| Description **parsing** (GPT-4o-mini) | ~$0.001 / frame, ~2 frames/sec | bottleneck = OpenAI rate-limit |
| GPT-5.5 vision baseline (1,915 calls, both datasets) | ~$31, ~22 min | concurrency 8 |
| Human annotation site | <100 ms/request | scales to hundreds of concurrent annotators on free-tier hosting |

## Troubleshooting

- **`FileNotFoundError` on 3RScan poses** — ensure each scene's
  `sequence.zip` was extracted. Run the extraction step manually
  via `scripts/dataset/extract_3rscan_sequence.sh <uuid>`.
- **Mesh index mismatch** — invalid faces are filtered automatically
  in the loader; this warning is non-fatal.
- **`No frames after IQA filtering`** — lower
  `dataset.scannetpp.iqa_threshold` (default 0.55) or
  `dataset.3rscan.iqa_threshold`.
- **PyTorch3D import errors** — `torch` and `pytorch3d` need matching
  CUDA. Rebuild pytorch3d from source against your current torch.
- **Annotation site can't load mesh** — check that the full-res or
  decimated `.ply` exists at `data/scans/<scene>/<scene>_vh_clean*.ply`;
  see [`tools/annotation_website/README.md`](tools/annotation_website/README.md#data-flow).

## Citation

Please cite our paper if you find this code useful. A full BibTeX
entry will be added here on camera-ready acceptance; the placeholder
below records the title, venue, and year for reference:

```bibtex
@inproceedings{langloc2026,
  title     = {LangLoc: Language-Based 3D Indoor Localization},
  author    = {<authors>},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

## License

This repository is released under the licence in [`LICENSE`](LICENSE).
ScanNet and 3RScan have separate licences — see their respective
terms-of-use documents before downloading.
