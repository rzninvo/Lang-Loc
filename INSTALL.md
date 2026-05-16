# Installation

## Requirements

| Item | Version / Notes |
|---|---|
| Python | 3.10 or newer |
| GPU | CUDA-capable, 8 GB+ VRAM recommended. Tested on A100 (40 GB) and RTX 5090. |
| Disk | ~60 GB for the ScanNet + 3RScan paper subsets. The full ScanNet release is 1.2 TB. |
| OS | Linux. Tested on Ubuntu 22.04 and 24.04. macOS is fine for the analysis tools but not for the rasterizer. |
| OpenAI API key | Needed for description generation and parsing (Sec. 3.1 of the paper). |

## Step-by-step setup

### 1. Conda environment

```bash
conda create -n langloc python=3.10 -y
conda activate langloc
```

### 2. Repository

```bash
git clone https://github.com/<your-org>/langloc.git
cd langloc
pip install -r requirements.txt
pip install -e .
```

### 3. PyTorch with CUDA

Adjust the index URL for your CUDA driver. Versions are pinned to
match the PyTorch3D build below.

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

### 4. PyTorch3D from source

CUDA bindings are version-sensitive, so building from source
against your installed PyTorch is required for a working build:

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . --no-build-isolation && cd ..
python -c "from pytorch3d.structures import Meshes; print('pytorch3d ok')"
```

### 5. spaCy word2vec embeddings

Used by the parsed-caption grounder (Sec. 3.3) to match text labels
to scene-graph node labels.

```bash
python -m spacy download en_core_web_lg
```

### 6. Annotation website (optional)

Only needed if you plan to run the FastAPI crowdsourcing site under
[`tools/annotation_website/`](tools/annotation_website/). It has an
isolated `requirements.txt`:

```bash
pip install -r tools/annotation_website/requirements.txt
```

## Environment variables

The repo reads its secrets from a `.env` file at the repo root.
Start from the template:

```bash
cp .env.example .env
```

Then edit `.env` and set the variables relevant to the run you plan
to do:

| Variable | Used by | Required? |
|---|---|---|
| `OPENAI_API_KEY` | `langloc.dataset.annotation.parse_descriptions`, [`tools/baselines/gpt_vlm/`](tools/baselines/gpt_vlm/) | Yes for any parsing or GPT-5.5 baseline run. |
| `LANGLOC_GPT_MAX_USD` | [`tools/baselines/gpt_vlm/run_descriptions.py`](tools/baselines/gpt_vlm/run_descriptions.py) | Optional cost cap for vision-baseline runs. |
| `LANGLOC_COOKIE_SECRET` | [`tools/annotation_website/`](tools/annotation_website/) | Yes if you run the website. |
| `LANGLOC_ADMIN_TOKEN` | [`tools/annotation_website/`](tools/annotation_website/) admin routes | Yes if you run the website. |

A separate `.env.example` template for the website itself lives at
[`tools/annotation_website/.env.example`](tools/annotation_website/.env.example).

## Sanity check

After the steps above, this one-liner imports the core modules:

```bash
python -c "
from langloc.retrieval.eval import main as r_main
from langloc.localization.cli import main as l_main
from langloc.dialogue.cli import main as d_main
from langloc.utils.seed import CANONICAL_SEED, set_seed
print('OK. CANONICAL_SEED =', CANONICAL_SEED)
"
```

Expected output: `OK. CANONICAL_SEED = 42`.

## Troubleshooting

**`ImportError` on `pytorch3d`.**
Most commonly a CUDA / PyTorch version mismatch. Confirm
`torch.version.cuda` matches the CUDA you compiled PyTorch3D
against, then rebuild PyTorch3D from source (step 4) against your
current torch. Pre-built wheels are usually too brittle.

**`OSError: [E050] Can't find model 'en_core_web_lg'`.**
spaCy embeddings were not installed. Re-run step 5.

**`openai.AuthenticationError`.**
`OPENAI_API_KEY` is missing or wrong in `.env`. The key should start
with `sk-`. If you only want to run already-parsed scenes (Tab. 4 /
Tab. 5 with `--skip_precompute`), the key is not strictly required.

**`CUDA out of memory` during keyframe selection.**
Lower `dataset.scannet.iqa_batch_size` or
`dataset.scannet.rasterization_batch_size` (both default 2 in
[`configs/dataset/default.yaml`](configs/dataset/default.yaml))
via Hydra override. The 3RScan side has matching keys under
`dataset.3rscan.*`.

**Conda env clashes with system Python.**
Always `conda activate langloc` before any pip install or run.
The reproducer scripts default `PYTHON_BIN` to
`$HOME/miniconda3/envs/langloc/bin/python`; override that env var
if your conda install lives elsewhere.

For runtime errors after install (data not found, Hydra config
errors), see [CONFIG.md](CONFIG.md#troubleshooting).
