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
    <td align="center" valign="middle">
      <img src="media/figures/logos/microsoft_logo.png" height="38"/>
    </td>
  </tr>
</table>
</div>

# LangLoc: Tell Me What You See

End-to-end pipeline for language-based 3D indoor localization,
accompanying our ECCV 2026 paper. Given a free-form description of
an observer's surroundings, LangLoc estimates the observer's 2D
position and heading within a known 3D environment. Evaluated on
[ScanNet](http://www.scan-net.org/) and
[3RScan](http://campar.in.tum.de/public_datasets/3RScan/).

## Pipeline

1. **Dataset creation** (Sec. 3.1): keyframe selection by image
   quality assessment (IQA) plus a two-stage Determinantal Point
   Process (DPP), followed by description generation via
   GPT-4o-mini.
2. **Scene retrieval** (Sec. 3.2): dual-branch Graph Attention
   Network v2 (GATv2) encoder with CLIP features, scoring the
   text-graph against database scene-graphs.
3. **Fine localization** (Sec. 3.3): dense floor-grid scored by
   ray-cast object visibility.
4. **Dialogue disambiguation** (Sec. 3.4): Bayesian posterior over
   poses, refined by targeted yes/no questions.

## Headline results

Numbers below are from rerunning the canonical reproducer scripts on
this repo with `seed=42`. See [REPRODUCE.md](REPRODUCE.md) for the
full per-table breakdown.

| Table | Metric | Paper | This repo |
|---|---|---|---|
| Tab. 1 (ScanScribe, 10-cand) | Top-1 Recall | 76.70 | **76.60 ± 4.29** |
| Tab. 2 (ScanScribe, full) | Top-10 Recall | 91.60 | **90.70 ± 3.58** |
| Tab. 3 (LLM-from-image, fair) | Top-1 Recall | 76.10 | **59.50 ± 5.26** (corrected protocol) |
| Tab. 4(a) 3RScan-100 (no dialog) | Pos median (m) | 1.551 | **1.470** |
| Tab. 4(b) ScanNet-100 (no dialog) | Pos median (m) | 1.314 | **0.998** |
| Tab. 5 full 1319-scene 3RScan | Pos median (m) | 1.308 | **1.230** |

Tabs. 4(b) and 5 beat the paper on position accuracy. Tab. 4(a)
median beats paper and the mean is within 5 cm. Retrieval recall
is within noise everywhere except Tab. 2 Top-5 (within the
reported standard deviation) and the Tab. 3 corrected-protocol
gap documented in [REPRODUCE.md](REPRODUCE.md).

## Quickstart

All three steps below are required in order. Step 3 reads files
laid down by step 2.

```bash
# Step 1. Clone and install. Full details in INSTALL.md.
git clone https://github.com/<your-org>/langloc.git
cd langloc
conda create -n langloc python=3.10 -y && conda activate langloc
pip install -r requirements.txt && pip install -e .
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
python -m spacy download en_core_web_lg
# PyTorch3D from source: see INSTALL.md.

# Step 2. Get the data. Required, not optional. Full details in DATA.md.
#         Easiest path: request the Paper_Dataset Google Drive bundle
#         and unpack into ./data/. The reproducer scripts read
#         data/3RScan/, data/scans/, data/processed_data/eval_pool/,
#         and data/model_checkpoints/graph2graph/paper/ from there.

# Step 3. Reproduce a table. See REPRODUCE.md for the full list.
#         Example: scene retrieval (Tabs. 1, 2, 3), about 3 s once
#         the bundle is in place.
bash scripts/retrieval/reproduce_paper_tables.sh
```

## Documentation

The README is the entry point. Detailed guides live in five focused
files:

| File | What it covers |
|---|---|
| [INSTALL.md](INSTALL.md) | Conda env, PyTorch3D from source, spaCy, environment variables, install-time troubleshooting. |
| [DATA.md](DATA.md) | Google Drive bundle (recommended) and from-scratch alternative; target layout under `data/`; sanity checks. |
| [REPRODUCE.md](REPRODUCE.md) | Per-table reproducer commands, paper-vs-this-repo numbers, dialog-row caveats, optional re-run of the dataset-creation pipeline. |
| [CONFIG.md](CONFIG.md) | Hydra config tree, project layout, performance notes, runtime troubleshooting. |
| [LICENSE](LICENSE) | Project licence. |

## Sub-projects

Self-contained tools with their own READMEs:

| Path | Purpose | Guide |
|---|---|---|
| [`tools/annotation_website/`](tools/annotation_website/) | FastAPI site for crowdsourcing human descriptions and first-person localizations. | [README](tools/annotation_website/README.md) |
| [`tools/baselines/gpt_vlm/`](tools/baselines/gpt_vlm/) | GPT-5.5 vision describer (rebuttal closed-loop-bias defense). | [README](tools/baselines/gpt_vlm/README.md) |
| [`tools/baselines/human/`](tools/baselines/human/) | Extract human-written descriptions from the annotation website's SQLite DB. | [README](tools/baselines/human/README.md) |
| [`tools/eval/`](tools/eval/) | Recall-at-threshold, error-distribution plots, dialog log statistics. | [README](tools/eval/README.md) |

## Citation

A full BibTeX entry will land here on camera-ready acceptance.
Placeholder, recording title, venue, and year:

```bibtex
@inproceedings{langloc2026,
  title     = {LangLoc: Tell Me What You See},
  author    = {<authors>},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

## License

This repository is released under the licence in [LICENSE](LICENSE).
ScanNet and 3RScan have separate licences; review their respective
terms-of-use documents before downloading data.
