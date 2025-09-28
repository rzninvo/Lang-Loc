# Scene Preparation & Annotation Tool (ScanNet + 3RScan)

This repo provides a **complete pipeline** for working with ScanNet and 3RScan datasets:

1. **Dataset Preparation**

   * Download & extract ScanNet / 3RScan scenes
   * Select sharp, diverse **Next-Best-View (NBV)** keyframes
   * Export RGB, depth, poses, and optional masks

2. **Web-based Annotation Tool**

   * Streamlit app for annotating selected frames
   * Supports saving annotations, marking uninterpretable frames, and managing progress
   * Admin dashboard for reviewing annotations

---

## 🚀 Features

* ScanNet & 3RScan support (UUID or scene IDs)
* Sharp image filtering (BRISQUE)
* NBV-based frame selection with clustering
* 16-bit semantic & instance mask export
* Auto-clean of raw extracted files
* Streamlit UI for annotation & dataset inspection

---

## 🔧 Requirements

* Python ≥ 3.10
* Linux recommended (for rendering & `.sens` extraction)
* Dataset credentials:

  * [ScanNet](http://www.scan-net.org/)
  * [3RScan](http://campar.in.tum.de/public_datasets/3RScan/)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Driven by `config/default.yaml`.

* `paths.dataset_path` → root folder for dataset (`data/scans` or `data/3RScan`)
* `render.output_folder` → subfolder for outputs (default: `output/`)
* `scannetpp.*` and `3rscan.*` → NBV selection settings

---

## 📂 Dataset Preparation

### Single Scene

**ScanNet**

```bash
bash scripts/setup_sample_data.sh --dataset scannet scene0000_00 config/default.yaml
```

**3RScan**

```bash
bash scripts/setup_sample_data.sh --dataset 3RScan <scene-uuid> config/default.yaml
```

### Multiple Scenes

```bash
bash scripts/setup_multiple_scenes.sh --dataset scannet config/default.yaml 50
bash scripts/setup_multiple_scenes.sh --dataset 3RScan config/default.yaml 50
```

### Download Only

```bash
bash scripts/download_subset.sh --dataset scannet scene0000_00
bash scripts/download_subset.sh --dataset 3RScan <scene-uuid>
```

---

## 🖼️ Annotation Tool

The **Streamlit-based web interface** lets you annotate prepared keyframes.

### Run locally

```bash
streamlit run app/app.py
```

Then open the provided URL (default: `http://localhost:8501`).

### Features

* **Sidebar Overview**: scene/image counts, progress, annotation stats
* **Sample Reference**: shows an example frame + description
* **Annotation Panel**:

  * Navigate frames (next/previous)
  * Enter annotations for each view
  * Mark frames as uninterpretable
* **Admin Tables**: review/save/export annotations
* **Instructions**: annotation guidelines inline

### Storage

Annotations and uninterpretable frame IDs are saved as JSON under:

* `CONFIG["paths"]["annotations_file"]`
* `CONFIG["paths"]["uninterpretable_file"]`

---

## 📊 Outputs

Per-scene `output/` folder contains:

* `color/` — selected RGB frames
* `depth/` — depth maps
* `pose/` — camera-to-world matrices
* `camera_pose.json` — consolidated pose file
* `instance/` — instance masks (optional)
* `semantic/` — semantic masks (optional)
* Cached NBV stats (optional, in `cache_scannetpp/`)

---

## 🛠️ Troubleshooting

* **FileNotFound (3RScan poses)** → ensure `sequence.zip` was extracted.
* **Mesh index mismatch** → invalid faces filtered automatically.
* **No frames after BRISQUE** → relax `imq_threshold`.
* **PyTorch3D errors** → install matching `torch` + `pytorch3d`.