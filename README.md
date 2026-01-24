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

### Installation

#### Step 1: Check Your CUDA Version

Before installing, check your CUDA toolkit version:

```bash
nvcc --version
```

This will show you the CUDA version (e.g., 12.6, 11.8, etc.). You'll need to install PyTorch with matching CUDA support.

#### Step 2: Create Conda Environment

```bash
# Create a new conda environment
conda create -n scene-annotation python=3.10 -y

# Activate the environment
conda activate scene-annotation
```

#### Step 3: Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (adjust cuda version based on Step 1)
# For CUDA 12.6:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.1:
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Install PyTorch3D from Source

PyTorch3D needs to be built from source to ensure compatibility:

```bash
# Clone PyTorch3D repository
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d

# Install from source
pip install -e . --no-build-isolation

# Return to project directory
cd ..
```

**Note:** Make sure to activate the `scene-annotation` environment before running any scripts or the annotation tool.

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