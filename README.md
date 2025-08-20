# ScanNet Scene Setup & Keyframe Generation

This repo automates preparing ScanNet scenes for annotation or model training.
It downloads required assets, extracts RGB/Depth/Poses from `.sens`, and runs a **ScanNet++-style Next-Best-View (NBV) pipeline** to pick sharp, diverse keyframes.

Selected frames (RGB, depth, poses, and optional instance/semantic masks) are saved in each scene’s `output/` folder. Temporary raw files can be auto-cleaned.

## Requirements

* Python ≥ 3.10
* Linux recommended (headless EGL rendering + `.sens` extraction)
* ScanNet credentials (see [ScanNet site](http://www.scan-net.org/))

### Python packages

Install everything:

```bash
pip install -r requirements.txt
```

Key dependencies used:

```
numpy
opencv-python
Pillow
tqdm
matplotlib
scikit-learn
scikit-image
scipy
PyYAML
open3d
torch            # + CUDA if available
pytorch3d
brisque
```

> ⚠️ Make sure `torch` and `pytorch3d` are installed with the correct CUDA/CPU build. See [PyTorch docs](https://pytorch.org/get-started/locally/) for wheels.

## Dataset Structure

After running the pipeline, each scene looks like:

```
data/scans/
└── scene0000_00
    ├── intrinsic/
    │   └── intrinsic_color.txt
    ├── output/
    │   ├── cache_scannetpp/         # per-scene caches (NBV order, stats)
    │   ├── camera_pose.json         # poses for selected frames
    │   ├── color/                   # selected RGB frames
    │   │   ├── 002800.jpg
    │   │   └── ...
    │   ├── depth/                   # matched depth maps
    │   │   ├── 002800.png
    │   │   └── ...
    │   ├── pose/                    # matched camera->world matrices
    │   │   ├── 002800.txt
    │   │   └── ...
    │   ├── instance/                # (optional) 16-bit instance masks
    │   └── semantic/                # (optional) 16-bit semantic masks
    ├── scene0000_00.aggregation.json
    ├── scene0000_00_vh_clean_2.0.010000.segs.json
    ├── scene0000_00_vh_clean_2.labels.ply
    └── scene0000_00_vh_clean_2.ply
```

> Raw `color/`, `depth/`, `pose/`, and `.sens` files are extracted temporarily.
> If `--auto_clean` is used (default), they are deleted after keyframe selection.
> The **final outputs** always live in `output/`.

## ⚙️ Configuration

The pipeline is driven by a YAML config (e.g. `config/default.yaml`).

### Key sections

- **`paths.dataset_path`** — root dataset folder (default: `data/scans/`)  
- **`render.output_folder`** — subfolder where outputs are written (default: `output/`)  
- **`scannetpp.*`** — Next-Best-View (NBV) & rasterization parameters, e.g.:  
  - `imq_threshold`: BRISQUE image quality cutoff  
  - `image_downsample_factor`: downsample factor for candidate images  
  - `kmeans_n_clusters`: number of pose clusters for spatial diversity  
  - `min_gain_pixels`: minimum new coverage (pixels) required to keep a view  

### Runtime flags

You can override defaults at runtime with:

- `--debug` — show extra visualization/debug info  
- `--auto_clean` — remove temporary raw folders after keyframe selection  
- `--save_semantic_masks` — export semantic masks (16-bit PNGs)  
- `--save_instance_masks` — export instance masks (16-bit PNGs)  


## Usage

### One scene

Run download + extraction + NBV selection in one command:

```bash
bash scripts/setup_sample_data.sh scene0000_00 config/default.yaml
```

Steps performed:

1. Downloads all required ScanNet files (`.ply`, `.sens`, annotations, labels).
2. Extracts RGB, depth, poses, intrinsics from `.sens`.
3. Runs NBV pipeline:

   * filters blurry frames (BRISQUE, auto-relaxes if needed),
   * maximizes object coverage + diversity,
   * clusters poses and picks best NBV per cluster.
4. Saves selected frames + optional masks into `output/`.
5. Cleans temporary raw files (if `--auto_clean` enabled).

### Many scenes

Loop through the first *N* scenes (default 20):

```bash
# Prepare the first 100 scenes
bash scripts/setup_multiple_scenes.sh config/default.yaml 100
```

Scenes already existing in `data/scans/` are skipped.

### Download only

Fetch scene files and extract RGB/Depth/Poses without NBV:

```bash
bash scripts/download_subset.sh scene0000_00
```

This will:

* Download files listed in config (`*_vh_clean_2.ply`, `.labels.ply`, `.aggregation.json`, `.segs.json`, `.sens`, etc.)
* Extract RGB/depth/poses/intrinsics from `.sens` via `src/utils/sens_reader.py`

## Pipeline Details

* **NBV Selection**
  Iteratively selects frames to maximize coverage across object instances.

* **Pose Clustering**
  After NBV ranking, camera translations are clustered (K-means).
  From each cluster, the **highest-ranked NBV** is chosen.
  → Enforces spatial diversity while keeping coverage intact.

* **BRISQUE Filtering**
  Low-quality frames are discarded.
  If all are filtered out, the threshold is auto-relaxed so enough candidates survive.

* **Outputs**

  * RGB (`color/`)
  * Depth (`depth/`)
  * Poses (`pose/` + `camera_pose.json`)
  * Instance/semantic masks (optional, 16-bit PNGs)
  * Cached NBV stats in `cache_scannetpp/`

## Troubleshooting

* **0 frames after BRISQUE** → Lower `imq_threshold` (e.g., 35–40) or enable auto-relax in config.
* **PyTorch3D errors** → Ensure `torch` and `pytorch3d` match your CUDA version.
* **Headless rendering issues** → Install EGL/GL drivers (`libegl1`, `mesa-utils`, etc.).
* **Scene skipped** → Confirm `config_loader.py` points to the right `base_dir` and `file_types`.