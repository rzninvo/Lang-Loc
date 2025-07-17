# 🏗️ ScanNet Scene Downloader & Renderer

This project helps you automatically download specific scenes from the [ScanNet dataset](http://www.scan-net.org/) and render **multi-view images** from them using `Open3D`.

---

## 📦 What’s Included?

- `download_subset.sh`: Bash script to download all required files for a given `scene_id`
- `render_scene.py`: Python script to generate 6 synthetic views around a scene
- Example output: `output/view_1.png` ... `view_6.png`

---

## ✅ Requirements

### Python Packages

Make sure you have Python ≥3.7 and the following packages:

```bash
pip install open3d numpy
````

> 🧠 Note: You need an OpenGL-compatible GPU and drivers. Open3D uses **EGL headless rendering** on Linux.

---

## 📂 Directory Structure

After running the scripts, your project will look like:

```
project-root/
│
├── download_subset.sh
├── render_scene.py
├── data/
│   └── scans/
│       └── scene0000_00/
│           ├── scene0000_00_vh_clean_2.ply
│           ├── ...
│           └── output/
                ├── camera_pose.json
│               ├── view_1.png
│               ├── ...
│               └── view_6.png
```

---

## 🚀 How to Use

### 1. Make the download script executable

```bash
chmod +x download_subset.sh
```

---

### 2. Render a scene

To download and render a specific ScanNet scene:

```bash
python3 render_scene.py scene0000_00
```

This will:

1. Check if `scene0000_00` exists locally.
2. If not, download it via `download_subset.sh`.
3. Render **6 images** from different camera angles and save them to:

```
data/scans/scene0000_00/output/
```

---

## 🔧 What Does It Download?

For each `scene_id`, the following files are fetched:

* `*_vh_clean_2.ply` – cleaned mesh
* `*_vh_clean_2.labels.ply` – semantic labels
* `*_vh_clean_2.0.010000.segs.json` – segment mapping
* `*.aggregation.json` – object instance annotations
* `*.txt` – metadata (e.g., intrinsics)
* `scannetv2-labels.combined.tsv` – global label map

---

## 🖼️ What Do the Rendered Images Show?

Each rendered image is a **60° rotation step** around the mesh, simulating a virtual camera orbit. You get **6 total views** (front, sides, back, etc.), rendered at:

* Resolution: 1920×1080
* Field of View: 90° vertical (customizable)
* View height: 1.0 m above mesh center

---

## 📌 Troubleshooting

* ❌ `PermissionError` on script:
  → Run `chmod +x download_subset.sh`

* ❌ Open3D rendering doesn't show expected mesh:
  → Make sure you're using `read_triangle_mesh` (not `read_point_cloud`)
  → Make sure you use a recent version of Open3D (≥0.16 recommended)

---

## 📬 Questions?

Feel free to open an issue or reach out if you'd like to:

* Add depth map rendering
* Use real camera poses from ScanNet
* Render semantic overlays
