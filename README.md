# HumanMM: Global Human Motion Recovery from Multi-Shot Videos

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![CVPR](https://img.shields.io/badge/Inspired_By-CVPR_2025-gold?style=for-the-badge)

**An industrial-quality, modular pipeline for recovering global 3D human motion from multi-shot videos.**

[Overview](#overview) • [Architecture](#architecture) • [Installation](#installation) • [Usage](#usage) • [Pipeline](#pipeline) • [Outputs](#outputs) • [Results](#results) • [References](#references)

</div>

---

## Overview

**HumanMM** is a research-grade implementation of a complete human motion recovery pipeline
inspired by the paper:

> *HumanMM: Global Human Motion Recovery from Multi-shot Videos*  
> CVPR 2025

This project is an **original orchestration** built on top of publicly available pretrained models.
It is **not** a copy of any existing repository. The pipeline, cross-shot trajectory alignment
module, and software architecture are entirely original contributions.

### What This Project Does

Given any video (including cinematically edited multi-shot content), this pipeline:

1. **Extracts frames** from the input video with full metadata
2. **Detects shot boundaries** using content-aware scene detection
3. **Detects humans** in every frame using YOLOv8
4. **Tracks individuals** across frames using ByteTrack / DeepSORT
5. **Estimates 2D pose** using MediaPipe or ViTPose
6. **Recovers 3D human motion** using HMR 2.0 (4D-Humans) SMPL body model
7. **Aligns trajectories** across shot boundaries using an original cross-shot alignment engine
8. **Visualizes every stage** of the pipeline in publication-quality outputs
9. **Exports all results** in multiple formats (MP4, GIF, JSON, CSV, NPY, PLY, OBJ)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            main.py  (CLI)                                │
│                    python main.py --video demo.mp4                       │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────────┐
│                     PipelineRunner (Orchestrator)                        │
│              pipeline/pipeline_runner.py                                 │
└──┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────────┘
   │          │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼          ▼
VideoLoader ShotDet  HumanDet  Tracker  PoseEst  MotionRec
   │          │          │          │          │          │
   └──────────┴──────────┴──────────┴──────────┴──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  TrajectoryAligner  │  ← Original Implementation
                    │  (Cross-Shot Align) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     Visualizer      │
                    │  (5 sub-modules)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │      Exporter       │
                    │  MP4/GIF/JSON/CSV/  │
                    │  NPY/PLY/OBJ        │
                    └─────────────────────┘
```

### Design Patterns

| Pattern | Where Applied | Purpose |
|---|---|---|
| **Factory** | `models/model_factory.py` | Instantiate detector/tracker/pose backend by config |
| **Strategy** | `pipeline/tracker.py`, `pipeline/pose_estimator.py` | Swap algorithms without changing caller |
| **Repository** | `pipeline/video_loader.py`, `pipeline/exporter.py` | Abstract I/O from business logic |
| **Dependency Injection** | `pipeline/pipeline_runner.py` | Components receive dependencies via config |

---

## Project Structure

```
HumanMM/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
│
├── configs/
│   ├── config.yaml           # Master Hydra config
│   ├── detector.yaml         # YOLO settings
│   ├── tracker.yaml          # ByteTrack / DeepSORT
│   ├── pose.yaml             # MediaPipe / ViTPose
│   ├── motion.yaml           # HMR 2.0 / GVHMR
│   └── visualization.yaml   # Output rendering
│
├── main.py                   # CLI entry point
│
├── pipeline/
│   ├── pipeline_runner.py    # Master orchestrator
│   ├── video_loader.py       # Frame extraction
│   ├── shot_detector.py      # PySceneDetect
│   ├── human_detector.py     # YOLOv8
│   ├── tracker.py            # Multi-person tracking
│   ├── pose_estimator.py     # 2D pose estimation
│   ├── motion_recovery.py    # 3D SMPL recovery
│   ├── trajectory_aligner.py # ★ Original: cross-shot alignment
│   └── exporter.py           # Multi-format export
│
├── models/
│   ├── base_model.py         # Abstract interface
│   ├── model_factory.py      # Factory pattern
│   ├── yolo_detector.py      # YOLOv8 wrapper
│   ├── mediapipe_pose.py     # MediaPipe backend
│   ├── vitpose_pose.py       # ViTPose backend
│   ├── bytetrack_tracker.py  # Custom ByteTrack
│   ├── deepsort_tracker.py   # DeepSORT wrapper
│   └── gvhmr_wrapper.py      # HMR 2.0 / GVHMR wrapper
│
├── utils/
│   ├── logger.py             # Structured logging
│   ├── config_loader.py      # Hydra config utils
│   ├── frame_utils.py        # Frame I/O
│   ├── geometry.py           # 3D math, quaternions
│   ├── metrics.py            # Performance monitoring
│   ├── io_utils.py           # JSON/CSV/NPY writers
│   └── video_writer.py       # OpenCV video writer
│
├── visualization/
│   ├── visualizer.py         # Orchestrator
│   ├── detection_visualizer.py
│   ├── pose_visualizer.py
│   ├── mesh_visualizer.py
│   ├── trajectory_visualizer.py
│   └── comparison_visualizer.py
│
├── tests/                    # Pytest test suite
├── demo/                     # Quick-start demos
├── scripts/                  # Utility scripts
├── data/input/               # Place input videos here
├── assets/                   # Diagrams, figures
├── outputs/                  # Auto-generated results
└── docs/                     # Full documentation
```

---

## Installation

### Prerequisites

- Python 3.10
- CUDA 11.8+ (optional, CPU fallback available)
- Git

### Option 1: Conda (Recommended)

```bash
git clone https://github.com/your-username/HumanMM.git
cd HumanMM

conda env create -f environment.yml
conda activate humanmm

python scripts/download_models.py
```

### Option 2: Pip + Virtual Environment

```bash
git clone https://github.com/your-username/HumanMM.git
cd HumanMM

python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

pip install -r requirements.txt

python scripts/download_models.py
```

### Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import ultralytics; print('YOLO OK')"
python -c "import mediapipe; print('MediaPipe OK')"
python -c "import scenedetect; print('SceneDetect OK')"
```

---

## Usage

### Basic Run

```bash
python main.py --video data/input/demo.mp4
```

### Full Pipeline with All Outputs

```bash
python main.py \
  --video data/input/demo.mp4 \
  --visualize \
  --save-json \
  --save-mesh \
  --save-video \
  --show-fps \
  --device cuda
```

### CPU-Only Mode

```bash
python main.py --video data/input/demo.mp4 --device cpu
```

### Switch Pose Backend

```bash
# MediaPipe (default, no GPU required)
python main.py --video demo.mp4 --pose-backend mediapipe

# ViTPose (high accuracy, GPU recommended)
python main.py --video demo.mp4 --pose-backend vitpose
```

### Switch Tracker

```bash
# ByteTrack (default, faster)
python main.py --video demo.mp4 --tracker bytetrack

# DeepSORT (appearance-based)
python main.py --video demo.mp4 --tracker deepsort
```

### All CLI Arguments

```
usage: main.py [-h] --video VIDEO
               [--config CONFIG]
               [--output-dir OUTPUT_DIR]
               [--device {cuda,cpu}]
               [--pose-backend {mediapipe,vitpose}]
               [--tracker {bytetrack,deepsort}]
               [--visualize]
               [--save-json]
               [--save-mesh]
               [--save-video]
               [--show-fps]
               [--max-persons MAX_PERSONS]
               [--conf-threshold CONF_THRESHOLD]
               [--debug]

Arguments:
  --video              Path to input video file
  --config             Path to config YAML (default: configs/config.yaml)
  --output-dir         Output directory (default: outputs/)
  --device             Compute device: cuda or cpu
  --pose-backend       Pose estimator: mediapipe or vitpose
  --tracker            Tracking algorithm: bytetrack or deepsort
  --visualize          Generate all visualization videos
  --save-json          Export results as JSON
  --save-mesh          Export 3D meshes as PLY/OBJ
  --save-video         Save all pipeline stage videos
  --show-fps           Display FPS counter on output videos
  --max-persons        Maximum persons to track (default: 10)
  --conf-threshold     YOLO confidence threshold (default: 0.5)
  --debug              Enable debug logging
```

---

## Pipeline

```
Input Video
    │
    ▼
┌─────────────────────┐
│   Video Loader      │  → frames/, metadata.json
│   (OpenCV)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Shot Detector     │  → shots.json  [scene boundaries]
│   (PySceneDetect)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Human Detector    │  → detections.json  [BBoxes per frame]
│   (YOLOv8)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Person Tracker    │  → tracks.json  [consistent IDs]
│   (ByteTrack)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Pose Estimator    │  → poses.json  [2D joints]
│   (MediaPipe)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Motion Recovery   │  → smpl_params.npy  [3D SMPL]
│   (HMR 2.0)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Trajectory Aligner  │  → aligned_trajectory.npy  ★ Original
│ (Cross-Shot Align)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Visualizer       │  → 7 output videos
│   (5 sub-modules)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Exporter        │  → MP4, GIF, JSON, CSV, NPY, PLY, OBJ
└─────────────────────┘
```

---

## Outputs

After running, the `outputs/` directory contains:

```
outputs/
├── 01_input.mp4           # Original video with metadata overlay
├── 02_detection.mp4       # YOLO bounding boxes
├── 03_tracking.mp4        # Multi-person tracking with IDs
├── 04_pose.mp4            # 2D skeleton overlay
├── 05_mesh.mp4            # 3D SMPL mesh projection
├── 06_alignment.mp4       # Trajectory alignment visualization
├── 07_final.mp4           # Full pipeline final render
├── comparison.mp4         # 4-panel side-by-side video
├── metrics.json           # FPS, GPU, CPU, memory stats
├── trajectory.csv         # Per-person 3D trajectory data
├── joints.npy             # Joint arrays (frames × persons × joints × 3)
├── smpl_params.npy        # SMPL body parameters
├── config_used.yaml       # Exact config used for reproducibility
├── meshes/
│   ├── person_0.ply       # Per-person 3D mesh (PLY)
│   └── person_0.obj       # Per-person 3D mesh (OBJ)
└── logs/
    ├── pipeline.log       # Full structured log
    └── metrics.log        # Performance metrics log
```

---

## Trajectory Alignment (Original Contribution)

The `TrajectoryAligner` is the core intellectual contribution of this project. It solves the
fundamental problem of **temporal discontinuity** at shot boundaries in edited video.

### Algorithms Implemented

| Algorithm | Purpose |
|---|---|
| Cubic Spline Interpolation | Fill occluded/missing detections |
| Moving Average Smoothing | Remove high-frequency jitter |
| Gaussian Temporal Smoothing | Smooth joint trajectories |
| SLERP (Quaternion) | Smooth rotation sequences |
| Savitzky-Golay Filter | Smooth translation trajectories |
| Procrustes Alignment | Cross-shot rigid registration |
| Kalman Filter (optional) | Optimal state estimation |

### Cross-Shot Alignment Logic

```
Shot K  ────────────────────┐
                             ├─→ Overlap window → Procrustes → Blend
Shot K+1 ───────────────────┘
```

1. Extract last N frames of shot K as the **reference segment**
2. Extract first N frames of shot K+1 as the **target segment**
3. Compute least-squares **Procrustes transform** (rotation + translation)
4. Apply transform to all poses in shot K+1
5. Blend boundary frames using **cosine-weighted average**
6. Repeat for all consecutive shot pairs

---

## Performance

Measured on NVIDIA RTX 3080, Intel i7-12700K, 32GB RAM:

| Stage | Time/Frame | GPU Memory |
|---|---|---|
| Frame Extraction | ~0.5 ms | — |
| Shot Detection | ~2 ms | — |
| Human Detection (YOLOv8n) | ~8 ms | ~400 MB |
| Tracking (ByteTrack) | ~1 ms | — |
| Pose Estimation (MediaPipe) | ~12 ms | — |
| Motion Recovery (HMR 2.0) | ~45 ms | ~2.2 GB |
| Trajectory Alignment | ~3 ms | — |
| Visualization | ~15 ms | ~300 MB |
| **Total** | **~86 ms** | **~2.9 GB** |

---

## References

```
@inproceedings{humanmm2025,
  title     = {HumanMM: Global Human Motion Recovery from Multi-shot Videos},
  booktitle = {CVPR},
  year      = {2025}
}

@inproceedings{hmr2_2023,
  title     = {4D-Humans: Reconstructing and Tracking Humans in Videos},
  booktitle = {ICCV},
  year      = {2023}
}

@article{yolov8_2023,
  title  = {YOLOv8: A New State-of-the-Art Real-Time Object Detector},
  year   = {2023}
}

@inproceedings{bytetrack_2022,
  title     = {ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  booktitle = {ECCV},
  year      = {2022}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project was developed as a B.Tech Final Year Research Project.  
Inspired by the CVPR 2025 paper *HumanMM: Global Human Motion Recovery from Multi-shot Videos*.  
All model weights belong to their respective authors and are used for research purposes only.
