# 🔬 ForensicAI — Digital Image Forensics & Manipulation Detection Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered platform that combines 6 forensic analysis techniques to detect image manipulation, forgery, and AI-generated content with explainable results.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Modules](#-forensic-modules) • [Training](#-model-training) • [Docker](#-docker-deployment)

</div>

---

## 🎯 Features

- **🔍 Error Level Analysis (ELA)** — Detects spliced regions via JPEG compression artifact analysis
- **📋 Copy-Move Forgery Detection** — Identifies cloned regions using ORB keypoint matching + RANSAC
- **📊 Noise Pattern Analysis** — Detects compositing via wavelet-based noise consistency analysis
- **🤖 AI-Generated Image Detection** — EfficientNet-B0 classifier with transfer learning
- **🌡️ Tampering Localization Heatmap** — Multi-signal fusion with Grad-CAM visualization
- **📂 Metadata Forensics** — EXIF/IPTC analysis for editing software & anomaly detection
- **🖥️ Premium Web Dashboard** — Real-time analysis with interactive visualizations
- **📋 Comprehensive Reports** — Aggregated authenticity scores with confidence metrics

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Web Dashboard (JS)                   │
│          Upload → Scan Animation → Results            │
├──────────────────────────────────────────────────────┤
│                FastAPI REST Backend                    │
│              /api/analyze  •  /api/report              │
├──────────────────────────────────────────────────────┤
│              Analysis Orchestrator                     │
│         Coordinates all modules in parallel            │
├────┬────┬────┬────────┬────────┬────────────────────┤
│ ELA│Copy│Noi │  AI    │Heatmap │    Metadata         │
│    │Move│ se │Detect  │  Gen   │    Forensics        │
│    │    │    │(CNN)   │(Fusion)│                     │
└────┴────┴────┴────────┴────────┴────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/forensicai.git
cd forensicai

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start the application
python run.py
```

Open your browser at **http://127.0.0.1:8000** 🎉

## 🔬 Forensic Modules

### Module 1: Error Level Analysis (ELA)
Detects manipulated regions by re-compressing the image at a known JPEG quality level and amplifying the pixel differences. Tampered areas appear brighter because they were saved at different compression levels.

**Techniques:** JPEG re-compression, pixel difference amplification, block-wise statistical analysis

### Module 2: Copy-Move Forgery Detection
Identifies when a region has been copied and pasted within the same image using feature-based matching with geometric verification.

**Techniques:** ORB keypoint detection, BFMatcher (Hamming distance), Lowe's ratio test, RANSAC geometric consistency

### Module 3: Noise Pattern Analysis
Detects composited images by analyzing sensor noise consistency. Authentic images have uniform noise; composited images mix noise from different sources.

**Techniques:** Wavelet decomposition (DWT), block-wise variance analysis, z-score anomaly detection

### Module 4: AI-Generated Image Detection
Classifies whether an image is a real photograph or AI-generated (GAN, diffusion models) using CNNs with Grad-CAM explainability.

**Techniques:** EfficientNet-B0, transfer learning (ImageNet), Grad-CAM (Explainable AI), binary classification

### Module 5: Tampering Localization Heatmap
Fuses signals from ELA, noise analysis, and edge density to produce a unified tampering probability heatmap showing the most suspicious regions.

**Techniques:** Multi-signal fusion, Gaussian smoothing, percentile-based scoring

### Module 6: Metadata Forensics
Parses EXIF/IPTC/XMP metadata to detect signs of editing software, timestamp anomalies, metadata stripping, and AI generation signatures.

**Techniques:** EXIF parsing, rule-based anomaly detection, editor signature matching

## 🧠 Model Training

### Dataset Setup
Organize your dataset like this:
```
data/dataset/
├── real/          # Authentic photographs
│   ├── img001.jpg
│   └── ...
└── ai/            # AI-generated images (or "fake/")
    ├── img001.jpg
    └── ...
```

### Train the Model (CPU)
```bash
python -m training.train_ai_detector \
    --data_dir data/dataset \
    --epochs 20 \
    --batch_size 8 \
    --lr 0.001 \
    --max_samples 1000
```

**Features:**
- Two-phase training (frozen backbone → full fine-tuning)
- Cosine annealing learning rate schedule
- Early stopping with patience
- Forensic-aware augmentations (JPEG compression artifacts)
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC)

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or with Docker directly
docker build -t forensicai .
docker run -p 8000:8000 forensicai
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web Dashboard |
| `/health` | GET | Health check |
| `/api/analyze` | POST | Full analysis (all 6 modules) |
| `/api/analyze/ela` | POST | Error Level Analysis only |
| `/api/analyze/copymove` | POST | Copy-Move Detection only |
| `/api/analyze/noise` | POST | Noise Pattern Analysis only |
| `/api/analyze/ai-detect` | POST | AI-Generated Detection only |
| `/api/analyze/heatmap` | POST | Tampering Heatmap only |
| `/api/analyze/metadata` | POST | Metadata Forensics only |
| `/docs` | GET | Interactive API Documentation (Swagger) |

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Deep Learning** | PyTorch, torchvision, EfficientNet-B0 |
| **Computer Vision** | OpenCV, scikit-image, Pillow |
| **Signal Processing** | PyWavelets (DWT) |
| **Machine Learning** | scikit-learn (metrics, evaluation) |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS3 (glassmorphism), Vanilla JS |
| **DevOps** | Docker, Docker Compose |
| **Metadata** | ExifRead |

## 📊 ML Skills Demonstrated

- ✅ Transfer Learning (EfficientNet-B0 → ImageNet → custom)
- ✅ Explainable AI / XAI (Grad-CAM visualization)
- ✅ Classical Computer Vision (ORB, SIFT, RANSAC)
- ✅ Signal Processing (Wavelet decomposition)
- ✅ Image Forensics (ELA, noise analysis)
- ✅ Multi-signal Fusion
- ✅ Data Augmentation (incl. JPEG artifact simulation)
- ✅ REST API Design & Deployment
- ✅ Full-stack ML Engineering
- ✅ Docker Containerization

## 📁 Project Structure

```
ForensicAI/
├── app/                          # FastAPI backend
│   ├── main.py                   # App entry point
│   ├── config.py                 # Central configuration
│   ├── routes/analysis.py        # API endpoints
│   └── services/orchestrator.py  # Module coordinator
├── forensics/                    # 6 forensic analysis modules
│   ├── base.py                   # Abstract base class
│   ├── ela_analyzer.py           # Error Level Analysis
│   ├── copymove_detector.py      # Copy-Move Detection
│   ├── noise_analyzer.py         # Noise Pattern Analysis
│   ├── ai_detector.py            # AI-Generated Detection
│   ├── heatmap_generator.py      # Tampering Heatmap
│   └── metadata_analyzer.py      # Metadata Forensics
├── models/                       # ML model architectures
│   └── efficientnet_detector.py  # EfficientNet-B0 binary classifier
├── training/                     # Training pipeline
│   ├── train_ai_detector.py      # Training script
│   └── dataset.py                # Dataset loader
├── utils/                        # Shared utilities
│   ├── image_utils.py            # Image I/O helpers
│   └── visualization.py          # Heatmap & overlay utilities
├── web/                          # Web dashboard
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── run.py                        # One-command launcher
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 📄 License

This project is licensed under the MIT License.

---
