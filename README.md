# Music Genre Classification — Capstone Deep Learning Project

**Team:** Nochiro's Group
**Training Environment:** Google Colab (GPU runtime)

---

## Overview

An end-to-end music genre classification system built as a capstone deep learning project.
The project classifies a song's genre from audio, demonstrating every major technique covered
across the semester's DL experiments — all on one dataset, in one coherent pipeline.

## Dataset

| Property | Value |
|---|---|
| Dataset | [FMA (Free Music Archive)](https://github.com/mdeff/fma) — **large subset** |
| Size | 106,574 tracks, 30-second clips each, ~93 GB, MP3 @ 320 kbps |
| Genre taxonomy | 161 raw genres → cleaned to top-level genre categories |

## Project Stages

Each stage maps to a semester experiment concept:

| Stage | Experiment Concept | Description |
|---|---|---|
| 1 | Expt 1 — Data Pipeline | Reusable ingestion, cleaning, and feature-extraction pipeline cached to Drive |
| 2 | Expt 2 — Baseline + Tracking | Simple ANN on aggregated MFCC features; runs logged in MLflow |
| 3 | Expt 3 — CNN + Real-World Constraints | CNN on mel-spectrogram images; class-weighting, SpecAugment augmentation |
| 4 | Expt 4 — Transfer Learning | Pretrained ResNet/VGG; fine-tuning vs. feature extraction comparison |
| 5 | Expt 6 — Sequence Modeling (adapted) | LSTM over spectrogram frame sequences across time |
| 6 | Expt 7 — Model Optimization | Quantize + prune the best model; latency vs. accuracy tradeoff |
| 7 | Report | Results write-up mapping each stage to its experiment concept |

### Nice-to-Have Extensions

| Stage | Description |
|---|---|
| 8 | Lyrics-based genre classifier (RNN/LSTM NLP pipeline) |
| 9 | Uncertainty + explainability (MC Dropout, calibration, saliency maps) |
| 10 | Full-stack deployment (FastAPI + React, Dockerized) |

## Tech Stack

| Layer | Technology |
|---|---|
| Feature extraction | `librosa` |
| Classical ML utilities | scikit-learn |
| Deep learning framework | PyTorch |
| Experiment tracking | MLflow |
| Transfer learning models | torchvision (ResNet, VGG) |
| Model optimization | `torch.quantization`, `torch.nn.utils.prune` |
| Training environment | Google Colab (GPU runtime) |
| Persistent storage | Google Drive (mounted) |

## Repository Structure

```
music-genre-capstone/
├── README.md
├── requirements.txt
│
├── notebooks/                      # one Colab notebook per stage
│   ├── 01_data_pipeline.ipynb
│   ├── 02_baseline_ann.ipynb
│   ├── 03_cnn.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 05_lstm_sequence.ipynb
│   ├── 06_optimization.ipynb
│   ├── 07_lyrics_nlp.ipynb         # nice-to-have
│   ├── 08_uncertainty.ipynb        # nice-to-have
│   └── 09_deployment_prep.ipynb    # nice-to-have
│
├── data_pipeline/                  # shared code imported by notebooks
│   ├── download_fma.py
│   ├── clean_genres.py
│   ├── feature_extraction.py
│   └── dataset_split.py
│
├── models/                         # shared model definitions
│   ├── baseline_ann.py
│   ├── cnn.py
│   ├── transfer_model.py
│   └── lstm_sequence.py
│
├── optimization/
│   ├── quantize.py
│   ├── prune.py
│   └── latency_accuracy_report.py
│
├── nlp_lyrics/                     # nice-to-have
├── uncertainty/                    # nice-to-have
├── serving/                        # nice-to-have — FastAPI + React
│
└── reports/
    ├── experiment_comparison.md
    └── final_report.md
```

## Setup

### Google Colab (recommended)

1. Clone the repository at the start of each Colab session:
   ```python
   !git clone https://github.com/<your-org>/music-genre-capstone.git
   %cd music-genre-capstone
   !pip install -r requirements.txt
   ```

2. Mount Google Drive for persistent storage:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Run notebooks in order — each stage loads cached outputs from previous stages stored on
   Drive.

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-org>/music-genre-capstone.git
   cd music-genre-capstone
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Notes

- **Feature extraction is slow** — run it once via `01_data_pipeline.ipynb`, cache results to
  Drive, and load from cache in all downstream notebooks.
- **Checkpoint frequently** — save model checkpoints to Drive every epoch to guard against
  Colab session disconnects.
- **Confirm your GPU** — run `!nvidia-smi` at the start of each Colab session to verify the
  assigned GPU type.
- Each notebook is self-contained: it mounts Drive, loads cached features/checkpoints, and
  does not assume in-memory state from other notebooks.
