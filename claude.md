# Music Genre Classification — Capstone DL Project

**Type:** DL Subject Mini Project / Capstone
**Group:** Nochiro's Group
**Training environment:** Google Colab (GPU runtime)

---

## Objective

Classify a song's genre from audio, while demonstrating — on one dataset, in one coherent
pipeline — every technique covered across the semester's DL experiments: data pipelining,
baseline modeling with experiment tracking, CNNs with real-world constraints, transfer
learning, sequence modeling (LSTM), model optimization for deployment, and an RNN/embedding-
based NLP pipeline.

Scope is split into **ESSENTIAL** (required to demonstrate all experiment concepts) and
**NICE-TO-HAVE** (build only if time permits) — see Section 4.

---

## Dataset

| Property | Value |
|---|---|
| Dataset | FMA (Free Music Archive) — **"large" subset** |
| Size | 106,574 tracks, 30-second clips each, ~93GB, MP3 @ 320kbps |
| Genre taxonomy | 161 raw genres — clean to top-level genre categories (drop tags with very few tracks) |
| Optional lyrics data | Not bundled with FMA — needs pairing with a lyrics source/API (nice-to-have branch only) |

---

## Google Colab Training — Environment Notes

**Your setup:** Google AI Pro subscription — this bundles 5TB of Google Drive storage and
200 Compute Credit Units for Colab, giving priority access to better GPUs (L4/A100/V100
instead of a shared free-tier T4) plus higher-memory VMs and longer session runtimes. This
removes the storage constraint entirely and meaningfully reduces (though doesn't eliminate)
disconnect risk. Plan around what's below accordingly — lighter caution than a free-tier setup
would need, but the same good habits still pay off.

### 1. Storage: no longer a bottleneck
- 93GB for FMA-large is trivial against 5TB — no need to fall back to FMA-medium.
- Since space isn't tight, cache generously: keep MFCCs, mel-spectrograms, AND frame-sequence
  features all on Drive simultaneously rather than deleting one representation to make room
  for another.
- Still mount Drive at the start of every notebook
  (`from google.colab import drive; drive.mount('/content/drive')`) and do all downloads,
  feature extraction, and checkpoints there — Colab's local `/content/` disk still resets
  every session regardless of your storage plan.

### 2. Feature extraction: do it once, cache it, don't repeat it
- Extracting features from 106,574 audio files is slow regardless of storage headroom — run
  it **once**, save to Drive, and have every downstream notebook (baseline, CNN, transfer
  learning, LSTM) load the cached features instead of reprocessing audio each time.

### 3. Session limits: checkpoint anyway
- Priority GPU access reduces pre-emption risk but doesn't eliminate disconnects entirely.
- Still **save model checkpoints to Drive every epoch (or every few epochs)** and structure
  training scripts to resume from the last checkpoint — cheap insurance against losing a long
  run.

### 4. GPU tier
- With Compute Units, you should get L4/A100/V100 priority rather than being stuck on T4 —
  size batch sizes assuming you have real headroom, but confirm the actual assigned GPU each
  session (`!nvidia-smi`) since allocation can still vary.

### 5. Notebook organization
- Use **separate Colab notebooks per stage** (data pipeline, baseline, CNN, transfer learning,
  LSTM, optimization) so the group can split work and avoid one person's long-running cell
  blocking everyone else.
- Each notebook should start by mounting Drive and loading cached features/checkpoints from
  previous stages — never assume in-memory state carries over between sessions.

---

## Experiment-to-Stage Mapping

| Stage | Concept reused | What we actually do | Status |
|---|---|---|---|
| 1 | Expt 1 — Data Pipeline | Reusable ingestion/cleaning/feature-extraction pipeline, cached to Drive | **ESSENTIAL** |
| 2 | Expt 2 — Baseline + Tracking | Simple ANN on aggregated MFCC features; every run logged in MLflow | **ESSENTIAL** |
| 3 | Expt 3 — CNN + real-world constraints | CNN on mel-spectrogram images; class-weighting for imbalance; SpecAugment-style augmentation | **ESSENTIAL** |
| 4 | Expt 4 — Transfer Learning | Pretrained ResNet/VGG — compare fine-tuning vs. feature extraction against the from-scratch CNN | **ESSENTIAL** |
| 5 | Expt 6 — Sequence modeling (adapted) | LSTM over the sequence of spectrogram frames across time. *Honest note: this reuses the sequence-modeling skill, not literal forecasting — say so plainly in the report.* | **ESSENTIAL** |
| 6 | Expt 7 — Model Optimization | Quantize + prune the best-performing model; measure latency vs. accuracy tradeoff | **ESSENTIAL** |
| 7 | RNN/LSTM NLP pipeline (IMDB-style) | Lyrics-based genre classifier: tokenize + pad, embedding layer, LSTM, precision/recall | NICE-TO-HAVE |
| 8 | Uncertainty + explainability | Monte Carlo Dropout confidence estimates, calibration error, spectrogram saliency map | NICE-TO-HAVE |
| 9 | Full-stack deployment | FastAPI + React, upload audio → predicted genre + confidence, Dockerized on a cloud VM | NICE-TO-HAVE |

---

## Tech Stack — Essentials

| Layer | Technology | Why |
|---|---|---|
| Feature extraction | `librosa` | MFCCs, mel-spectrograms |
| Baseline + classical bits | scikit-learn | Train/test split, class-weight computation |
| Deep learning framework | PyTorch | CNN, LSTM, transfer learning models |
| Experiment tracking | MLflow (run in Colab, log to Drive or a shared tracking URI) | Same tool as Expt 2 |
| Transfer learning source models | torchvision pretrained ResNet/VGG | Fine-tuning vs. feature-extraction, same as Expt 4 |
| Model optimization | `torch.quantization`, `torch.nn.utils.prune` | Same technique as Expt 7 |
| Training environment | **Google Colab (GPU runtime)** | See Colab-specific notes above |
| Persistent storage | **Google Drive (mounted)** | Dataset, cached features, checkpoints — survives session resets |

---

## Repository Structure

```
music-genre-capstone/
├── README.md
├── requirements.txt
│
├── notebooks/                      # one Colab notebook per stage
│   ├── 01_data_pipeline.ipynb      # Expt 1 — run once, caches features to Drive
│   ├── 02_baseline_ann.ipynb       # Expt 2 — loads cached features
│   ├── 03_cnn.ipynb                # Expt 3
│   ├── 04_transfer_learning.ipynb  # Expt 4
│   ├── 05_lstm_sequence.ipynb      # Expt 6 (adapted)
│   ├── 06_optimization.ipynb       # Expt 7
│   ├── 07_lyrics_nlp.ipynb         # NICE-TO-HAVE
│   ├── 08_uncertainty.ipynb        # NICE-TO-HAVE
│   └── 09_deployment_prep.ipynb    # NICE-TO-HAVE — export model for serving
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
├── nlp_lyrics/                     # NICE-TO-HAVE
├── uncertainty/                    # NICE-TO-HAVE
├── serving/                        # NICE-TO-HAVE — FastAPI + React
│
└── reports/
    ├── experiment_comparison.md
    └── final_report.md
```

**Note on structure vs. Colab:** the `notebooks/` folder is what you'll actually run in Colab
day to day. The `data_pipeline/`, `models/`, etc. folders hold importable `.py` modules —
either upload them to Drive and `sys.path.append()` them in Colab, or `!git clone` the repo
at the start of each notebook session so the shared code is always available without manual
uploads.

---

## Build Phases

1. **Data Pipeline (Expt 1)** — download FMA-large to Drive, clean genre taxonomy, extract
   and cache MFCC + mel-spectrogram + frame-sequence features. *This is the slowest, most
   Colab-constraint-sensitive phase — budget extra time.*
2. **Baseline + Tracking (Expt 2)** — simple ANN on aggregated features, MLflow logging.
3. **CNN (Expt 3)** — CNN on spectrograms, class-weighting, augmentation.
4. **Transfer Learning (Expt 4)** — fine-tune vs. feature-extract pretrained CNN, benchmark
   against from-scratch CNN.
5. **Sequence Modeling (Expt 6, adapted)** — LSTM over frame sequences, compare results.
6. **Model Optimization (Expt 7)** — quantize + prune best model, report latency/accuracy
   tradeoff.
7. **Report** — write up results, explicitly mapping each stage back to its experiment
   concept.
8. *(Nice-to-have)* **Lyrics/NLP branch** — embedding + LSTM text classifier on lyrics.
9. *(Nice-to-have)* **Uncertainty + explainability** — MC Dropout, calibration, saliency maps.
10. *(Nice-to-have)* **Full deployment** — FastAPI + React, Dockerized, cloud-hosted.

---

## Report Framing (for your teacher)

> "We built a single end-to-end music genre classification project that deliberately reuses
> every major technique from this semester's DL experiments: a reusable data pipeline
> (Expt 1), a tracked baseline model (Expt 2), a CNN handling real-world class imbalance and
> augmentation (Expt 3), transfer learning with a fine-tuning vs. feature-extraction
> comparison (Expt 4), an LSTM-based sequence model applying the temporal-modeling skill from
> time-series forecasting (Expt 6), and model optimization via quantization and pruning for
> deployment (Expt 7)."

---

*Essentials (Phases 1–7) fully satisfy the stated experiment-concept requirements on their
own. Nice-to-haves (Phases 8–10) extend this into a stronger standalone portfolio piece if
time permits, but are not required for the subject deliverable.*
