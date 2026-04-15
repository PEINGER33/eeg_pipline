# Project Context — EEG Artifact Detection

## Project summary
Development of AI-based methods for elimination of artifacts in biomedical signals (EEG).
The core novelty: a real-time EEG artifact detection pipeline deployable on mobile + web,
without relying on cloud or external computing clusters.

---

## Branch strategy

| Branch | Approach | Status |
|--------|----------|--------|
| `master` | 100% Dart on-device (no server) | Working — do not break |
| `develop` | Python backend (FastAPI) + Flutter client | Active development |

**Current active branch: `develop`**
Instructions below apply to `develop` only.
If on `master`, do NOT introduce any server-side code.

---

## Architecture — branch `develop`

```
Laptop (local server)
└── FastAPI + Python
    ├── MNE / scipy / scikit-learn
    ├── Preprocessing: notch filter, low-pass filter
    ├── ICA: FastICA (MNE) → ORICA → RobustICA
    ├── Classification: heuristics → ICLabel
    └── WebSocket endpoint

Flutter (mobile + web)
└── Connects via WebSocket (same WiFi network)
    ├── Displays raw signal AND cleaned signal side-by-side
    ├── Receives: raw window + cleaned window + artifact labels
    └── NO signal processing — display only
```

**Key decisions for `develop`:**
- Python handles ALL signal processing (preprocessing, ICA, classification)
- Flutter handles ONLY UI and WebSocket communication
- No ICA in Dart on this branch
- Local laptop server for dev, Railway/Render (free tier) for final demo

---

## Full pipeline

```
Raw EEG signal (dataset, simulated real-time via sliding window)
      ↓
Preprocessing        — notch filter (50/60 Hz), low-pass filter  ← DONE
      ↓
ICA decomposition    — FastICA (MNE built-in), then ORICA, RobustICA
      ↓
Component classification  — heuristics (kurtosis, freq power, correlation)
                            then ICLabel (mne-icalabel, pre-trained, 7 classes)
      ↓
Artifact suppression + signal reconstruction
      ↓
WebSocket → Flutter: { raw window, cleaned window, artifact labels }
      ↓
Flutter: side-by-side display — raw | cleaned (same time window, synchronized)
```

---

## Artifact classes

| Class | Source | ICA signature | Detection method |
|-------|--------|---------------|------------------|
| **EOG vertical** | Eye blink | Large amplitude, frontal channels, slow | Correlation with Fp1/Fp2 |
| **EOG horizontal** | Eye movement | Step-like, frontal | Correlation with frontal channels |
| **EMG** | Jaw / neck muscles | High frequency (>30 Hz), diffuse | High-freq power ratio |
| **ECG** | Heartbeat | ~1 Hz regular, posterior channels | Autocorrelation pattern |
| **Line noise** | 50/60 Hz electrical | Spectral peak | Notch filter (pre-ICA) |
| **Movement** | Electrode displacement | Very large amplitude, all channels | Amplitude threshold + ICA |

---

## ICA algorithms — one per mode

The core goal of the project is **online/real-time** EEG processing.
The dataset simulation mimics a live EEG stream — the algorithm must behave accordingly.

| Algorithm | Mode | Type | Rationale |
|-----------|------|------|-----------|
| **ORICA** | Online (real-time simulation) | Streaming | Updates W incrementally per window — the correct algorithm for streaming. **Primary target.** |
| **FastICA** | Offline (EDF/CSV in RAM) | Batch | Needs full signal in RAM, better quality for post-hoc analysis. Implement first as simpler baseline. |

**Why ORICA for online:**
FastICA requires the entire signal to decompose — impossible in true real-time.
ORICA uses natural gradient descent: each new window updates the unmixing matrix W
without needing historical data. It converges after a few seconds of signal.

**Implementation order:** FastICA first (already done for offline), then ORICA for online mode.
RobustICA is deprioritized.

---

## Classification strategy — fine-tuned ICLabel on TUAR

NO heuristics. The classifier is ICLabel fine-tuned on TUAR data.

**Why:** ICLabel has 7 broad classes. TUAR enables splitting Muscle and Channel Noise
into more specific sub-classes → more targeted removal per artifact type.

**Class mapping — ICLabel (7) → fine-tuned model (10):**
```
Brain           → Brain          (unchanged)
Eye             → EYEM           (unchanged)
Muscle          → MUSC           (general muscle)
                  CHEW           (NEW — was merged into Muscle in ICLabel)
                  SHIV           (NEW — was merged into Muscle in ICLabel)
Channel Noise   → ELEC           (electrode artifact)
                  ELPP           (NEW — electrode pop, was merged into Channel Noise)
Heart           → Heart          (unchanged)
Line Noise      → Line Noise     (unchanged, handled by notch pre-ICA)
Other           → Other          (unchanged)
```

**Fine-tuning approach:**
- Base model: ICLabel pre-trained weights (PyTorch, via mne-icalabel)
- Freeze: feature extraction layers
- Retrain: classification head only (7 → 10 output classes)
- Training data: ICA components extracted from TUAR + TUAR artifact labels
- Features: same as ICLabel (PSD, autocorrelation, scalp topomap) — computed with MNE

**Per-class removal strategy (the research contribution):**
| Class | Removal strategy |
|-------|-----------------|
| EYEM  | ICA rejection |
| MUSC  | ICA rejection + aggressive low-pass (>40 Hz) |
| CHEW  | ICA rejection + temporal burst masking |
| SHIV  | ICA rejection + narrow-band filter (tremor frequency) |
| ELEC  | Channel interpolation (preferred over ICA) |
| ELPP  | Amplitude jump detection + interpolation |
| Heart | ICA rejection |

---

## WebSocket protocol (extended)

```json
// Backend → Flutter (existing)
{ "type": "meta", "channels": [...], "sampling_rate": 256, "total_samples": 65536,
  "channel_min": [...], "channel_max": [...], "mode": "offline" }

// Backend → Flutter (existing)
{ "type": "window", "start": 1024, "data": [[...ch0...], [...ch1...], ...] }

// Backend → Flutter (new — ICA mode)
{ "type": "ica_window",
  "start": 1024,
  "raw":   [[...ch0 raw...], [...]],
  "clean": [[...ch0 clean...], [...]],
  "removed_components": [0, 3],
  "labels": ["eye_blink", "muscle"] }

// Flutter → Backend (new)
{ "type": "set_ica", "enabled": true, "algorithm": "fastica", "n_components": 20 }
```

---

## Flutter display — dual window

```
┌──────────────────────┬──────────────────────┐
│   Signal brut        │   Signal nettoyé     │
│   (avec artefacts)   │   (après ICA)        │
│                      │                      │
│   ≈≈≈∧∧∧≈≈≈≈        │   ≈≈≈≈≈≈≈≈≈≈         │
└──────────────────────┴──────────────────────┘
         ↑ même fenêtre temporelle, synchronisées
         ↑ légende: composantes retirées + labels
```

---

## Current phase (15 April 2026)

**Completed:**
- FastAPI server with WebSocket streaming
- EDF (online lazy + offline) and CSV upload
- Sliding window simulation (pause/resume/seek)
- Configurable window size
- Preprocessing: notch filter + low-pass filter (offline: applied on full dataset)

**Immediate next task:**

Step 1 — ICLabel pre-trained (get the full pipeline working end-to-end):
- Install mne-icalabel in venv
- FastICA via MNE on offline data
- ICLabel classification (7 classes, pre-trained, no training required)
- Signal reconstruction (remove artifact components)
- New `ica_window` WebSocket message (raw + clean + labels)
- Flutter: dual-panel display (raw | clean)

Step 2 — Fine-tuning on TUAR (after Step 1 works):
- Separate training project (Colab + Google Drive)
- Parse TUAR EDF + annotations, extract ICA features
- Fine-tune ICLabel head: 7 → 10 classes (add CHEW, SHIV, ELPP)
- Export model_finetuned.pt → drop into backend/
- Per-class removal strategies

**Not yet started:**
- ORICA (online ICA) — after Step 1
- Performance metrics — after Step 2

---

## Dataset
- Public EEG datasets — EEGBCI, DEAP, CHB-MIT or similar
- EDF format preferred (MNE reads natively)
- Simulate real-time by sliding a window along the dataset

---

## Planning milestones
| Date | Milestone |
|------|-----------|
| 09 Apr 2026 | Real-time EEG visualization ← DONE |
| 15 Apr 2026 | Preprocessing (notch + low-pass) ← DONE |
| 22 Apr 2026 | FastICA + heuristic classification + dual display ← current |
| 06 May 2026 | Artifact-specific pipeline (ORICA, RobustICA) |
| 13 May 2026 | User-friendly computational framework |
| 14 May 2026 | Performance metrics |
| 09 Jun 2026 | Final presentation |

---

## Tech stack — `develop` branch
| Layer | Tool |
|-------|------|
| Mobile + Web UI | Flutter 3.x (Dart) |
| Backend | FastAPI + uvicorn (Python) |
| Signal processing | MNE-Python, scipy, numpy |
| ICA | MNE built-in (FastICA), then ORICA, RobustICA |
| IC classification | Heuristics → mne-icalabel |
| Real-time transport | WebSocket (JSON) |
| Dev server | Localhost (laptop) |
| Demo server | Railway or Render (free tier) |

---

## Instructions for Claude on this branch

- Always use FastAPI + WebSocket for server-client communication
- Flutter must NOT do any signal processing — only display
- Python server must NOT require GPU or cloud — runs on a standard laptop
- Use MNE for EEG I/O, preprocessing, and ICA
- ICA is applied per window (batch) — not sample-by-sample unless implementing ORICA
- Classification: start with heuristics, add ICLabel only if explicitly requested
- Dual display: `ica_window` message carries both raw and clean arrays
- Prefer simple solutions — this is a prototype, not production code
- When adding ICA, add `scipy` to requirements if not present; add `mne-icalabel` only for Phase 2
