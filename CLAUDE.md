# Project Context — EEG Artifact Detection

## Project summary
Development of AI-based methods for elimination of artifacts in biomedical signals (EEG).
Core research question: for each artifact type, what is the most efficient removal pipeline —
ICA-based or a dedicated direct method?
The pipeline runs locally (laptop), no cloud required, with real-time visualization via Flutter.

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

## Research objective

Compare ICA variants (FastICA, ORICA) and artifact-specific direct methods across a defined
set of artifact classes, each pipeline tailored to its distinctive signal characteristics.

Reference paper: Hsu et al. (2015) "Validating Online Recursive ICA on EEG Data" — used
as methodology template for the comparison (Fig. 2–4: scalp maps, MIR, near-dipolar %).

Evaluation dataset: TUH EEG Artifact Corpus (TUAR) — provides ground truth artifact annotations.

---

## Two-phase evaluation

### Phase 1 — ICA decomposition quality
*"Is the ICA algorithm decomposing sources correctly?"*
- Quantitative: MIR, % near-dipolar components (r.v. < 5%), computation time
- Qualitative: scalp maps of components, activity spectra
- Comparison: FastICA vs ORICA
- Metrics TBD

### Phase 2 — Artifact removal quality
*"Is the cleaned signal actually clean?"*
- Quantitative: TBD (SNR, precision/recall on TUAR annotations, ...)
- Qualitative: topographic maps before / after removal, signal plots
- Comparison: ICA-based removal vs direct methods per artifact class
- Metrics TBD

---

## Architecture — branch `develop`

```
Laptop (local server)
└── FastAPI + Python
    ├── MNE / scipy / scikit-learn
    ├── Preprocessing: notch filter, low-pass filter, FFT filter
    ├── ICA: FastICA (offline) + ORICA (online/streaming)
    ├── ICLabel: component classification (7 classes, pre-trained, no fine-tuning)
    ├── Direct methods: per-artifact non-ICA removal algorithms
    ├── Evaluation: Phase 1 + Phase 2 metrics
    └── WebSocket endpoint

Flutter (mobile + web) — UI in English
└── Connects via WebSocket (same WiFi network)
    ├── Raw signal display
    ├── Clean signal display (side-by-side)
    ├── ICA results visualization (artifact timeline)
    ├── Topography maps
    ├── Performance metrics display
    └── NO signal processing — display only
```

---

## Pipelines

### Mode offline
```
Signal EEG brut (TUAR)
      ↓
Preprocessing — notch, low-pass, FFT filter
      ↓
Méthodes directes spécifiques à chaque artefact   ← première passe
  (EOG regression, template subtraction, wavelet, ...)
      ↓
FastICA — décomposition sur artefacts résiduels
      ↓
[Phase 1 evaluation] — qualité de décomposition ICA (MIR, near-dipolar, scalp maps)
      ↓
ICLabel — classification des composantes résiduelles
      ↓
Suppression des composantes artefact + reconstruction
      ↓
[Phase 2 evaluation] — qualité du signal nettoyé (topomaps avant/après, métriques TBD)
      ↓
WebSocket → Flutter
```

### Mode online (temps réel)
```
Signal EEG brut — fenêtre par fenêtre
      ↓
Preprocessing — notch, low-pass, FFT filter
      ↓
Classifier TFLite (inspiré Kim & Keene, entraîné sur TUAR)
  → détecte le type d'artefact sur le signal brut (eyem, chew, musc, elpp, null)
  → indépendant de la méthode de removal
      ↓
ORICA — mise à jour W + suppression de la composante correspondante
      ↓
[Phase 2 evaluation] — signal nettoyé comparé à offline
      ↓
WebSocket → Flutter
```

**Remarque importante :** ICLabel n'est pas adapté au contexte ORICA (temps réel).
Il est remplacé par un classifier léger entraîné sur signal brut, compatible avec
n'importe quelle méthode de removal (ORICA, FastICA, méthode directe).

---

## Artifact classes

| Artifact | Distinctive signal characteristics | ICA approach | Direct method |
|----------|------------------------------------|--------------|---------------|
| Eye blink | Large amplitude, <4Hz, frontal (Fp1/Fp2) | FastICA/ORICA + ICLabel | EOG regression |
| Eye movement | Step-like, frontal, slow | FastICA/ORICA + ICLabel | EOG regression |
| EMG / muscle | >30Hz bursts, diffuse | FastICA/ORICA + ICLabel | Aggressive low-pass / wavelet |
| ECG | ~1Hz periodic, autocorrelation | FastICA/ORICA + ICLabel | Template subtraction |
| Line noise | Spectral peak at 50/60Hz | Not needed | Notch filter ← DONE |
| Movement | Very large amplitude, all channels | FastICA/ORICA + ICLabel | Amplitude thresholding |

---

## ICA algorithms

| Algorithm | Mode | Status |
|-----------|------|--------|
| **FastICA** | Offline (full signal in RAM) | Implemented |
| **ORICA** | Online (streaming, cooling FF + orthogonalization) | Implemented |

ICLabel (pre-trained, 7 classes) classifies components — no fine-tuning.

---

## Task list (from supervisor meeting, 2026-04-23)

| Priority | Task | Status |
|----------|------|--------|
| 1 | Translate Flutter app to English | TODO |
| 2 | Add FFT filter | TODO |
| 3 | Phase 1: ICA decomposition quality results (scalp maps, metrics) | TODO |
| 4 | Phase 2: clean signal visualization side-by-side | TODO |
| 5 | Artifact timeline visualization (when artifacts appear) | TODO |
| 6 | Topography maps before/after | TODO |
| 7 | Performance metrics display | TODO |
| 8 | Direct methods (start with eye blink EOG regression) | TODO |

---

## Planning milestones

| Date | Milestone |
|------|-----------|
| 09 Apr 2026 | Real-time EEG visualization ← DONE |
| 15 Apr 2026 | Preprocessing (notch + low-pass) ← DONE |
| 23 Apr 2026 | ORICA rewrite (proper cooling FF + orthogonalization) ← DONE |
| 06 May 2026 | FFT filter + Phase 1 results + dual display |
| 13 May 2026 | Direct methods (eye blink first) + Phase 2 topomaps |
| 14 May 2026 | Full comparison on TUAR + metrics |
| 09 Jun 2026 | Final presentation |

---

## Tech stack — `develop` branch

| Layer | Tool |
|-------|------|
| Mobile + Web UI | Flutter 3.x (Dart) — in English |
| Backend | FastAPI + uvicorn (Python) |
| Signal processing | MNE-Python, scipy, numpy |
| ICA | MNE built-in (FastICA), ORICA (custom) |
| IC classification (offline) | mne-icalabel (pre-trained, no fine-tuning) |
| Artifact detection (online) | Classifier TFLite custom (entraîné sur TUAR) |
| Real-time transport | WebSocket (JSON) |
| Dev server | Localhost (laptop) |
| Demo server | Railway or Render (free tier) |

---

## Instructions for Claude on this branch

- Always use FastAPI + WebSocket for server-client communication
- Flutter must NOT do any signal processing — only display
- Flutter UI must be in English
- Python server must NOT require GPU or cloud — runs on a standard laptop
- Use MNE for EEG I/O, preprocessing, and ICA
- ICLabel is pre-trained only — no fine-tuning
- Two-phase evaluation: Phase 1 = decomposition quality, Phase 2 = removal quality
- Metrics for both phases are TBD — do not hardcode specific metrics
- Direct methods run alongside ICA for comparison per artifact class
- First artifact to implement end-to-end: Eye blink
- Prefer simple solutions — this is a prototype, not production code
