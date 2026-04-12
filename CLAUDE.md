# Project Context — EEG Artifact Detection

## Project summary
Development of AI-based methods for elimination of artifacts in biomedical signals (EEG).

The core novelty is implementing a **real-time EEG pipeline deployable on mobile device**,
without relying on external computing clusters or cloud servers.

---

## Architecture decisions

- **No external server** — processing must run on-device or on a local machine (not cloud)
- **Cross-platform target** — Android, iOS, Web via Flutter
- **Frontend** — Flutter (Dart)
- **On-device ICA** — ICA algorithms implemented in Dart (ml_linalg) — this is the core technical challenge
- **On-device classifier** — ML-based component classification via TFLite (artifact vs brain signal)
- **Both ICA AND classifier must run on-device** — no server dependency

---

## Preprocessing module (à implémenter)

Ajouter `lib/preprocessing.dart` avec :
- `ButterworthLowPass` — filtre IIR zero-phase, bilinear transform
- `NotchFilter` — coupe-bande 50/60 Hz, IIR 2nd ordre zero-phase  
- `MorletWavelet` — convolution avec ondelette de Morlet complexe
- `EEGPreprocessingPipeline` — orchestrateur, applique les filtres canal par canal

Intégration dans `main.dart` :
- Panel sidebar avec toggles + sliders (cutoff, notch freq, wavelet freq)
- Appel pipeline.process(signal) avant SimpleICA.fit() dans _runICA()
- Message de statut "Preprocessing…" pendant le traitement

Contraintes : Dart pur, zéro package externe, compatible Flutter Web.

## Full pipeline (on-device)

```
Raw EEG signal
      ↓
Preprocessing        — filtering, re-referencing (Dart)
      ↓
ICA decomposition    — FastICA / ORICA / RobustICA (Dart / ml_linalg)
      ↓              — THIS is the heavy computation, the core challenge
Independent components
      ↓
Component classification  — artifact vs brain signal (TFLite model)
      ↓
Artifact suppression      — remove artifact components, reconstruct signal
      ↓
Clean EEG signal
```

---

## ICA algorithms to compare
- **FastICA** — stable reference, well documented, easiest to implement
- **ORICA** — online ICA, designed for real-time, more complex
- **RobustICA** — better on noisy signals
- Comparison done by varying window sizes to evaluate latency vs precision tradeoff

---

## Current phase (week of 27 March – 9 April 2026)

**Task: Implement simulation of real-time EEG data and detect artifact presence**

For this week's demo:
- Display a scrolling EEG signal in real-time on Flutter (mobile + web)
- Dataset loaded locally on the device (no network required)
- Sliding window simulation — no ICA or classification yet
- Just visualization of raw EEG signal

---

## Dataset
- Public EEG datasets (to be selected: EEGBCI, DEAP, CHB-MIT or similar)
- Format: likely .edf or .csv — must be pre-converted to JSON/CSV if .edf
- Simulated as real-time by sliding a window along the dataset

---

## Planning milestones
| Date | Milestone |
|------|-----------|
| 09 Apr 2026 | Real-time EEG simulation on mobile (current) |
| 22 Apr 2026 | Visualization tool to validate performance |
| 13 May 2026 | User-friendly computational framework |
| 06 May 2026 | Artifact-specific elimination pipeline |
| 14 May 2026 | Performance metrics computation |
| 09 Jun 2026 | Final oral presentation |

---

## Tech stack
| Layer | Tool |
|-------|------|
| Mobile/Web UI | Flutter 3.x (Dart) |
| ICA on-device | Dart — ml_linalg (core challenge) |
| ML classifier on-device | TFLite (runs after ICA, lightweight) |
| Dataset pre-processing | Python one-time — MNE, scikit-learn |
| Dataset format in app | CSV or JSON (pre-converted from .edf) |

---

## Key constraints
- No cloud server dependency (core project objective — explicitly stated in project brief)
- "Ideally" on-device — backend on local laptop acceptable if on-device proves infeasible
- ICA must run on-device — this is the novelty, not just the classifier
- Proto fonctionnel expected as deliverable (not a store-ready app)
- Existing libraries allowed (no from-scratch requirement)
