import asyncio
import json
import os
import re
import tempfile
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from scipy.signal import iirnotch, butter, filtfilt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # serveur actif
    # ── Arrêt propre ──────────────────────────────────────────────────────────
    if _raw is not None:
        try:
            _raw.close()
        except Exception:
            pass
    _cleanup_tmp()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── État global ───────────────────────────────────────────────────────────────
_raw:             mne.io.BaseRaw | None = None   # EDF online (lazy)
_data:            np.ndarray | None     = None   # données brutes (offline)
_data_filtered:   np.ndarray | None     = None   # données filtrées (offline)
_data_direct:     np.ndarray | None     = None   # après méthode directe (eye blink)
_data_clean:      np.ndarray | None     = None   # données nettoyées par ICA (offline)
_ica_labels:      list[str]             = []     # labels des composantes retirées
_ica_removed:     list[int]             = []     # indices des composantes retirées
_tmp_path:        str | None            = None
_channel_names:   list[str]             = []
_sampling_rate:   float                 = 256.0
_channel_min:     list[float]           = []
_channel_max:     list[float]           = []
_n_samples:       int                   = 0
_mode:            str                   = ""     # "online" | "offline"
_orica_global:    "ORICAProcessor | None" = None  # référence pour /clean

WINDOW_SEC     = 4.0
FRAME_INTERVAL = 0.1
SCAN_CHUNK_SEC = 30.0

# Classes ICLabel considérées comme artefacts (labels exacts retournés par mne-icalabel)
ARTIFACT_CLASSES = {"muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"}


# ── Filtres ───────────────────────────────────────────────────────────────────

def apply_filters(
    data: np.ndarray,
    sr: float,
    notch_enabled: bool,
    notch_freq: float,
    lowpass_enabled: bool,
    lowpass_cutoff: float,
    fft_enabled: bool = False,
    fft_low: float = 1.0,
    fft_high: float = 40.0,
) -> np.ndarray:
    """Apply notch, low-pass and/or FFT bandpass filter to data (n_channels, n_samples)."""
    min_samples = 20
    if data.shape[1] < min_samples:
        return data

    out = data.copy()

    if notch_enabled:
        b, a = iirnotch(notch_freq, Q=30, fs=sr)
        for i in range(out.shape[0]):
            out[i] = filtfilt(b, a, out[i])

    if lowpass_enabled and 0 < lowpass_cutoff < sr / 2:
        b, a = butter(4, lowpass_cutoff / (sr / 2), btype="low")
        for i in range(out.shape[0]):
            out[i] = filtfilt(b, a, out[i])

    if fft_enabled:
        from scipy.fft import rfft, irfft, rfftfreq
        n     = out.shape[1]
        freqs = rfftfreq(n, d=1.0 / sr)
        spec  = rfft(out, axis=1)
        mask  = (freqs < fft_low) | (freqs > fft_high)
        spec[:, mask] = 0.0
        out = irfft(spec, n=n, axis=1)

    return out


# ── ICA ───────────────────────────────────────────────────────────────────────

def _normalize_channel_name(ch: str, montage_lookup: dict[str, str]) -> str:
    """Try several naming conventions to find a match in the montage lookup."""
    candidates = [ch]
    # Strip trailing dots (PhysioNet: "Fp1.")
    stripped = ch.rstrip(".")
    candidates.append(stripped)
    # TUAR format: "EEG Fp1-REF", "EEG Fp1-LE", "EEG Fp1-AVG"
    for prefix in ("EEG ", "eeg "):
        if stripped.upper().startswith(prefix.upper()):
            core = stripped[len(prefix):]
            for suffix in ("-REF", "-LE", "-AVG", "-ref", "-le", "-avg"):
                if core.upper().endswith(suffix.upper()):
                    candidates.append(core[: -len(suffix)])
                    break
            candidates.append(core)
    for candidate in candidates:
        found = montage_lookup.get(candidate.lower())
        if found:
            return found
    return stripped


def _get_positioned_channels(
    data: np.ndarray,
    channel_names: list[str],
    sr: float,
) -> tuple[list[str], list[int], dict[str, int]]:
    """
    Résout les noms de canaux vers le montage standard_1005 et retourne
    (positioned_names, orig_indices, clean_to_orig).

    positioned_names : noms nettoyés des canaux avec position valide
    orig_indices     : indices dans channel_names de ces canaux
    clean_to_orig    : dict {nom_nettoyé: index_original}
    """
    montage        = mne.channels.make_standard_montage("standard_1005")
    montage_lookup = {ch.lower(): ch for ch in montage.ch_names}
    clean_names    = {}
    for ch in channel_names:
        clean_names[ch] = _normalize_channel_name(ch, montage_lookup)

    clean_to_orig = {v: i for i, v in enumerate(clean_names.values())}

    info = mne.create_info(
        ch_names=[clean_names[c] for c in channel_names],
        sfreq=sr,
        ch_types="eeg",
    )
    raw_tmp = mne.io.RawArray(data[:, :min(256, data.shape[1])].copy(), info, verbose=False)
    raw_tmp.set_montage(montage, on_missing="ignore", verbose=False)

    positioned = [
        ch["ch_name"] for ch in raw_tmp.info["chs"]
        if not np.any(np.isnan(ch["loc"][:3]))
    ]
    orig_indices = [clean_to_orig[ch] for ch in positioned if ch in clean_to_orig]

    return positioned, orig_indices, clean_to_orig


def compute_ica_offline(
    data: np.ndarray,
    channel_names: list[str],
    sr: float,
    n_components: int = 15,
) -> tuple[np.ndarray, list[int], list[str]]:
    """
    FastICA + ICLabel sur le dataset complet.
    Retourne (clean_data, removed_indices, removed_labels).
    Tourne dans un thread (run_in_executor) pour ne pas bloquer l'event loop.
    """
    positioned, orig_indices, clean_to_orig = _get_positioned_channels(
        data, channel_names, sr
    )

    n_comp = min(n_components, len(positioned) - 1)
    if n_comp < 2:
        raise ValueError(
            f"ICLabel nécessite des noms de canaux standard 10-20. "
            f"Seulement {len(positioned)}/{len(channel_names)} canaux reconnus. "
            f"Utilisez un EDF avec montage standard (10-05)."
        )

    montage = mne.channels.make_standard_montage("standard_1005")
    montage_lookup = {ch.lower(): ch for ch in montage.ch_names}
    clean_names = {}
    for ch in channel_names:
        clean_names[ch] = _normalize_channel_name(ch, montage_lookup)

    info = mne.create_info(
        ch_names=[clean_names[c] for c in channel_names],
        sfreq=sr,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    raw_pos = raw.copy().pick(positioned, verbose=False)

    # Preprocessing recommandé par ICLabel
    h_freq = min(100.0, sr / 2.0 - 1.0)
    raw_pos.filter(l_freq=1.0, h_freq=h_freq, verbose=False)
    raw_pos.set_eeg_reference("average", projection=False, verbose=False)

    # ICA — infomax extended recommandé par ICLabel
    ica = ICA(
        n_components=n_comp,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=42,
        verbose=False,
    )
    ica.fit(raw_pos, verbose=False)

    # Classification ICLabel
    pred   = label_components(raw_pos, ica, method="iclabel")
    labels = pred["labels"]

    # Exclure les composantes artefact
    exclude = [i for i, lbl in enumerate(labels) if lbl in ARTIFACT_CLASSES]

    # Reconstruction sur les canaux positionnés
    ica.exclude = exclude
    raw_clean = raw_pos.copy()
    ica.apply(raw_clean, verbose=False)
    clean_pos = raw_clean.get_data()

    # Reconstruire le signal complet (canaux sans position → inchangés)
    result = data.copy()
    for local_idx, ch_name in enumerate(positioned):
        orig_idx = clean_to_orig.get(ch_name)
        if orig_idx is not None:
            result[orig_idx] = clean_pos[local_idx]

    return result, exclude, [labels[i] for i in exclude]


# ── Eye blink removal — Zhang et al. (2017) ───────────────────────────────────

def _fp1_channel_idx(channel_names: list[str]) -> int | None:
    """Return index of Fp1 (or Fp2 / Fpz). Handles TUAR 'EEG Fp1-REF' style."""
    for target in ("fp1", "fp2", "fpz"):
        for i, ch in enumerate(channel_names):
            norm = (ch.lower()
                    .replace("eeg ", "").rstrip(".")
                    .replace("-ref", "").replace("-le", "").replace("-avg", "")
                    .strip())
            if norm == target:
                return i
    return None


def _argextrema(sig: np.ndarray, a: int, b: int, kind: str) -> int:
    a = max(0, a); b = min(len(sig) - 1, b)
    if a > b:
        return a
    fn = np.argmin if kind == "min" else np.argmax
    return a + int(fn(sig[a:b + 1]))


def _arg_nearest_zero(sig: np.ndarray, a: int, b: int) -> int:
    a = max(0, a); b = min(len(sig) - 1, b)
    if a > b:
        return a
    return a + int(np.argmin(np.abs(sig[a:b + 1])))


def _lstsq_line(sig: np.ndarray, p0: int, p1: int) -> tuple[float, float]:
    """
    Fit a line to sig over the inner 10–90 % of [p0, p1].
    Returns (slope, intercept) in global sample-index coordinates.
    """
    L  = max(1, p1 - p0)
    i0 = min(len(sig) - 1, p0 + max(1, int(0.10 * L)))
    i1 = min(len(sig) - 1, p0 + min(L - 1, int(0.90 * L)))
    i1 = max(i0, i1)
    x  = np.arange(i0, i1 + 1, dtype=np.float64)
    y  = sig[i0:i1 + 1].astype(np.float64)
    if len(x) < 2:
        m = (float(sig[min(p1, len(sig)-1)]) - float(sig[p0])) / L
        return m, float(sig[p0]) - m * p0
    xm, ym = x.mean(), y.mean()
    ss = float(np.sum((x - xm) ** 2))
    if ss < 1e-30:
        return 0.0, ym
    m = float(np.dot(x - xm, y - ym) / ss)
    return m, ym - m * xm


def _build_zhang_template(fp1: np.ndarray, kpts: list[int]) -> np.ndarray:
    """
    Piecewise linear template between key points (Z1, I1, I2, Z2, ...).
    Each segment: line fitted on the inner 10–90 % of the real signal.
    Adjacent lines are extrapolated to their intersection (breakpoint).
    Intersection zones are 3-point smoothed (midpoint outward) — §2.4.4.
    Returns array of length kpts[-1] - kpts[0].
    """
    t0 = kpts[0]
    T  = kpts[-1] - t0
    if T <= 0 or len(kpts) < 2:
        return np.zeros(max(0, T))

    lines: list[tuple[float, float]] = [
        _lstsq_line(fp1, kpts[j], kpts[j + 1])
        for j in range(len(kpts) - 1)
    ]

    # Breakpoints: intersection of each consecutive pair of lines
    bkpts = [kpts[0]]
    for j in range(len(lines) - 1):
        m1, b1 = lines[j];  m2, b2 = lines[j + 1]
        dm = m1 - m2
        if abs(dm) > 1e-15:
            tc = float(np.clip((b2 - b1) / dm, kpts[j], kpts[j + 2]))
        else:
            tc = float(kpts[j + 1])
        bkpts.append(int(round(tc)))
    bkpts.append(kpts[-1])

    tmpl = np.zeros(T)
    for j, (m, b) in enumerate(lines):
        s = max(t0, bkpts[j])
        e = min(t0 + T - 1, bkpts[j + 1])
        for t_g in range(s, e + 1):
            tl = t_g - t0
            if 0 <= tl < T:
                tmpl[tl] = m * t_g + b

    # 3-point smoothing from midpoint of first segment to midpoint of second
    for j, bp in enumerate(bkpts[1:-1]):
        bpl = bp - t0
        hw  = min((bp - bkpts[j]) // 2, (bkpts[j + 2] - bp) // 2)
        for step in range(hw):
            for tl in (bpl - step, bpl + step):
                if 1 <= tl < T - 1:
                    tmpl[tl] = (tmpl[tl - 1] + tmpl[tl] + tmpl[tl + 1]) / 3.0

    return tmpl


def detect_and_remove_eyeblinks_zhang2017(
    data: np.ndarray,
    channel_names: list[str],
    sr: float,
) -> tuple[np.ndarray, int, list[float]]:
    """
    Zhang et al. (2017) — single-channel physiology-based eye blink removal.

    1. Detect blinks via upward threshold crossings in Fp1.
    2. Per blink: locate inflection points I1–I4 and zero points Z1–Z4.
    3. Build a piecewise linear template (§2.4.4).
    4. Scale template to each channel by least squares (Gratton 1998) and subtract.

    Returns
    -------
    clean  : corrected data array (same shape as `data`)
    n      : number of blinks removed
    times  : blink peak times in seconds
    """
    fp1_idx = _fp1_channel_idx(channel_names)
    if fp1_idx is None:
        return data.copy(), 0, []

    fp1 = data[fp1_idx].astype(np.float64)
    n   = len(fp1)

    def ms(v_ms: float) -> int:
        return max(1, int(round(v_ms * sr / 1000.0)))

    # ── Detection threshold ──────────────────────────────────────────────────
    pos_vals = fp1[fp1 > 0]
    if len(pos_vals) < 10:
        return data.copy(), 0, []
    peak_ref   = np.percentile(pos_vals, 99)
    thresh_low = 0.25 * peak_ref   # 25–100 % window (§2.4)

    # ── Find upward threshold crossings → Start points S ────────────────────
    blink_starts: list[int] = []
    i = 0
    while i < n - 1:
        if fp1[i] < thresh_low <= fp1[i + 1]:
            blink_starts.append(i + 1)
            i += ms(200)   # minimum 200 ms between detections
        else:
            i += 1

    if not blink_starts:
        return data.copy(), 0, []

    result      = data.astype(np.float64).copy()
    blink_times: list[float] = []
    n_removed   = 0
    done_regions: list[tuple[int, int]] = []

    for S in blink_starts:
        # ── Inflection points ────────────────────────────────────────────────
        I2 = _argextrema(fp1, S, min(n - 1, S + ms(500)), "max")
        if fp1[I2] < thresh_low:
            continue
        I1 = _argextrema(fp1, max(0, S - ms(500)), S, "min")

        # I3: MIN, at least 12 samples (at 128 Hz) after I2
        i3_gap = max(ms(80), int(round(12 * sr / 128)))
        I3c    = _argextrema(fp1, I2 + i3_gap, min(n - 1, I2 + ms(500)), "min")
        has_I3 = (I3c > I2) and (fp1[I3c] < fp1[I2] * 0.5)
        I3     = I3c if has_I3 else None

        # I4: MAX, at least 13 samples after I3
        has_I4 = False; I4 = None
        if has_I3:
            i4_gap = max(ms(80), int(round(13 * sr / 128)))
            I4c    = _argextrema(fp1, I3 + i4_gap, min(n - 1, I3 + ms(500)), "max")
            has_I4 = (I4c > I3) and (fp1[I4c] > fp1[I3])
            I4     = I4c if has_I4 else None

        # ── Zero (baseline) points ───────────────────────────────────────────
        Z1 = _arg_nearest_zero(fp1, max(0, I1 - ms(500)), I1)
        gap12 = int(round(12 * sr / 128))
        gap14 = int(round(14 * sr / 128))

        if has_I3:
            Z2 = _arg_nearest_zero(fp1, I2 + gap12, I3)
        else:
            Z2 = _arg_nearest_zero(fp1, I2 + gap12, min(n - 1, I2 + ms(500)))

        if has_I3 and has_I4:
            Z3 = _arg_nearest_zero(fp1, I3 + gap14, I4)
            Z4 = _arg_nearest_zero(fp1, I4 + gap14, min(n - 1, I4 + ms(390)))
        elif has_I3:
            Z3 = _arg_nearest_zero(fp1, I3 + gap14, min(n - 1, I3 + ms(390)))
            Z4 = Z3
        else:
            Z3 = Z4 = Z2

        # ── Key points (temporal order, no duplicates) ───────────────────────
        kpts_raw = [Z1, I1, I2, Z2]
        if has_I3: kpts_raw += [I3, Z3]
        if has_I4: kpts_raw += [I4, Z4]
        kpts = sorted(set(kpts_raw))
        if len(kpts) < 2:
            continue

        # ── Template extent: L0 (−625 ms) … L8 (+625 ms) ────────────────────
        t_start = max(0, kpts[0] - ms(625))
        t_end   = min(n, kpts[-1] + ms(625))
        if (t_end - t_start) < ms(100):
            continue

        # Skip if this region overlaps a blink already removed
        if any(t_start < re and t_end > rs for rs, re in done_regions):
            continue

        # ── Build template ───────────────────────────────────────────────────
        tmpl_core = _build_zhang_template(fp1, kpts)
        T_seg     = t_end - t_start
        tmpl      = np.zeros(T_seg)
        offset    = kpts[0] - t_start
        core_end  = min(T_seg, offset + len(tmpl_core))
        tmpl[offset:core_end] = tmpl_core[:core_end - offset]

        # ── Subtract scaled template from every channel ──────────────────────
        denom = float(np.dot(tmpl, tmpl))
        if denom < 1e-30:
            continue

        for ch_i in range(result.shape[0]):
            ch_seg = result[ch_i, t_start:t_end]
            slope  = float(np.dot(tmpl, ch_seg)) / denom
            result[ch_i, t_start:t_end] -= slope * tmpl

        blink_times.append(float(I2) / sr)
        done_regions.append((t_start, t_end))
        n_removed += 1

    return result, n_removed, blink_times


class ORICAProcessor:
    """
    Online Recursive ICA (ORICA) — Hsu et al. (2012).
    Blanchiment fixe via sqrtm(cov). Mise à jour de W par gradient naturel
    avec facteur d'oubli cooling. Orthogonalisation après chaque bloc.
    Retourne les activations IC Y = W · sphere · X, fenêtre par fenêtre.
    """

    def __init__(
        self,
        X_init_pos: np.ndarray,   # (n_positioned, n_samples) — buffer initial
        orig_indices: list[int],  # indices dans le signal complet
        lambda_0: float = 0.995,
        gamma: float = 0.6,
        block_size: int = 8,
    ):
        self.orig_indices = np.asarray(orig_indices)
        self.lambda_0     = lambda_0
        self.gamma        = gamma
        self.block_size   = block_size
        self.t            = 1
        self.n_comp       = X_init_pos.shape[0]

        cov = np.cov(X_init_pos)
        sqrtm_cov   = scipy.linalg.sqrtm(cov).real
        self.sphere = 2.0 * np.linalg.inv(sqrtm_cov)
        self.W      = np.eye(self.n_comp, dtype=np.float64)

    def _cooling_ff(self, t_range: np.ndarray) -> np.ndarray:
        return self.lambda_0 / np.power(t_range, self.gamma)

    def process(self, X_full: np.ndarray) -> np.ndarray:
        """
        Met à jour W et retourne les activations IC Y = W·sphere·X_pos.
        Shape retournée : (n_comp, n_samples).
        """
        X_pos   = X_full[self.orig_indices, :]
        X_white = self.sphere @ X_pos

        _, n_samp = X_white.shape
        n_blocks  = max(1, n_samp // self.block_size)

        for bi in range(n_blocks):
            start = bi * n_samp // n_blocks
            end   = min(n_samp, (bi + 1) * n_samp // n_blocks)
            block = X_white[:, start:end]
            n_blk = block.shape[1]

            t_range = np.arange(self.t, self.t + n_blk, dtype=float)
            lam     = self._cooling_ff(t_range)

            Y = self.W @ block
            f = -2.0 * np.tanh(Y)

            fy_dot      = np.einsum("it,it->t", f, Y)
            q           = 1.0 + lam * (fy_dot - 1.0)
            lambda_prod = np.prod(1.0 / (1.0 - lam))
            correction  = (Y * (lam / q)) @ f.T
            self.W      = lambda_prod * (self.W - correction @ self.W)

            D_val, V_val = np.linalg.eig(self.W @ self.W.T)
            D_val        = np.abs(D_val.real)
            D_isqrt      = np.diag(1.0 / np.sqrt(D_val + 1e-12))
            self.W       = V_val.real @ D_isqrt @ V_val.real.T @ self.W

            self.t += n_blk

        return self.W @ X_white  # (n_comp, n_samp)


def _init_orica(
    raw: mne.io.BaseRaw,
    channel_names: list[str],
    sr: float,
    n_components: int,
) -> "ORICAProcessor":
    """
    Initialise ORICA avec W = identité — pas de bootstrap FastICA.
    W convergera progressivement via les fenêtres du streaming.
    ICLabel sera lancé pour la première fois à la 50ème fenêtre (~5s).
    Tourne dans run_in_executor (non-bloquant).
    """
    # Lire juste quelques samples pour obtenir les indices de canaux positionnés
    probe_samples = min(int(sr * 2), int(raw.n_times))
    X_probe = raw.get_data(start=0, stop=probe_samples)

    _, orig_indices, _ = _get_positioned_channels(X_probe, channel_names, sr)

    if not orig_indices:
        raise ValueError("Aucun canal positionné trouvé pour ORICA.")

    X_init_pos = X_probe[np.asarray(orig_indices), :]

    return ORICAProcessor(X_init_pos, orig_indices=orig_indices)


def _cleanup_tmp():
    global _tmp_path
    if _tmp_path and os.path.exists(_tmp_path):
        try:
            os.unlink(_tmp_path)
        except OSError:
            pass
    _tmp_path = None


def _compute_minmax_lazy(raw: mne.io.BaseRaw) -> tuple[list[float], list[float]]:
    n_ch      = len(raw.ch_names)
    n_samples = int(raw.n_times)
    chunk     = int(raw.info["sfreq"] * SCAN_CHUNK_SEC)

    ch_min = np.full(n_ch,  np.inf)
    ch_max = np.full(n_ch, -np.inf)

    for start in range(0, n_samples, chunk):
        stop  = min(start + chunk, n_samples)
        block = raw.get_data(start=start, stop=stop)
        ch_min = np.minimum(ch_min, block.min(axis=1))
        ch_max = np.maximum(ch_max, block.max(axis=1))

    return ch_min.tolist(), ch_max.tolist()


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    mode: str = Form("offline"),
):
    global _raw, _data, _data_filtered, _data_clean, _tmp_path
    global _channel_names, _sampling_rate, _channel_min, _channel_max, _n_samples, _mode
    global _ica_labels, _ica_removed

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".edf", ".csv"):
        return {"status": "error", "message": f"Format non supporté : {suffix}"}

    if suffix == ".csv":
        mode = "offline"

    content = await file.read()

    try:
        if _raw is not None:
            _raw.close()
            _raw = None
        _cleanup_tmp()

        # Réinitialise l'état à chaque nouveau fichier
        _data_direct = None
        _data_clean  = None
        _ica_labels  = []
        _ica_removed = []

        if suffix == ".edf" and mode == "online":
            tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
            tmp.write(content)
            tmp.close()

            raw = mne.io.read_raw_edf(tmp.name, preload=False, verbose=False)

            _raw           = raw
            _data          = None
            _tmp_path      = tmp.name
            _channel_names = list(raw.ch_names)
            _sampling_rate = float(raw.info["sfreq"])
            _n_samples     = int(raw.n_times)
            _mode          = "online"
            _channel_min, _channel_max = _compute_minmax_lazy(raw)

        elif suffix == ".edf" and mode == "offline":
            tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
            tmp.write(content)
            tmp.close()
            try:
                raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
            finally:
                os.unlink(tmp.name)

            data           = raw.get_data().astype(np.float64)
            _raw           = None
            _data          = data
            _data_filtered = data.copy()
            _channel_names = list(raw.ch_names)
            _sampling_rate = float(raw.info["sfreq"])
            _n_samples     = data.shape[1]
            _mode          = "offline"
            _channel_min   = data.min(axis=1).tolist()
            _channel_max   = data.max(axis=1).tolist()

        else:
            text  = content.decode("utf-8", errors="replace")
            lines = text.splitlines()

            sr = 256.0
            for line in lines[:10]:
                m = re.search(r"sampling_rate\s*=\s*([\d.]+)", line)
                if m:
                    sr = float(m.group(1))
                    break

            clean = "\n".join(l for l in lines if not l.startswith("#"))
            df    = pd.read_csv(StringIO(clean))
            df    = df.select_dtypes(include=[np.number])

            data           = df.to_numpy().T.astype(np.float64)
            _raw           = None
            _data          = data
            _data_filtered = data.copy()
            _channel_names = list(df.columns)
            _sampling_rate = sr
            _n_samples     = data.shape[1]
            _mode          = "offline"
            _channel_min   = data.min(axis=1).tolist()
            _channel_max   = data.max(axis=1).tolist()

        return {
            "status":        "ok",
            "mode":          _mode,
            "channels":      len(_channel_names),
            "sampling_rate": _sampling_rate,
            "duration_sec":  round(_n_samples / _sampling_rate, 2),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Clean signal export ───────────────────────────────────────────────────────



# ── Stream WebSocket ──────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if _n_samples == 0:
        await websocket.send_text(json.dumps({
            "type":    "error",
            "message": "Aucun fichier chargé.",
        }))
        await websocket.close()
        return

    window_size = int(_sampling_rate * WINDOW_SEC)
    step        = max(1, int(_sampling_rate * FRAME_INTERVAL))

    await websocket.send_text(json.dumps({
        "type":          "meta",
        "channels":      _channel_names,
        "sampling_rate": _sampling_rate,
        "total_samples": _n_samples,
        "channel_min":   _channel_min,
        "channel_max":   _channel_max,
        "mode":          _mode,
    }))

    paused          = False
    pos             = 0
    notch_enabled   = False
    notch_freq      = 50.0
    lowpass_enabled = False
    lowpass_cutoff  = 40.0
    fft_enabled     = False
    fft_low         = 1.0
    fft_high        = 40.0
    ica_enabled     = False
    orica: ORICAProcessor | None = None
    _ica_removed    = []
    _ica_labels     = []

    async def receive_commands():
        nonlocal paused, pos, window_size, ica_enabled, orica
        nonlocal notch_enabled, notch_freq, lowpass_enabled, lowpass_cutoff
        nonlocal fft_enabled, fft_low, fft_high
        global _data_filtered, _data_direct, _data_clean, _ica_labels, _ica_removed
        try:
            async for raw_msg in websocket.iter_text():
                cmd = json.loads(raw_msg)
                t   = cmd.get("type")

                if t == "pause":
                    paused = True

                elif t == "resume":
                    paused = False

                elif t == "seek" and _mode == "offline":
                    seek = int(cmd.get("position", 0))
                    pos  = max(0, min(seek, _n_samples - window_size))

                elif t == "set_window":
                    secs        = float(cmd.get("seconds", WINDOW_SEC))
                    window_size = max(1, int(_sampling_rate * secs))
                    pos = min(pos, max(0, _n_samples - window_size))

                elif t == "set_filters":
                    notch_enabled   = bool(cmd.get("notch_enabled",   False))
                    notch_freq      = float(cmd.get("notch_freq",     50.0))
                    lowpass_enabled = bool(cmd.get("lowpass_enabled", False))
                    lowpass_cutoff  = float(cmd.get("lowpass_cutoff", 40.0))
                    fft_enabled     = bool(cmd.get("fft_enabled",     False))
                    fft_low         = float(cmd.get("fft_low",        1.0))
                    fft_high        = float(cmd.get("fft_high",       40.0))
                    if _mode == "offline" and _data is not None:
                        _data_filtered = apply_filters(
                            _data, _sampling_rate,
                            notch_enabled, notch_freq,
                            lowpass_enabled, lowpass_cutoff,
                            fft_enabled, fft_low, fft_high,
                        )
                        # Réinitialise méthode directe et ICA si les filtres changent
                        _data_direct = None
                        if ica_enabled:
                            _data_clean  = None
                            _ica_labels  = []
                            _ica_removed = []

                elif t == "set_ica":
                    enabled = bool(cmd.get("enabled", False))
                    n_comp  = int(cmd.get("n_components", 15))
                    ica_enabled = enabled

                    if enabled and _mode == "offline" and _data_filtered is not None:
                        # ── Mode offline : FastICA + ICLabel sur dataset complet ──
                        await websocket.send_text(json.dumps({
                            "type":   "ica_status",
                            "status": "computing",
                        }))
                        try:
                            # ICA s'applique sur la sortie de la méthode directe si disponible
                            src  = _data_direct if _data_direct is not None else _data_filtered
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None,
                                compute_ica_offline,
                                src, _channel_names, _sampling_rate, n_comp,
                            )
                            _data_clean, _ica_removed, _ica_labels = result
                            await websocket.send_text(json.dumps({
                                "type":               "ica_status",
                                "status":             "done",
                                "removed_components": _ica_removed,
                                "labels":             _ica_labels,
                            }))
                        except Exception as e:
                            ica_enabled = False
                            await websocket.send_text(json.dumps({
                                "type":    "ica_status",
                                "status":  "error",
                                "message": str(e),
                            }))

                    elif enabled and _mode == "online" and _raw is not None:
                        await websocket.send_text(json.dumps({
                            "type":   "ica_status",
                            "status": "computing",
                        }))
                        try:
                            loop  = asyncio.get_event_loop()
                            orica = await loop.run_in_executor(
                                None, _init_orica,
                                _raw, _channel_names, _sampling_rate, n_comp,
                            )
                            global _orica_global
                            _orica_global = orica
                            await websocket.send_text(json.dumps({
                                "type":   "ica_status",
                                "status": "ready",
                                "n_comp": orica.n_comp,
                            }))
                        except Exception as e:
                            ica_enabled = False
                            orica       = None
                            await websocket.send_text(json.dumps({
                                "type":    "ica_status",
                                "status":  "error",
                                "message": str(e),
                            }))

                    elif not enabled:
                        orica        = None
                        _data_clean  = None
                        _ica_labels  = []
                        _ica_removed = []

                elif t == "set_direct_method" and _mode == "offline":
                    method  = cmd.get("method", "eyeblink")
                    enabled = bool(cmd.get("enabled", False))

                    if enabled and method == "eyeblink" and _data_filtered is not None:
                        await websocket.send_text(json.dumps({
                            "type":   "direct_method_status",
                            "status": "computing",
                            "method": method,
                        }))
                        try:
                            loop = asyncio.get_event_loop()
                            clean, n_blinks, times = await loop.run_in_executor(
                                None,
                                detect_and_remove_eyeblinks_zhang2017,
                                _data_filtered, _channel_names, _sampling_rate,
                            )
                            _data_direct = clean
                            # Si ICA déjà calculée, l'invalider (elle doit tourner sur le nouveau signal)
                            _data_clean  = None
                            _ica_labels  = []
                            _ica_removed = []
                            await websocket.send_text(json.dumps({
                                "type":          "direct_method_status",
                                "status":        "done",
                                "method":        method,
                                "effective":     n_blinks > 0,
                                "n_blinks":      n_blinks,
                                "blink_times_s": times,
                            }))
                        except Exception as e:
                            await websocket.send_text(json.dumps({
                                "type":    "direct_method_status",
                                "status":  "error",
                                "method":  method,
                                "message": str(e),
                            }))
                    elif not enabled:
                        _data_direct = None
                        _data_clean  = None
                        _ica_labels  = []
                        _ica_removed = []

        except Exception:
            pass

    recv_task = asyncio.create_task(receive_commands())

    try:
        while pos + window_size <= _n_samples:
            if not paused:
                if _mode == "online" and _raw is not None:
                    # Online : lecture de la fenêtre complète
                    window = _raw.get_data(start=pos, stop=pos + window_size)
                    window = apply_filters(
                        window, _sampling_rate,
                        notch_enabled, notch_freq,
                        lowpass_enabled, lowpass_cutoff,
                        fft_enabled, fft_low, fft_high,
                    )

                    if ica_enabled and orica is not None:
                        # ORICA : met à jour W, retourne activations IC
                        ic_act = orica.process(window)
                        await websocket.send_text(json.dumps({
                            "type":           "ic_window",
                            "start":          pos,
                            "raw":            window.tolist(),
                            "ic_activations": ic_act.tolist(),
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type":  "window",
                            "start": pos,
                            "data":  window.tolist(),
                        }))

                else:
                    # Offline : données pré-filtrées
                    src     = _data_filtered if _data_filtered is not None else _data
                    raw_win = src[:, pos : pos + window_size]

                    if ica_enabled and _data_clean is not None:
                        # ICA prête → raw vs ICA clean
                        clean_win = _data_clean[:, pos : pos + window_size]
                        await websocket.send_text(json.dumps({
                            "type":               "ica_window",
                            "start":              pos,
                            "raw":                raw_win.tolist(),
                            "clean":              clean_win.tolist(),
                            "removed_components": _ica_removed,
                            "labels":             _ica_labels,
                        }))
                    elif _data_direct is not None:
                        # Eye blink removal seul → raw vs direct clean
                        direct_win = _data_direct[:, pos : pos + window_size]
                        await websocket.send_text(json.dumps({
                            "type":  "direct_window",
                            "start": pos,
                            "raw":   raw_win.tolist(),
                            "clean": direct_win.tolist(),
                        }))
                    else:
                        # Aucun traitement → fenêtre brute
                        await websocket.send_text(json.dumps({
                            "type":  "window",
                            "start": pos,
                            "data":  raw_win.tolist(),
                        }))

                pos += step
            await asyncio.sleep(FRAME_INTERVAL)

        await websocket.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()
