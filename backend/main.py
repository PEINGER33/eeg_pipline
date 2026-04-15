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
) -> np.ndarray:
    """Applique notch et/ou low-pass à data (n_channels, n_samples)."""
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

    return out


# ── ICA ───────────────────────────────────────────────────────────────────────

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
        stripped = ch.rstrip(".")
        clean_names[ch] = montage_lookup.get(stripped.lower(), stripped)

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
        stripped = ch.rstrip(".")
        clean_names[ch] = montage_lookup.get(stripped.lower(), stripped)

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


class ORICAProcessor:
    """
    Online Recursive ICA (ORICA) pour le mode streaming.

    Initialisation : PCA whitening ajusté sur un buffer de 10 s.
    Les composantes à exclure sont déterminées une fois via ICLabel sur ce buffer.
    Chaque appel à process() met à jour W (gradient naturel) et reconstruit
    le signal sans les composantes artefact.

    Référence : Hsu et al. (2012) "Real-time adaptive EEG source separation
    using online recursive independent component analysis."
    """

    def __init__(
        self,
        X_init_pos: np.ndarray,   # (n_positioned, n_samples) — buffer initial
        n_components: int,
        exclude: list[int],
        labels: list[str],
        orig_indices: list[int],  # indices dans le signal complet
        lr: float = 0.005,
    ):
        from sklearn.decomposition import PCA

        self.exclude      = set(exclude)
        self.labels       = labels
        self.orig_indices = np.asarray(orig_indices)
        self.n_comp       = n_components
        self.lr           = lr
        self.t            = 0

        # Ajuster PCA sur le buffer initial (whitening)
        self._pca = PCA(n_components=n_components, whiten=True)
        self._pca.fit(X_init_pos.T)  # fit attend (n_samples, n_features)

        # W initialisé à l'identité — convergera vers FastICA en ligne
        self.W = np.eye(n_components, dtype=np.float64)

    def process(self, X_full: np.ndarray) -> np.ndarray:
        """
        X_full : (n_channels_total, n_samples)
        Retourne un signal de même forme avec les composantes artefact supprimées.
        """
        n_samp = X_full.shape[1]

        # Extraire les canaux positionnés
        X_pos = X_full[self.orig_indices, :]                    # (n_pos, n_samp)

        # Blanchiment PCA
        X_white = self._pca.transform(X_pos.T).T               # (n_comp, n_samp)

        # Mise à jour ORICA — gradient naturel
        Y = self.W @ X_white                                    # (n_comp, n_samp)
        f_Y = np.tanh(Y)
        lr_t = self.lr / (1.0 + self.t / 100_000.0)
        self.W += lr_t * (np.eye(self.n_comp) - (f_Y @ Y.T) / n_samp) @ self.W
        self.t += n_samp

        # Mettre à zéro les composantes artefact
        Y_clean = Y.copy()
        for i in self.exclude:
            if i < Y_clean.shape[0]:
                Y_clean[i] = 0.0

        # Reconstruction : W⁻¹ @ Y_clean → espace capteurs → dé-blanchiment
        W_inv = np.linalg.pinv(self.W)
        X_clean_pos = self._pca.inverse_transform(
            (W_inv @ Y_clean).T
        ).T                                                      # (n_pos, n_samp)

        # Injecter dans le signal complet
        result = X_full.copy()
        result[self.orig_indices, :] = X_clean_pos

        return result


def _init_orica(
    raw: mne.io.BaseRaw,
    channel_names: list[str],
    sr: float,
    n_components: int,
) -> "ORICAProcessor":
    """
    Lit un buffer de 10 s depuis `raw`, lance ICLabel pour obtenir les labels,
    et retourne un ORICAProcessor initialisé.
    Tourne dans run_in_executor (non-bloquant).
    """
    buf_samples = min(int(sr * 10), int(raw.n_times))
    X_buf = raw.get_data(start=0, stop=buf_samples)

    # Lancer ICLabel sur le buffer pour identifier les composantes artefact
    _, exclude, labels = compute_ica_offline(X_buf, channel_names, sr, n_components)

    # Obtenir les indices de canaux positionnés (même logique que compute_ica_offline)
    _, orig_indices, _ = _get_positioned_channels(X_buf, channel_names, sr)

    if not orig_indices:
        raise ValueError("Aucun canal positionné trouvé pour ORICA.")

    n_comp = min(n_components, len(orig_indices) - 1)
    X_init_pos = X_buf[orig_indices, :]

    return ORICAProcessor(X_init_pos, n_comp, exclude, labels, orig_indices)


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

        # Réinitialise l'état ICA à chaque nouveau fichier
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
    ica_enabled     = False
    orica: ORICAProcessor | None = None   # actif en mode online uniquement

    async def receive_commands():
        nonlocal paused, pos, window_size, ica_enabled, orica
        nonlocal notch_enabled, notch_freq, lowpass_enabled, lowpass_cutoff
        global _data_filtered, _data_clean, _ica_labels, _ica_removed
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
                    if _mode == "offline" and _data is not None:
                        _data_filtered = apply_filters(
                            _data, _sampling_rate,
                            notch_enabled, notch_freq,
                            lowpass_enabled, lowpass_cutoff,
                        )
                        # Si ICA déjà calculée, la recalculer sur les données re-filtrées
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
                            src  = _data_filtered
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
                        # ── Mode online : initialiser ORICA via ICLabel sur 10 s ──
                        await websocket.send_text(json.dumps({
                            "type":   "ica_status",
                            "status": "computing",
                        }))
                        try:
                            loop = asyncio.get_event_loop()
                            orica = await loop.run_in_executor(
                                None,
                                _init_orica,
                                _raw, _channel_names, _sampling_rate, n_comp,
                            )
                            _ica_removed = sorted(orica.exclude)
                            _ica_labels  = orica.labels
                            await websocket.send_text(json.dumps({
                                "type":               "ica_status",
                                "status":             "done",
                                "removed_components": _ica_removed,
                                "labels":             _ica_labels,
                            }))
                        except Exception as e:
                            ica_enabled = False
                            orica = None
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

        except Exception:
            pass

    recv_task = asyncio.create_task(receive_commands())

    try:
        while pos + window_size <= _n_samples:
            if not paused:
                if _mode == "online" and _raw is not None:
                    # Online : lecture lazy + filtre par fenêtre
                    window = _raw.get_data(start=pos, stop=pos + window_size)
                    window = apply_filters(
                        window, _sampling_rate,
                        notch_enabled, notch_freq,
                        lowpass_enabled, lowpass_cutoff,
                    )

                    if ica_enabled and orica is not None:
                        # ORICA : mise à jour W + reconstruction
                        clean_win = orica.process(window)
                        await websocket.send_text(json.dumps({
                            "type":               "ica_window",
                            "start":              pos,
                            "raw":                window.tolist(),
                            "clean":              clean_win.tolist(),
                            "removed_components": _ica_removed,
                            "labels":             _ica_labels,
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type":  "window",
                            "start": pos,
                            "data":  window.tolist(),
                        }))

                else:
                    # Offline : données pré-filtrées
                    src = _data_filtered if _data_filtered is not None else _data

                    if ica_enabled and _data_clean is not None:
                        # ICA prête → envoie les deux fenêtres
                        raw_win   = src[:, pos : pos + window_size]
                        clean_win = _data_clean[:, pos : pos + window_size]
                        await websocket.send_text(json.dumps({
                            "type":               "ica_window",
                            "start":              pos,
                            "raw":                raw_win.tolist(),
                            "clean":              clean_win.tolist(),
                            "removed_components": _ica_removed,
                            "labels":             _ica_labels,
                        }))
                    else:
                        # Pas d'ICA (ou en cours de calcul) → fenêtre normale
                        window = src[:, pos : pos + window_size]
                        await websocket.send_text(json.dumps({
                            "type":  "window",
                            "start": pos,
                            "data":  window.tolist(),
                        }))

                pos += step
            await asyncio.sleep(FRAME_INTERVAL)

        await websocket.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()
