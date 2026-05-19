# Journal de développement — EEG Artifact Detection Pipeline

**Projet :** Développement de méthodes basées IA pour l'élimination d'artefacts dans les signaux biomédicaux (EEG)
**Objectif central :** Pipeline temps réel déployable sur mobile, sans serveur externe

---

## Architecture cible

```
Signal EEG brut
      ↓
Préprocessing        — filtrage, re-référencement (Dart)
      ↓
Décomposition ICA    — FastICA / ORICA / RobustICA (Dart / ml_linalg)
      ↓
Classification       — artefact vs signal cérébral (TFLite on-device)
      ↓
Suppression          — reconstruction du signal sans les composantes artefacts
      ↓
Signal EEG propre
```

**Contrainte fondamentale :** tout doit tourner on-device (Android / iOS / Web via Flutter) — aucune dépendance à un serveur cloud.

---

## Jalons du projet

| Date | Jalon |
|------|-------|
| 09 avr. 2026 | Simulation EEG temps réel sur mobile ← *en cours* |
| 22 avr. 2026 | Outil de visualisation pour valider les performances |
| 06 mai 2026 | Pipeline d'élimination spécifique aux artefacts |
| 13 mai 2026 | Framework computationnel accessible |
| 14 mai 2026 | Calcul des métriques de performance |
| 09 jun. 2026 | Présentation orale finale |

---

## Entrées du journal

---

### [08 avr. 2026] — Problème : import de fichiers EDF volumineux

#### Contexte
L'application Flutter charge des fichiers EDF (format standard EEG) pour les visualiser en temps réel via une fenêtre glissante. Lors de l'import de fichiers EDF volumineux (enregistrements longs, nombreux canaux), l'application se ralentit significativement voire se bloque.

#### Cause identifiée
Après analyse du code, trois problèmes ont été identifiés :

**1. Chargement intégral en RAM à l'import** (`main.dart`, import FilePicker)
```dart
// Problème : withData: true charge TOUT le fichier en mémoire d'un coup
withData: true,
```
Le file_picker avec `withData: true` lit l'intégralité du fichier EDF en RAM au moment de l'import, sans aucune lecture différée.

**2. Stockage inefficace en `List<double>`** (`eeg_signal.dart`, `EEGEDFParser`)
```dart
// Problème : chaque sample est stocké en double (8 octets)
// alors que le format EDF utilise des int16 (2 octets) → 4× plus lourd
final data = List.generate(nValid, (j) => List<double>.filled(totSamples[j], 0.0));
```
Un fichier EDF d'1h à 256 Hz avec 64 canaux représente environ **450 Mo en RAM** avec ce format.

**3. Calcul min/max recalculé à chaque frame** (`main.dart`, `_SignalPreview`)
```dart
// Problème : parcourt TOUT le canal (potentiellement des millions de samples)
// à chaque rebuild de l'UI (60 fps pendant la simulation)
final yMin = full.reduce(math.min);
final yMax = full.reduce(math.max);
```

#### Approche suggérée
Utilisation du **memory mapping (mmap)** : au lieu de charger tout le fichier en RAM, le système OS mappe le fichier dans l'espace d'adressage virtuel et ne lit que les pages demandées à la volée.

#### Analyse de faisabilité dans le contexte Flutter

| Aspect | Réalité |
|--------|---------|
| Dart natif | Pas d'API `mmap` directe — nécessiterait du FFI (appels C natifs) |
| Cible Web | Impossible — pas d'accès au système de fichiers |
| Complexité | FFI multiplateforme = lourd à maintenir |

#### Solution retenue
Adopter le **principe** du memory mapping (lecture partielle à la demande), implémenté directement en Dart pur via `RandomAccessFile` — sans FFI, sans dépendance externe, cross-platform natif.

Architecture introduite :

```
EEGDataSource (interface abstraite)
  ├── InMemoryEEGDataSource   ← CSV, ou EDF sur Web (chargement complet inévitable)
  └── EDFChunkedReader        ← EDF sur natif (Android/iOS/Desktop) — lecture lazy
```

**`EDFChunkedReader`** (`edf_chunked_reader_native.dart`) :
- À l'ouverture : lit uniquement le **header EDF** + fait un **scan séquentiel** pour calculer les min/max par canal (une seule passe, données non gardées en RAM)
- À chaque tick de simulation : `getWindow(start, count)` positionne le curseur avec `setPosition()` et lit **uniquement les data records** couvrant la fenêtre affichée
- Les données ne sont jamais toutes en RAM simultanément

```dart
await _raf.setPosition(headerBytes + rec * bytesPerRecord); // seek direct
await _raf.readInto(recBuf);                                // lecture du record
```

**Export conditionnel** (`edf_chunked_reader.dart`) :
```dart
export 'edf_chunked_reader_native.dart'
    if (dart.library.html) 'edf_chunked_reader_web.dart';
```
Sur Web, un stub est utilisé — l'EDF est chargé en RAM via le parser existant.

**Gain obtenu :**
- Min/max calculés une seule fois à l'ouverture → plus recalculés à chaque frame (60 fps)
- Seule la fenêtre de 4 secondes est en RAM à tout moment (au lieu de tout le signal)
- Aucun FFI, aucune dépendance supplémentaire

#### Fichiers modifiés / créés
| Fichier | Rôle |
|---------|------|
| `lib/eeg_signal.dart` | Ajout interface `EEGDataSource` + `InMemoryEEGDataSource` |
| `lib/edf_chunked_reader_native.dart` | Lecteur lazy EDF via `RandomAccessFile` |
| `lib/edf_chunked_reader_web.dart` | Stub web (EDF non lazy sur web) |
| `lib/edf_chunked_reader.dart` | Export conditionnel natif/web |
| `lib/main.dart` | Migré vers `EEGDataSource`, chargement async de fenêtre |

#### Statut
- [x] Interface `EEGDataSource` et `InMemoryEEGDataSource`
- [x] `EDFChunkedReader` avec lecture lazy par `RandomAccessFile`
- [x] Export conditionnel natif / web
- [x] `main.dart` migré — min/max mis en cache, fenêtre chargée async

---

---

### [09 avr. 2026] — Implémentation du module de prétraitement (Preprocessing)

#### Contexte
Le pipeline complet prévoit une étape de prétraitement avant l'ICA : filtrage passe-bas, suppression du bruit secteur (50/60 Hz), et analyse temps-fréquence par ondelette. Cette étape améliore la qualité des composantes ICA en réduisant le bruit avant décomposition.

#### Choix techniques

**Contrainte principale :** Dart pur, zéro package externe, compatible Flutter Web.
Impossible d'utiliser `scipy.signal` ou `numpy` — tout est réimplémenté manuellement.

**Algorithmes implémentés (`lib/preprocessing.dart`) :**

| Composant | Méthode | Détail |
|-----------|---------|--------|
| `ButterworthLowPass` | IIR 2nd ordre, transformation bilinéaire (Tustin) | Zero-phase via filtfilt (avant + arrière) |
| `NotchFilter` | Bi-quad IIR coupe-bande | Zero-phase, Q=30 (bande étroite), 50 ou 60 Hz |
| `MorletWavelet` | Convolution complexe | Retourne l'amplitude instantanée dans la bande cible |
| `EEGPreprocessingPipeline` | Orchestrateur | Applique les filtres canal par canal |

**Zero-phase (filtfilt) :**
Chaque filtre IIR est appliqué en avant puis en arrière sur le signal. Cela annule le déphasage et double l'ordre effectif du filtre — critique pour l'EEG où le décalage temporel fausserait la localisation des artefacts.

```dart
List<double> _filtfilt(List<double> x, List<double> b, List<double> a) {
  final forward  = _iirFilter(x, b, a);
  final reversed = forward.reversed.toList();
  final backward = _iirFilter(reversed, b, a);
  return backward.reversed.toList();
}
```

#### Intégration dans l'UI (`main.dart`)

- **Bouton `tune`** dans l'AppBar : ouvre/ferme le panel latéral, change de couleur si un filtre est actif
- **Panel sidebar** (`_PreprocessingPanel`) : toggles + sliders pour chaque filtre
  - Passe-bas : activer/désactiver, slider cutoff 5–100 Hz
  - Notch : activer/désactiver, choix 50 Hz / 60 Hz
  - Morlet : activer/désactiver, slider fréquence centrale 1–40 Hz
- **Appel avant ICA** dans `_runICA()` : si au moins un filtre est actif, le pipeline est appliqué et le statut affiche "Preprocessing…" avant de passer à l'ICA

#### Fichiers créés / modifiés
| Fichier | Modification |
|---------|-------------|
| `lib/preprocessing.dart` | Nouveau — implémentation complète des filtres |
| `lib/main.dart` | Import, état `_ppParams`/`_ppPanelOpen`, appel pipeline, panel UI |

---

### [13 avr. 2026] — Transition vers architecture backend Python + frontend Flutter

#### Contexte
La version initiale (branche `master`) implémentait tout en Dart on-device : parsing EDF, filtres IIR, FastICA. Cette approche posait un problème fondamental : réimplémenter from scratch des algorithmes matures (filtres, ICA) dans un langage peu adapté au traitement numérique, alors que Python dispose d'écosystèmes éprouvés (MNE, NumPy, scikit-learn).

Décision : migrer le traitement de données vers un backend Python local, Flutter restant exclusivement pour l'affichage. Cette stratégie reste conforme au cahier des charges ("backend sur laptop local acceptable si on-device s'avère infaisable").

#### Nouvelle architecture

```
Fichier EDF
      ↓
Flutter — file picker → HTTP POST
      ↓
Backend Python (FastAPI)
  - Lecture EDF via MNE (mne.io.read_raw_edf)
  - Préprocessing, ICA — à venir
  - Simulation temps réel (fenêtre glissante)
      ↓
WebSocket (ws://localhost:8000/ws)
      ↓
Flutter — affichage signal scrollant (inchangé visuellement)
```

#### Ce qui a changé

**Supprimé de `lib/` :**
- `ica.dart` — FastICA réimplémenté en Dart (remplacé par scikit-learn côté Python)
- `preprocessing.dart` — filtres IIR Dart (remplacé par MNE/SciPy)
- `eeg_signal.dart`, `edf_chunked_reader*.dart` — parsing EDF Dart (remplacé par `mne.io.read_raw_edf`)

Ces fichiers sont conservés dans `archive_dart_all/` pour référence.

**Créé `backend/` :**
| Fichier | Rôle |
|---------|------|
| `main.py` | FastAPI — endpoint `/upload` (HTTP POST EDF) + `/ws` (WebSocket stream) |
| `requirements.txt` | fastapi, uvicorn, mne, numpy, python-multipart |

**`lib/main.dart` — côté Flutter :**
- Suppression de toute logique de traitement de signal
- Ajout upload HTTP multipart (`file_picker` + `http`)
- Connexion WebSocket automatique après upload réussi
- Réception messages `meta` (infos canaux) et `window` (données fenêtre)
- Visuel identique : même `_SignalPreview`, `_ChannelRow`, `_MiniPlot`

**Protocole WebSocket (JSON) :**
```json
// Backend → Flutter
{ "type": "meta", "channels": [...], "sampling_rate": 256, "total_samples": 65536, "channel_min": [...], "channel_max": [...] }
{ "type": "window", "start": 1024, "data": [[...canal 0...], [...canal 1...], ...] }
{ "type": "done" }
```

#### Lancement
```bash
# Terminal 1 — backend
cd backend && source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Flutter
flutter run -d linux   # ou -d chrome
```

#### Statut
- [x] Backend FastAPI opérationnel
- [x] Upload EDF via Flutter → parsing MNE
- [x] Stream WebSocket → affichage signal réel
- [ ] Preprocessing Python (MNE) — à venir
- [ ] ICA Python (scikit-learn / MNE) — à venir

---

### [15 avr. 2026] — Planification : ICA + classification + affichage dual

#### Objectif de la phase

Passer de la visualisation brute à un pipeline complet d'élimination d'artefacts avec comparaison visuelle avant/après.

#### Architecture cible

```
Fenêtre EEG (données pré-filtrées, mode offline)
      ↓
ICA decomposition (FastICA via MNE)
      ↓
Classification des composantes (heuristiques)
      ↓
Reconstruction du signal sans composantes artefact
      ↓
WebSocket → Flutter : { raw_window, clean_window, labels }
      ↓
Affichage côte-à-côte : signal brut | signal nettoyé
```

#### Angle de recherche retenu

Fine-tuning d'ICLabel sur le corpus **TUAR (TUH EEG Artifact Corpus)**.

**Problème avec ICLabel original :** 7 classes trop larges — CHEW et SHIV sont noyés dans
"Muscle", ELPP dans "Channel Noise". Impossible d'appliquer une stratégie de retrait différenciée.

**Contribution :** fine-tuner ICLabel pour obtenir 10 classes plus spécifiques issues de TUAR,
puis définir une stratégie de retrait optimisée par classe.

#### Mapping de classes

| ICLabel original (7) | Modèle fine-tuné (10) | Nouveauté |
|---------------------|----------------------|-----------|
| Brain | Brain | — |
| Eye | EYEM | — |
| Muscle | MUSC | — |
| Muscle | CHEW | **nouvelle classe** |
| Muscle | SHIV | **nouvelle classe** |
| Channel Noise | ELEC | — |
| Channel Noise | ELPP | **nouvelle classe** |
| Heart | Heart | — |
| Line Noise | Line Noise | — |
| Other | Other | — |

#### Stratégies de retrait par classe

| Classe | Stratégie |
|--------|-----------|
| EYEM | Rejet ICA standard |
| MUSC | Rejet ICA + filtre passe-bas agressif (>40 Hz) |
| CHEW | Rejet ICA + masquage temporel (bursts courts) |
| SHIV | Rejet ICA + filtre bande étroite (fréq. tremblement) |
| ELEC | Interpolation du canal |
| ELPP | Détection saut d'amplitude + interpolation |
| Heart | Rejet ICA standard |

#### Plan d'implémentation en 3 phases

**Phase 1 — Pipeline de données TUAR** (scripts Python indépendants)
- Parsing EDF TUAR + annotations
- FastICA sur chaque enregistrement → composantes ICA
- Extraction features ICLabel (PSD, autocorrélation, topomap scalp)
- Construction dataset `(features, label_TUAR)`

**Phase 2 — Fine-tuning ICLabel**
- Chargement des poids ICLabel pré-entraînés (PyTorch)
- Remplacement de la tête de classification (7 → 10 classes)
- Entraînement sur TUAR, évaluation sur hold-out
- Export du modèle fine-tuné (`.pt`)

**Phase 3 — Intégration temps réel**
- Chargement du modèle dans FastAPI
- FastICA → features → classification → reconstruction
- Nouveau message WebSocket `ica_window` (raw + clean + labels)
- Flutter : double panneau synchronisé (raw | clean)
- ORICA pour le mode online (après que l'offline fonctionne)

#### Décision d'implémentation

Priorité : faire fonctionner le pipeline complet avec **ICLabel pré-entraîné** (7 classes)
avant d'attaquer le fine-tuning TUAR. Le fine-tuning vient dans un second temps,
dans un projet d'entraînement séparé (Colab + Google Drive).

#### Statut
- [x] Angle de recherche défini (fine-tuning ICLabel sur TUAR)
- [x] Classes et stratégies de retrait documentées
- [x] Plan établi : ICLabel pré-entraîné d'abord, fine-tuning TUAR ensuite
- [ ] Étape 1 — ICLabel pré-entraîné + FastICA + double panneau Flutter
- [ ] Étape 2 — Fine-tuning ICLabel sur TUAR (projet séparé)
- [ ] ORICA (mode online)

---

### [15 avr. 2026 (suite)] — Implémentation ORICA pour le mode online

#### Contexte
Le mode offline utilise FastICA (batch, sur le dataset complet). Pour le mode online
(streaming lazy depuis un EDF), FastICA est impossible — il ne peut pas traiter un signal
dont on ne connaît pas encore l'intégralité. ORICA (Online Recursive ICA) résout ce problème :
il met à jour la matrice de séparation W à chaque fenêtre, sans charger le fichier entier.

#### Algorithme ORICA

ORICA est un algorithme ICA à gradient naturel mis à jour de façon récursive :

```python
Y = W @ X_whitened               # sources estimées
f_Y = tanh(Y)                    # non-linéarité super-gaussienne
lr_t = lr / (1 + t / 100_000)   # taux d'apprentissage décroissant
W += lr_t * (I - f_Y @ Y.T / N) @ W
```

Le taux d'apprentissage décroît pour assurer la convergence. W converge vers la
matrice de séparation des sources indépendantes.

#### Initialisation via ICLabel

À l'activation ICA en mode online :
1. Lecture d'un buffer de 10 s depuis `_raw` (lazy, sans tout charger)
2. `compute_ica_offline` sur ce buffer → ICLabel → `exclude` + `labels`
3. Création de `ORICAProcessor` avec PCA whitening ajusté sur ce buffer

Avantage : ICLabel fournit une classification robuste initiale des composantes ;
ORICA s'adapte ensuite en temps réel aux changements de signal.

#### Architecture `ORICAProcessor`

| Étape | Détail |
|-------|--------|
| Whitening | PCA (sklearn, `whiten=True`), ajusté sur buffer initial |
| W initial | Identité — converge en ligne vers FastICA |
| process(X) | Extrait canaux positionnés → PCA → ORICA update → reconstruction |
| Reconstruction | `pinv(W) @ Y_clean` → `pca.inverse_transform` → signal complet |

Seuls les canaux positionnés (montage standard_1005) participent à l'ICA ;
les canaux sans position sont retransmis inchangés.

#### Refactoring helper `_get_positioned_channels`

Extraction de la logique de résolution des noms de canaux (strip `.`, lookup
case-insensitive dans le montage) en une fonction partagée entre
`compute_ica_offline` et `_init_orica`.

#### Flutter — bouton ICA disponible en online

Le bouton `Icons.psychology` est maintenant actif en mode online (ORICA) et en
mode offline (FastICA). Le tooltip indique l'algorithme utilisé.
Le message de statut affiche "ORICA prête" ou "FastICA prête".

#### Fichiers modifiés
| Fichier | Modification |
|---------|-------------|
| `backend/main.py` | `_get_positioned_channels`, `ORICAProcessor`, `_init_orica`, handler `set_ica` online |
| `lib/main.dart` | Bouton ICA en mode online, tooltip algo, statut "ORICA"/"FastICA" |

#### Statut
- [x] ORICA implémenté (gradient naturel, whitening PCA)
- [x] Initialisation via ICLabel sur buffer de 10 s
- [x] Streaming loop online envoie `ica_window` via ORICA
- [x] Flutter : bouton ICA activé en mode online

---

### [15 mai 2026] — Correction bug `n_comp` dans le handler `set_ica` offline

#### Problème
Dans `receive_commands()` (WebSocket handler), l'appel à `compute_ica_offline` passait
une variable `n_comp` qui n'existait pas dans ce scope — uniquement définie à l'intérieur
de la fonction elle-même. Résultat : `NameError` au moment d'activer l'ICA en mode offline,
rendant FastICA inutilisable.

#### Correction
Suppression de l'argument superflu. `compute_ica_offline` utilise sa valeur par défaut
`n_components=15`, ce qui est le comportement voulu.

```python
# Avant (crash)
result = await loop.run_in_executor(
    None, compute_ica_offline,
    src, _channel_names, _sampling_rate, n_comp,
)

# Après (correct)
result = await loop.run_in_executor(
    None, compute_ica_offline,
    src, _channel_names, _sampling_rate,
)
```

#### Fichier modifié
| Fichier | Ligne |
|---------|-------|
| `backend/main.py` | ~1081 — appel `compute_ica_offline` dans handler `set_ica` |

---

### [19 mai 2026] — Remplacement du filtre FFT par un bandpass Butterworth + corrections UI

#### Contexte
Le pipeline disposait d'un filtre "FFT bandpass" qui utilisait la FFT comme outil intermédiaire (zeroing spectral entre `fft_low` et `fft_high`, puis IFFT). Ce filtre posait deux problèmes :
1. **Artefacts de Gibbs** : la coupure rectangulaire parfaite en fréquence introduit des oscillations dans le domaine temporel.
2. **Hypothèse de périodicité** : la FFT suppose un signal périodique, ce qui est faux pour l'EEG — distorsions aux bords de fenêtre.
De plus, il faisait doublon partiel avec le filtre low-pass Butterworth déjà en place.

#### Solution
Remplacement du filtre FFT par un **bandpass Butterworth ordre 4 zero-phase** (`filtfilt`), cohérent avec le notch et le low-pass déjà implémentés.

```python
# Avant — brick-wall FFT
spec[:, mask] = 0.0
out = irfft(spec, n=n, axis=1)

# Après — bandpass Butterworth
b, a = butter(4, [bandpass_low / (sr / 2), bandpass_high / (sr / 2)], btype="bandpass")
out[i] = filtfilt(b, a, out[i])
```

#### Renommage des paramètres
Les paramètres `fft_enabled / fft_low / fft_high` ont été renommés `bandpass_enabled / bandpass_low / bandpass_high` côté Python et Flutter. Le message WebSocket `set_filters` utilise les nouvelles clés.

#### Correction label "Clean (FastICA)" en mode ORICA
En mode online, la fenêtre "clean" affichait incorrectement le label `Clean (FastICA)` alors que c'est ORICA qui effectue la décomposition. Corrigé : le label est maintenant dynamique.

```dart
_cleanLabel = _isOnline ? 'Clean (ORICA)' : 'Clean (FastICA)';
```

#### Suppression de l'indicateur de convergence ORICA
Le panel "IC Activations" affichait une barre de progression et un texte "Converging… / Converged" basés sur le `nonstatidx`. Ces éléments ont été supprimés de l'UI — le panel affiche uniquement le titre et les composantes IC.

#### Fichiers modifiés
| Fichier | Modification |
|---------|-------------|
| `backend/main.py` | FFT supprimé, bandpass Butterworth ajouté, paramètres renommés |
| `lib/main.dart` | Paramètres renommés, label clean corrigé, indicateur convergence supprimé |

---

### [19 mai 2026 (suite)] — Correction bug mode online/offline après ORICA + ICLabel

#### Problème identifié
Après application de ORICA + ICLabel, plusieurs comportements incorrects apparaissaient :
1. Le label "Clean (FastICA)" s'affichait à la place de "Clean (ORICA)"
2. Le bouton eye blink devenait accessible (réservé au mode offline)
3. Re-appuyer sur le bouton ICA déclenchait FastICA (comportement offline)

#### Cause racine
La fonction `_run_iclabel_on_orica` (backend) force `_mode = "offline"` après avoir construit le signal nettoyé. Quand Flutter rappelle `_connect()` après ICLabel, le backend renvoie un `meta` avec `mode: "offline"`, ce qui bascule tout l'état Flutter en mode offline.

Ce comportement était intentionnel à l'origine (permettre la navigation libre après ICLabel), mais l'UI Flutter n'était pas prévue pour ce basculement.

#### Solution retenue
Une fois ORICA + ICLabel appliqué, l'état est définitif — impossible de désactiver le clean, il faut fermer le dataset et recommencer. Implémenté via un flag `_iclabelApplied` :

- `_iclabelApplied = true` après succès de `_applyIclabelOrica`
- Le bouton ICA est désactivé avec tooltip "Close dataset to restart"
- `_cleanLabel = 'Clean (ORICA)'` fixé directement dans `_applyIclabelOrica`, indépendamment de `_isOnline` (contourne le bug de label)
- Flag remis à `false` au disconnect et au reset

#### Problème résiduel connu
Le label "Clean (FastICA)" apparaît toujours dans certains cas pour ORICA. À investiguer.

#### Fichiers modifiés
| Fichier | Modification |
|---------|-------------|
| `lib/main.dart` | Ajout `_iclabelApplied`, bouton ICA bloqué post-ICLabel, label ORICA fixé |

---

*— fin des entrées actuelles —*
