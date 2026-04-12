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

*— fin des entrées actuelles —*
