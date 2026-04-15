// lib/preprocessing.dart
//
// Preprocessing EEG — Dart pur, zéro dépendance externe, compatible Web.
//
// Composants :
//   ButterworthLowPass  — filtre passe-bas IIR Butterworth 2nd ordre, zero-phase
//   NotchFilter         — coupe-bande IIR 2nd ordre zero-phase (50 ou 60 Hz)
//   MorletWavelet       — convolution avec ondelette de Morlet complexe (amplitude)
//   EEGPreprocessingPipeline — orchestrateur canal par canal
//
// Zero-phase : chaque filtre IIR est appliqué en avant puis en arrière (filtfilt),
// ce qui annule le déphasage et double l'ordre effectif du filtre.

import 'dart:math' as math;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers IIR
// ─────────────────────────────────────────────────────────────────────────────

/// Applique un filtre IIR Direct-Form II transposé (causal, un seul passage).
/// [b] = coefficients numérateur, [a] = coefficients dénominateur (a[0] == 1).
List<double> _iirFilter(List<double> x, List<double> b, List<double> a) {
  final n  = x.length;
  final nb = b.length;
  final na = a.length;
  final nz = math.max(nb, na) - 1; // taille de l'état interne
  final y  = List<double>.filled(n, 0.0);
  final z  = List<double>.filled(nz, 0.0);

  for (int i = 0; i < n; i++) {
    y[i] = b[0] * x[i] + (nz > 0 ? z[0] : 0.0);
    for (int j = 0; j < nz - 1; j++) {
      z[j] = b[j + 1] * x[i] - a[j + 1] * y[i] + z[j + 1];
    }
    if (nz > 0) {
      z[nz - 1] = b[math.min(nz, nb - 1)] * x[i]
          - a[math.min(nz, na - 1)] * y[i];
    }
  }
  return y;
}

/// Filtre zero-phase (filtfilt) : passage avant + passage arrière.
/// Élimine le déphasage introduit par le filtre IIR.
List<double> _filtfilt(List<double> x, List<double> b, List<double> a) {
  final forward  = _iirFilter(x, b, a);
  final reversed = forward.reversed.toList();
  final backward = _iirFilter(reversed, b, a);
  return backward.reversed.toList();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Butterworth Low-Pass 2nd ordre
// ─────────────────────────────────────────────────────────────────────────────
//
// Conception via transformation bilinéaire (Tustin) :
//   Wc = 2 * tan(pi * fc / fs)   (pré-distorsion de fréquence)
//   Puis calcul des coefficients b/a du filtre numérique 2nd ordre.

class ButterworthLowPass {
  final double cutoffHz;
  final double samplingRate;

  late final List<double> _b;
  late final List<double> _a;

  ButterworthLowPass({required this.cutoffHz, required this.samplingRate}) {
    final wc = 2.0 * math.tan(math.pi * cutoffHz / samplingRate);
    final k  = wc * wc;
    final q  = math.sqrt(2.0); // Q Butterworth 2nd ordre
    final d  = k + q * wc + 1.0;

    _b = [k / d, 2.0 * k / d, k / d];
    _a = [1.0, (2.0 * k - 2.0) / d, (k - q * wc + 1.0) / d];
  }

  /// Filtre un signal (liste de samples) — zero-phase.
  List<double> filter(List<double> signal) => _filtfilt(signal, _b, _a);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Notch Filter 2nd ordre (coupe-bande)
// ─────────────────────────────────────────────────────────────────────────────
//
// Filtre IIR notch (bi-quad) via transformation bilinéaire.
// Facteur de qualité Q contrôle la largeur de bande : Q=30 → bande étroite.

class NotchFilter {
  final double notchHz;
  final double samplingRate;
  final double q;

  late final List<double> _b;
  late final List<double> _a;

  NotchFilter({
    required this.notchHz,
    required this.samplingRate,
    this.q = 30.0,
  }) {
    final w0  = 2.0 * math.pi * notchHz / samplingRate;
    final bw  = w0 / q;
    final cos0 = math.cos(w0);

    // Coefficients du filtre notch numérique (bilinéaire simplifié)
    final d  = 1.0 + bw / 2.0;
    _b = [(1.0) / d, (-2.0 * cos0) / d, (1.0) / d];
    _a = [1.0, (-2.0 * cos0) / d, (1.0 - bw / 2.0) / d];
  }

  /// Filtre un signal — zero-phase.
  List<double> filter(List<double> signal) => _filtfilt(signal, _b, _a);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Morlet Wavelet (convolution, amplitude instantanée)
// ─────────────────────────────────────────────────────────────────────────────
//
// Ondelette de Morlet complexe : ψ(t) = exp(i·2π·fc·t) · exp(-t²/2σ²)
// On retourne l'amplitude (module) de la convolution complexe.
// Utile pour extraire la puissance dans une bande de fréquence cible.

class MorletWavelet {
  final double centerHz;   // fréquence centrale
  final double samplingRate;
  final double nCycles;    // nombre de cycles — contrôle durée / résolution

  MorletWavelet({
    required this.centerHz,
    required this.samplingRate,
    this.nCycles = 7.0,
  });

  /// Retourne l'amplitude instantanée du signal dans la bande [centerHz].
  List<double> amplitude(List<double> signal) {
    final sigma = nCycles / (2.0 * math.pi * centerHz);
    final halfLen = (3.0 * sigma * samplingRate).round();
    final kernelLen = 2 * halfLen + 1;

    // Génère les parties réelle et imaginaire de l'ondelette
    final kernelRe = List<double>.filled(kernelLen, 0.0);
    final kernelIm = List<double>.filled(kernelLen, 0.0);
    for (int i = 0; i < kernelLen; i++) {
      final t = (i - halfLen) / samplingRate;
      final env = math.exp(-t * t / (2.0 * sigma * sigma));
      final phase = 2.0 * math.pi * centerHz * t;
      kernelRe[i] = env * math.cos(phase);
      kernelIm[i] = env * math.sin(phase);
    }

    final n   = signal.length;
    final out = List<double>.filled(n, 0.0);

    for (int i = 0; i < n; i++) {
      double re = 0.0, im = 0.0;
      for (int k = 0; k < kernelLen; k++) {
        final si = i - halfLen + k;
        if (si < 0 || si >= n) continue;
        re += signal[si] * kernelRe[k];
        im += signal[si] * kernelIm[k];
      }
      out[i] = math.sqrt(re * re + im * im);
    }
    return out;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Paramètres du pipeline
// ─────────────────────────────────────────────────────────────────────────────

class PreprocessingParams {
  final bool   lowPassEnabled;
  final double lowPassCutoff;   // Hz

  final bool   notchEnabled;
  final double notchFreq;       // Hz (50 ou 60)

  final bool   waveletEnabled;
  final double waveletCenterHz; // Hz

  const PreprocessingParams({
    this.lowPassEnabled  = true,
    this.lowPassCutoff   = 40.0,
    this.notchEnabled    = true,
    this.notchFreq       = 50.0,
    this.waveletEnabled  = false,
    this.waveletCenterHz = 10.0,
  });

  PreprocessingParams copyWith({
    bool?   lowPassEnabled,
    double? lowPassCutoff,
    bool?   notchEnabled,
    double? notchFreq,
    bool?   waveletEnabled,
    double? waveletCenterHz,
  }) => PreprocessingParams(
    lowPassEnabled:  lowPassEnabled  ?? this.lowPassEnabled,
    lowPassCutoff:   lowPassCutoff   ?? this.lowPassCutoff,
    notchEnabled:    notchEnabled    ?? this.notchEnabled,
    notchFreq:       notchFreq       ?? this.notchFreq,
    waveletEnabled:  waveletEnabled  ?? this.waveletEnabled,
    waveletCenterHz: waveletCenterHz ?? this.waveletCenterHz,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pipeline orchestrateur
// ─────────────────────────────────────────────────────────────────────────────

class EEGPreprocessingPipeline {
  final PreprocessingParams params;
  final double              samplingRate;

  EEGPreprocessingPipeline({
    required this.params,
    required this.samplingRate,
  });

  /// Version async : cède le thread principal entre chaque canal
  /// pour ne pas bloquer l'UI pendant le filtrage.
  /// [onProgress] reçoit l'index du canal traité et le total.
  Future<List<List<double>>> processAsync(
    List<List<double>> data, {
    void Function(int current, int total)? onProgress,
  }) async {
    final lp = params.lowPassEnabled
        ? ButterworthLowPass(cutoffHz: params.lowPassCutoff, samplingRate: samplingRate)
        : null;
    final notch = params.notchEnabled
        ? NotchFilter(notchHz: params.notchFreq, samplingRate: samplingRate)
        : null;
    final wavelet = params.waveletEnabled
        ? MorletWavelet(centerHz: params.waveletCenterHz, samplingRate: samplingRate)
        : null;

    final result = <List<double>>[];
    for (int i = 0; i < data.length; i++) {
      // Yield au scheduler Flutter — l'UI reste réactive
      await Future.microtask(() {});
      result.add(_processChannel(data[i], lp: lp, notch: notch, wavelet: wavelet));
      onProgress?.call(i + 1, data.length);
    }
    return result;
  }

  List<double> _processChannel(
    List<double> ch, {
    ButterworthLowPass? lp,
    NotchFilter?        notch,
    MorletWavelet?      wavelet,
  }) {
    var out = ch;
    if (lp     != null) out = lp.filter(out);
    if (notch  != null) out = notch.filter(out);
    if (wavelet != null) out = wavelet.amplitude(out);
    return out;
  }
}
