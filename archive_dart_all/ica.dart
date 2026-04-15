import 'dart:math' as math;

// ─────────────────────────────────────────────
//  Résultat ICA
// ─────────────────────────────────────────────

class ICAResult {
  final List<List<double>> components;   // [nComposantes][nÉchantillons]
  final List<List<double>> mixingMatrix; // [nCanaux][nComposantes]
  final double samplingRate;

  ICAResult({
    required this.components,
    required this.mixingMatrix,
    required this.samplingRate,
  });

  int get nComponents => components.length;
  int get nSamples    => components.isEmpty ? 0 : components[0].length;

  /// Reconstruit le signal en excluant les composantes artefacts
  List<List<double>> reconstructWithout(Set<int> artifactIndices) {
    final nCh   = mixingMatrix.length;
    final nComp = components.length;
    final nT    = nSamples;

    final result = List.generate(nCh, (_) => List<double>.filled(nT, 0.0));

    for (int c = 0; c < nComp; c++) {
      if (artifactIndices.contains(c)) continue;
      for (int ch = 0; ch < nCh; ch++) {
        final a = mixingMatrix[ch][c];
        for (int t = 0; t < nT; t++) {
          result[ch][t] += a * components[c][t];
        }
      }
    }
    return result;
  }
}

// ─────────────────────────────────────────────
//  Pipeline ICA simplifié — 100% Dart pur
// ─────────────────────────────────────────────

class SimpleICA {
  final int    maxIter;
  final double tolerance;

  const SimpleICA({
    this.maxIter   = 100,
    this.tolerance = 1e-4,
  });

  ICAResult fit(
    List<List<double>> data,
    double samplingRate, {
    void Function(int current, int total)? onProgress,
  }) {
    final nCh = data.length;
    final nT  = data[0].length;

    // ── 1. Centrage ───────────────────────────────────────────────
    final Xc = _center(data);

    // ── 2. Blanchiment PCA ────────────────────────────────────────
    // Covariance C = Xc @ Xc.T / (T-1)
    final C = _cov(Xc, nT);

    // Eigendecomposition par itération de Jacobi (symétrique)
    final eigen = _jacobiEigen(C);
    final vals  = eigen.$1; // valeurs propres
    final vecs  = eigen.$2; // vecteurs propres en colonnes [nCh][nCh]

    // Matrice de blanchiment W = D^(-1/2) @ V.T
    // Matrice de déblanchiment A = V @ D^(1/2)
    final W    = List.generate(nCh, (i) {
      final scale = 1.0 / math.sqrt(vals[i].abs().clamp(1e-10, double.infinity));
      return List.generate(nCh, (j) => vecs[j][i] * scale);
    });

    final Winv = List.generate(nCh, (i) {
      final scale = math.sqrt(vals[i].abs().clamp(1e-10, double.infinity));
      return List.generate(nCh, (j) => vecs[i][j] * scale);
    });

    // Signal blanchi Z = W @ Xc  [nCh × nT]
    final Z = _matMul(W, Xc);

    // ── 3. FastICA déflationnaire ─────────────────────────────────
    final rng  = math.Random(42);
    final Wica = <List<double>>[];

    for (int c = 0; c < nCh; c++) {
      onProgress?.call(c + 1, nCh);

      // Initialisation aléatoire
      var w = _normalize(
        List.generate(nCh, (_) => rng.nextDouble() - 0.5),
      );

      for (int iter = 0; iter < maxIter; iter++) {
        // Projections u = w.T @ Z  → [nT]
        final u = List.generate(nT, (t) {
          double s = 0;
          for (int k = 0; k < nCh; k++) s += w[k] * Z[k][t];
          return s;
        });

        // Nonlinéarité logcosh : g(u) = tanh(u), g'(u) = 1 - tanh²(u)
        // tanh implémenté manuellement car math.tanh n'existe pas en Dart
        final gu     = u.map(_tanh).toList();
        final guDerM = u.fold(0.0, (s, v) {
          final t = _tanh(v);
          return s + (1.0 - t * t);
        }) / nT;

        // Mise à jour w_new = E[Z * g(u)] - E[g'(u)] * w
        final wNew = List.generate(nCh, (k) {
          double s = 0;
          for (int t = 0; t < nT; t++) s += Z[k][t] * gu[t];
          return s / nT - guDerM * w[k];
        });

        // Déflation de Gram-Schmidt
        for (final wp in Wica) {
          final proj = _dot(wNew, wp);
          for (int k = 0; k < nCh; k++) wNew[k] -= proj * wp[k];
        }

        final wNorm = _normalize(wNew);

        // Convergence
        final delta = 1.0 - _dot(w, wNorm).abs();
        w = wNorm;
        if (delta < tolerance) break;
      }

      Wica.add(w);
    }

    // ── 4. Composantes S = Wica @ Z  [nCh × nT] ─────────────────
    final S = _matMul(Wica, Z);

    // ── 5. Matrice de mélange A = Winv @ Wica.T  [nCh × nCh] ────
    final WicaT = _transpose(Wica);
    final A     = _matMul(Winv, WicaT); // [nCh × nCh]
    // On veut A[canal][composante] → transposée
    final Afinal = _transpose(A);

    return ICAResult(
      components:   S,
      mixingMatrix: Afinal,
      samplingRate: samplingRate,
    );
  }

  // ─────────────────────────────────────────────
  //  Helpers mathématiques
  // ─────────────────────────────────────────────

  /// tanh via la formule exponentielle (dart:math n'a pas tanh)
  double _tanh(double x) {
    if (x >  20.0) return  1.0;
    if (x < -20.0) return -1.0;
    final e2x = math.exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
  }

  /// Centre chaque ligne en soustrayant sa moyenne
  List<List<double>> _center(List<List<double>> X) {
    return X.map((row) {
      final mean = row.reduce((a, b) => a + b) / row.length;
      return row.map((v) => v - mean).toList();
    }).toList();
  }

  /// Matrice de covariance [nCh × nCh]
  List<List<double>> _cov(List<List<double>> X, int T) {
    final n = X.length;
    return List.generate(n, (i) =>
      List.generate(n, (j) {
        double s = 0;
        for (int t = 0; t < T; t++) s += X[i][t] * X[j][t];
        return s / (T - 1);
      })
    );
  }

  /// Eigendecomposition d'une matrice symétrique par itération de Jacobi
  /// Retourne (valeurs propres, vecteurs propres en colonnes)
  (List<double>, List<List<double>>) _jacobiEigen(List<List<double>> A) {
    final n = A.length;

    // Copie de travail
    final a = List.generate(n, (i) => List<double>.from(A[i]));

    // Matrice des vecteurs propres (initialisée à l'identité)
    final v = List.generate(n, (i) =>
      List.generate(n, (j) => i == j ? 1.0 : 0.0)
    );

    // Itérations de Jacobi
    for (int sweep = 0; sweep < 50; sweep++) {
      // Calcule la norme off-diagonale
      double offNorm = 0;
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          offNorm += a[i][j] * a[i][j];
        }
      }
      if (offNorm < 1e-12) break;

      // Rotation de Jacobi pour chaque paire (p, q)
      for (int p = 0; p < n - 1; p++) {
        for (int q = p + 1; q < n; q++) {
          if (a[p][q].abs() < 1e-14) continue;

          final theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
          final t = theta >= 0
              ?  1.0 / (theta + math.sqrt(1.0 + theta * theta))
              : -1.0 / (-theta + math.sqrt(1.0 + theta * theta));
          final cosA = 1.0 / math.sqrt(1.0 + t * t);
          final sinA = t * cosA;
          final tau  = sinA / (1.0 + cosA);

          // Met à jour la matrice a
          final app = a[p][p];
          final aqq = a[q][q];
          final apq = a[p][q];

          a[p][p] = app - t * apq;
          a[q][q] = aqq + t * apq;
          a[p][q] = 0.0;
          a[q][p] = 0.0;

          for (int r = 0; r < n; r++) {
            if (r == p || r == q) continue;
            final arp = a[r][p];
            final arq = a[r][q];
            a[r][p] = arp - sinA * (arq + tau * arp);
            a[p][r] = a[r][p];
            a[r][q] = arq + sinA * (arp - tau * arq);
            a[q][r] = a[r][q];
          }

          // Met à jour les vecteurs propres
          for (int r = 0; r < n; r++) {
            final vrp = v[r][p];
            final vrq = v[r][q];
            v[r][p] = vrp - sinA * (vrq + tau * vrp);
            v[r][q] = vrq + sinA * (vrp - tau * vrq);
          }
        }
      }
    }

    // Valeurs propres = diagonale de a, triées par ordre décroissant
    final eigenPairs = List.generate(n, (i) => (a[i][i], i));
    eigenPairs.sort((x, y) => y.$1.compareTo(x.$1));

    final sortedVals = eigenPairs.map((e) => e.$1).toList();
    final sortedVecs = List.generate(n, (i) =>
      List.generate(n, (j) => v[j][eigenPairs[i].$2])
    );

    return (sortedVals, sortedVecs);
  }

  /// Multiplication matricielle A [m×k] × B [k×n] → [m×n]
  List<List<double>> _matMul(List<List<double>> A, List<List<double>> B) {
    final m = A.length;
    final k = B.length;
    final n = B[0].length;
    return List.generate(m, (i) =>
      List.generate(n, (j) {
        double s = 0;
        for (int p = 0; p < k; p++) s += A[i][p] * B[p][j];
        return s;
      })
    );
  }

  /// Transposée
  List<List<double>> _transpose(List<List<double>> M) {
    final rows = M.length;
    final cols = M[0].length;
    return List.generate(cols, (j) =>
      List.generate(rows, (i) => M[i][j])
    );
  }

  /// Produit scalaire
  double _dot(List<double> a, List<double> b) {
    double s = 0;
    for (int i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  /// Normalisation L2
  List<double> _normalize(List<double> v) {
    final norm = math.sqrt(v.fold(0.0, (s, x) => s + x * x));
    if (norm < 1e-12) return List.filled(v.length, 0.0);
    return v.map((x) => x / norm).toList();
  }
}
