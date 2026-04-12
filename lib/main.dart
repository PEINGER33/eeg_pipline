import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show SystemChrome, DeviceOrientation;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:file_picker/file_picker.dart';
import 'dart:math' as math;
import 'dart:async';

import 'eeg_signal.dart';
import 'edf_chunked_reader.dart';
import 'ica.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EEG ICA Pipeline',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        colorScheme: ColorScheme.dark(
          primary: const Color(0xFF4F8EF7),
          surface: const Color(0xFF1C2130),
        ),
      ),
      home: const HomePage(),
    );
  }
}

// ─────────────────────────────────────────────
//  Page principale
// ─────────────────────────────────────────────

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {

  // État de l'application
  EEGDataSource? _dataSource;
  ICAResult?     _ica;
  String         _status  = 'Importe un fichier CSV ou EDF pour commencer';
  bool           _running = false;
  int            _progressCurrent = 0;
  int            _progressTotal   = 0;

  // Composantes marquées comme artefacts
  final Set<int> _artifacts = {};

  // ── Import ──────────────────────────────────────────────────────
  bool   _importing      = false;
  String _importLabel    = '';
  double _importProgress = 0.0; // 0.0 = indéterminé, >0 = progression réelle

  // ── Simulation temps réel ────────────────────────────────────────
  bool   _simulating   = false;
  bool   _landscape    = false;
  double _windowPos    = 0.0;
  Timer? _simTimer;

  // Fenêtre courante décodée (chargée async depuis le data source)
  List<List<double>> _windowData   = [];
  bool               _windowLoading = false;

  static const int    _tickMs    = 16;    // ~60 fps — avance interne
  static const int    _renderMs  = 100;   // 10 fps — rafraîchissement UI
  static const double _windowSec = 4.0;

  int get _windowSize => _dataSource == null
      ? 256
      : (_dataSource!.samplingRate * _windowSec).round().clamp(1, _dataSource!.numSamples);

  double get _stepPerTick => _dataSource!.samplingRate * _tickMs / 1000.0;

  int get _windowStart => _windowPos.round();

  // ── Chargement + rendu de la fenêtre ────────────────────────────
  // Appelé uniquement par le timer de rendu (10 fps), pas à chaque tick.
  Future<void> _loadWindowAsync() async {
    if (_dataSource == null || _windowLoading) return;
    _windowLoading = true;
    try {
      final start = _windowStart;
      final size  = _windowSize;
      final data  = await _dataSource!.getWindow(start, size);
      if (mounted) setState(() => _windowData = data);
    } finally {
      _windowLoading = false;
    }
  }

  // ── Simulation ───────────────────────────────────────────────────
  // Deux timers :
  //   _simTimer   : avance _windowPos à 60fps (pas de setState, pas de rebuild)
  //   _renderTimer: déclenche le rebuild UI à 10fps seulement
  Timer? _renderTimer;

  void _startSimulation() {
    final windowSize = _windowSize;
    final maxStart   = (_dataSource!.numSamples - windowSize).toDouble();
    if (maxStart <= 0) return;

    setState(() => _simulating = true);

    // Timer interne — avance la position sans rebuilder l'UI
    _simTimer = Timer.periodic(
      const Duration(milliseconds: _tickMs),
      (_) {
        if (!mounted) { _simTimer?.cancel(); return; }
        if (_windowPos >= maxStart) {
          _simTimer?.cancel();
          _renderTimer?.cancel();
          setState(() => _simulating = false);
          return;
        }
        _windowPos = math.min(_windowPos + _stepPerTick, maxStart);
      },
    );

    // Timer de rendu — rebuild UI à 10fps
    _renderTimer = Timer.periodic(
      const Duration(milliseconds: _renderMs),
      (_) {
        if (!mounted || !_simulating) return;
        _loadWindowAsync();
      },
    );
  }

  void _stopSimulation() {
    _simTimer?.cancel();
    _renderTimer?.cancel();
    setState(() => _simulating = false);
  }

  void _toggleOrientation() {
    final next = !_landscape;
    SystemChrome.setPreferredOrientations(next
        ? [DeviceOrientation.landscapeLeft, DeviceOrientation.landscapeRight]
        : [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    setState(() => _landscape = next);
  }

  void _restartSimulation() {
    _simTimer?.cancel();
    _renderTimer?.cancel();
    _windowPos = 0.0;
    setState(() => _windowData = []);
    _loadWindowAsync();
    _startSimulation();
  }

  @override
  void dispose() {
    _simTimer?.cancel();
    _renderTimer?.cancel();
    _dataSource?.close();
    super.dispose();
  }

  // ── Import fichier ───────────────────────────────────────────────
  Future<void> _importFile() async {
    // CSV : withData pour lire le texte
    // EDF natif : withData: false, on utilise file.path avec RandomAccessFile
    // EDF web   : withData: true, dart:io indisponible donc chargement complet inévitable
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,  // FileType.custom grise les .edf sur Android (MIME inconnu)
      withData: true,
    );
    if (result == null) return;

    final file = result.files.first;
    final ext  = file.name.toLowerCase().split('.').last;

    setState(() {
      _importing       = true;
      _importProgress  = 0.0;
      _importLabel     = 'Lecture de ${file.name}…';
      _status          = 'Chargement de ${file.name}…';
      _running         = true;
    });

    // Laisse Flutter peindre l'overlay avant de démarrer le travail lourd
    await Future.delayed(const Duration(milliseconds: 50));

    try {
      EEGDataSource dataSource;

      if (ext == 'csv') {
        final content = String.fromCharCodes(file.bytes!);
        final signal  = EEGCSVParser.parse(content);
        dataSource    = InMemoryEEGDataSource(signal);

      } else if (ext == 'edf') {
        if (kIsWeb) {
          final signal = EEGEDFParser.parse(file.bytes!);
          dataSource = InMemoryEEGDataSource(signal);
        } else {
          setState(() {
            _importLabel    = 'Analyse du fichier EDF…';
            _importProgress = 0.0;
          });
          dataSource = await EDFChunkedReader.open(
            file.path!,
            onProgress: (p) => setState(() {
              _importProgress = p;
              _importLabel    = 'Analyse… ${(p * 100).toInt()} %';
            }),
          );
        }
      } else {
        throw Exception('Format non supporté : .$ext');
      }

      // Ferme l'ancien data source si nécessaire
      await _dataSource?.close();

      setState(() {
        _dataSource  = dataSource;
        _ica         = null;
        _artifacts.clear();
        _windowPos   = 0.0;
        _windowData  = [];
        _running     = false;
        _importing   = false;
        _status      = '✓ ${dataSource.numChannels} canaux · '
                       '${dataSource.numSamples} échantillons · '
                       '${dataSource.samplingRate.toStringAsFixed(0)} Hz';
      });

      // Charge la première fenêtre immédiatement
      _loadWindowAsync();

    } catch (e) {
      setState(() {
        _running   = false;
        _importing = false;
        _status    = 'ERREUR: $e';
      });
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Erreur'),
          content: Text(e.toString()),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('OK'),
            ),
          ],
        ),
      );
    }
  }

  // ── Lancer ICA ───────────────────────────────────────────────────
  Future<void> _runICA() async {
    if (_dataSource == null || _running) return;

    final allData = _dataSource!.allData;
    if (allData == null) {
      setState(() => _status = 'ICA non disponible : chargez un fichier CSV '
          'ou un EDF de taille raisonnable.');
      return;
    }

    setState(() {
      _running         = true;
      _ica             = null;
      _artifacts.clear();
      _progressCurrent = 0;
      _progressTotal   = _dataSource!.numChannels;
      _status          = 'ICA en cours…';
    });

    await Future.delayed(const Duration(milliseconds: 50));

    try {
      final ica    = SimpleICA(maxIter: 100, tolerance: 1e-4);
      final result = ica.fit(
        allData,
        _dataSource!.samplingRate,
        onProgress: (current, total) {
          setState(() {
            _progressCurrent = current;
            _progressTotal   = total;
            _status          = 'ICA : composante $current / $total';
          });
        },
      );
      setState(() {
        _ica     = result;
        _running = false;
        _status  = '✓ ${result.nComponents} composantes extraites';
      });
    } catch (e) {
      setState(() {
        _running = false;
        _status  = 'Erreur ICA : $e';
      });
    }
  }

  // ── Reconstruction ───────────────────────────────────────────────
  void _reconstruct() {
    if (_ica == null) return;
    final cleaned = _ica!.reconstructWithout(_artifacts);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          'Signal reconstruit : ${cleaned.length} canaux × ${cleaned[0].length} échantillons'
          ' (${_artifacts.length} IC retirée(s))',
        ),
        backgroundColor: Colors.green.shade700,
      ),
    );
  }

  // ── Build ─────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('EEG ICA Pipeline'),
        actions: [
          // Bouton rotation portrait / paysage
          IconButton(
            icon: Icon(_landscape ? Icons.stay_current_portrait : Icons.stay_current_landscape),
            tooltip: _landscape ? 'Mode portrait' : 'Mode paysage',
            onPressed: _toggleOrientation,
          ),
          // Bouton Import
          TextButton.icon(
            onPressed: _importFile,
            icon: const Icon(Icons.upload_file),
            label: const Text('Importer CSV / EDF'),
          ),
          const SizedBox(width: 8),
          // Boutons Simulation
          if (_dataSource != null && _ica == null) ...[
            if (_windowStart > 0)
              IconButton(
                icon: const Icon(Icons.replay),
                tooltip: 'Recommencer depuis le début',
                onPressed: _running ? null : _restartSimulation,
              ),
            FilledButton.icon(
              onPressed: _running ? null : (_simulating ? _stopSimulation : _startSimulation),
              icon: Icon(_simulating ? Icons.pause : Icons.sensors),
              label: Text(_simulating ? 'Pause' : 'Simuler temps réel'),
              style: FilledButton.styleFrom(
                backgroundColor: _simulating ? Colors.orange.shade800 : Colors.teal.shade700,
              ),
            ),
          ],
          const SizedBox(width: 8),
          // Bouton Reset
          if (_dataSource != null)
            IconButton(
              icon: const Icon(Icons.close),
              tooltip: 'Supprimer le dataset',
              onPressed: () {
                _stopSimulation();
                _dataSource?.close();
                setState(() {
                  _dataSource = null;
                  _ica        = null;
                  _artifacts.clear();
                  _windowData = [];
                  _status     = 'Importe un fichier CSV ou EDF pour commencer';
                });
              },
            ),
          const SizedBox(width: 8),
        ],
      ),
      body: Stack(
        children: [
          // ── Contenu principal ──────────────────────────────────
          Column(
            children: [
              _StatusBar(
                status:  _status,
                running: _running,
                current: _progressCurrent,
                total:   _progressTotal,
              ),
              Expanded(
                child: Row(
                  children: [
                    Expanded(
                      child: _ica != null
                          ? _ComponentsView(
                              ica:           _ica!,
                              artifacts:     _artifacts,
                              onToggle:      (i) => setState(() {
                                if (_artifacts.contains(i)) _artifacts.remove(i);
                                else _artifacts.add(i);
                              }),
                              onReconstruct: _reconstruct,
                            )
                          : _dataSource != null
                              ? _SignalPreview(
                                  dataSource:  _dataSource!,
                                  windowData:  _windowData,
                                  windowStart: _windowStart,
                                  windowSize:  _windowSize,
                                  simulating:  _simulating,
                                  loading:     _windowLoading && _windowData.isEmpty,
                                )
                              : const _EmptyView(),
                    ),
                  ],
                ),
              ),
            ],
          ),
          // ── Overlay import ─────────────────────────────────────
          if (_importing)
            Container(
              color: Colors.black54,
              child: Center(
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 24),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(_importLabel,
                            style: const TextStyle(fontSize: 14)),
                        const SizedBox(height: 16),
                        SizedBox(
                          width: 240,
                          child: LinearProgressIndicator(
                            value: _importProgress > 0 ? _importProgress : null,
                            minHeight: 6,
                            borderRadius: BorderRadius.circular(3),
                          ),
                        ),
                        if (_importProgress > 0) ...[
                          const SizedBox(height: 8),
                          Text(
                            '${(_importProgress * 100).toInt()} %',
                            style: const TextStyle(fontSize: 12, color: Colors.grey),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Barre de statut
// ─────────────────────────────────────────────

class _StatusBar extends StatelessWidget {
  final String status;
  final bool   running;
  final int    current;
  final int    total;

  const _StatusBar({
    required this.status,
    required this.running,
    required this.current,
    required this.total,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      color: Theme.of(context).colorScheme.surface,
      child: Row(
        children: [
          if (running)
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(status, style: const TextStyle(fontSize: 12)),
                  const SizedBox(height: 4),
                  LinearProgressIndicator(
                    value: total > 0 ? current / total : null,
                  ),
                ],
              ),
            )
          else
            Expanded(
              child: Text(status, style: const TextStyle(fontSize: 12)),
            ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Vue vide
// ─────────────────────────────────────────────

class _EmptyView extends StatelessWidget {
  const _EmptyView();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.show_chart, size: 64, color: Colors.grey),
          SizedBox(height: 16),
          Text('Aucun signal chargé',
               style: TextStyle(fontSize: 18, color: Colors.grey)),
          SizedBox(height: 8),
          Text('Clique sur "Importer CSV / EDF" pour commencer',
               style: TextStyle(color: Colors.grey)),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Aperçu du signal (avant ICA)
// ─────────────────────────────────────────────

class _SignalPreview extends StatelessWidget {
  final EEGDataSource        dataSource;
  final List<List<double>>   windowData;   // fenêtre déjà décodée
  final int  windowStart;
  final int  windowSize;
  final bool simulating;
  final bool loading;

  const _SignalPreview({
    required this.dataSource,
    required this.windowData,
    required this.windowStart,
    required this.windowSize,
    required this.simulating,
    required this.loading,
  });

  @override
  Widget build(BuildContext context) {
    final windowEnd = (windowStart + windowSize).clamp(0, dataSource.numSamples);
    final tStart    = windowStart / dataSource.samplingRate;
    final tEnd      = windowEnd   / dataSource.samplingRate;
    final totalDur  = dataSource.numSamples / dataSource.samplingRate;
    final progress  = dataSource.numSamples > windowSize
        ? windowStart / (dataSource.numSamples - windowSize)
        : 0.0;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // ── En-tête ──────────────────────────────────────────────
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
          child: Row(
            children: [
              Text(
                'Signal EEG — ${dataSource.numChannels} canaux',
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(width: 12),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: simulating ? Colors.teal.shade800 : Colors.grey.shade800,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '${tStart.toStringAsFixed(1)}s – ${tEnd.toStringAsFixed(1)}s'
                  '  /  ${totalDur.toStringAsFixed(1)}s',
                  style: const TextStyle(fontSize: 11),
                ),
              ),
            ],
          ),
        ),
        // ── Barre de position globale ────────────────────────────
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 2),
          child: SizedBox(
            height: 16,
            child: CustomPaint(
              painter: _ProgressScrubber(progress: progress.clamp(0.0, 1.0)),
              size: const Size(double.infinity, 16),
            ),
          ),
        ),
        // ── Canaux ───────────────────────────────────────────────
        Expanded(
          child: loading
              ? const Center(child: CircularProgressIndicator())
              : ListView.builder(
                  itemCount: dataSource.numChannels,
                  itemBuilder: (_, i) {
                    // min/max précompilés dans le dataSource (pas recalculés à chaque frame)
                    return _ChannelRow(
                      name: dataSource.channelNames[i],
                      data: i < windowData.length ? windowData[i] : const [],
                      yMin: dataSource.channelMin(i),
                      yMax: dataSource.channelMax(i),
                    );
                  },
                ),
        ),
        // ── Axe des abscisses (temps) ─────────────────────────────
        Padding(
          padding: const EdgeInsets.only(left: 64, right: 16, bottom: 8),
          child: SizedBox(
            height: 20,
            child: CustomPaint(
              painter: _TimeAxis(tStart: tStart, tEnd: tEnd),
              size: const Size(double.infinity, 20),
            ),
          ),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────
//  Barre de progression / scrubber global
// ─────────────────────────────────────────────

class _ProgressScrubber extends CustomPainter {
  final double progress;

  const _ProgressScrubber({required this.progress});

  @override
  void paint(Canvas canvas, Size size) {
    final track = Paint()
      ..color = Colors.white12
      ..style = PaintingStyle.fill;
    final fill = Paint()
      ..color = Colors.teal.shade400
      ..style = PaintingStyle.fill;
    final indicator = Paint()
      ..color = Colors.white70
      ..style = PaintingStyle.fill;

    final rrect = RRect.fromRectAndRadius(
      Rect.fromLTWH(0, 4, size.width, 8),
      const Radius.circular(4),
    );
    canvas.drawRRect(rrect, track);
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(0, 4, size.width * progress, 8),
        const Radius.circular(4),
      ),
      fill,
    );
    final cx = size.width * progress;
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(cx - 3, 0, 6, 16),
        const Radius.circular(3),
      ),
      indicator,
    );
  }

  @override
  bool shouldRepaint(_ProgressScrubber old) => old.progress != progress;
}

// ─────────────────────────────────────────────
//  Axe temps
// ─────────────────────────────────────────────

class _TimeAxis extends CustomPainter {
  final double tStart;
  final double tEnd;

  const _TimeAxis({required this.tStart, required this.tEnd});

  @override
  void paint(Canvas canvas, Size size) {
    final duration = tEnd - tStart;
    if (duration <= 0) return;

    final textStyle = const TextStyle(color: Colors.grey, fontSize: 10);
    final tickPaint = Paint()
      ..color = Colors.white24
      ..strokeWidth = 1;

    final double rawStep = duration / 5;
    final double step    = _niceStep(rawStep);

    double t = (tStart / step).ceil() * step;
    while (t <= tEnd) {
      final x = (t - tStart) / duration * size.width;
      canvas.drawLine(Offset(x, 0), Offset(x, 4), tickPaint);

      final label = t >= 60
          ? '${(t / 60).floor()}m${(t % 60).toStringAsFixed(0)}s'
          : '${t.toStringAsFixed(t < 10 ? 1 : 0)}s';
      final tp = TextPainter(
        text: TextSpan(text: label, style: textStyle),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(x - tp.width / 2, 6));
      t += step;
    }
  }

  double _niceStep(double raw) {
    const steps = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0];
    for (final s in steps) { if (raw <= s) return s; }
    return 60.0;
  }

  @override
  bool shouldRepaint(_TimeAxis old) => old.tStart != tStart || old.tEnd != tEnd;
}

class _ChannelRow extends StatelessWidget {
  final String       name;
  final List<double> data;
  final double?      yMin;
  final double?      yMax;

  const _ChannelRow({required this.name, required this.data, this.yMin, this.yMax});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 48,
            child: Text(name,
              style: const TextStyle(fontSize: 11, color: Colors.grey)),
          ),
          Expanded(
            child: SizedBox(
              height: 40,
              child: CustomPaint(painter: _MiniPlot(data: data, yMin: yMin, yMax: yMax)),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Vue des composantes ICA
// ─────────────────────────────────────────────

class _ComponentsView extends StatelessWidget {
  final ICAResult        ica;
  final Set<int>         artifacts;
  final void Function(int) onToggle;
  final VoidCallback       onReconstruct;

  const _ComponentsView({
    required this.ica,
    required this.artifacts,
    required this.onToggle,
    required this.onReconstruct,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Text(
                '${ica.nComponents} composantes indépendantes',
                style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
              ),
              const Spacer(),
              if (artifacts.isNotEmpty)
                FilledButton.icon(
                  onPressed: onReconstruct,
                  icon: const Icon(Icons.auto_fix_high, size: 16),
                  label: Text('Reconstruire sans ${artifacts.length} IC'),
                  style: FilledButton.styleFrom(
                    backgroundColor: Colors.green.shade700,
                  ),
                ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: ica.nComponents,
            itemBuilder: (_, i) => _ComponentCard(
              index:      i,
              data:       ica.components[i],
              isArtifact: artifacts.contains(i),
              onToggle:   () => onToggle(i),
            ),
          ),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────
//  Carte d'une composante ICA
// ─────────────────────────────────────────────

class _ComponentCard extends StatelessWidget {
  final int          index;
  final List<double> data;
  final bool         isArtifact;
  final VoidCallback onToggle;

  const _ComponentCard({
    required this.index,
    required this.data,
    required this.isArtifact,
    required this.onToggle,
  });

  double get kurtosis {
    final n = data.length;
    if (n < 4) return 0;
    final mean = data.reduce((a, b) => a + b) / n;
    double s2 = 0, s4 = 0;
    for (final x in data) {
      final d = x - mean;
      s2 += d * d;
      s4 += d * d * d * d;
    }
    final v = s2 / n;
    return v > 1e-10 ? (s4 / n) / (v * v) - 3.0 : 0.0;
  }

  @override
  Widget build(BuildContext context) {
    final kurt    = kurtosis;
    final suspect = kurt.abs() > 2.0;

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      color: isArtifact ? Colors.red.shade900.withOpacity(0.3) : null,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(
          color: isArtifact
              ? Colors.red.shade400
              : suspect
                  ? Colors.orange.shade400
                  : Colors.green.shade700,
          width: 1.5,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Row(
          children: [
            SizedBox(
              width: 100,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('IC${index + 1}',
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                  const SizedBox(height: 2),
                  Text('kurt: ${kurt.toStringAsFixed(2)}',
                    style: TextStyle(
                      fontSize: 11,
                      color: suspect ? Colors.orange : Colors.grey,
                    )),
                ],
              ),
            ),
            Expanded(
              child: SizedBox(
                height: 50,
                child: CustomPaint(
                  painter: _MiniPlot(
                    data:  data,
                    color: isArtifact
                        ? Colors.red.shade300
                        : suspect
                            ? Colors.orange
                            : Colors.blue.shade300,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 12),
            TextButton(
              onPressed: onToggle,
              style: TextButton.styleFrom(
                foregroundColor: isArtifact ? Colors.red : Colors.grey,
              ),
              child: Text(isArtifact ? 'Démarquer' : 'Artefact'),
            ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Mini tracé de signal (CustomPainter)
// ─────────────────────────────────────────────

class _MiniPlot extends CustomPainter {
  final List<double> data;
  final Color        color;
  final double?      yMin;
  final double?      yMax;

  const _MiniPlot({
    required this.data,
    this.color = Colors.blue,
    this.yMin,
    this.yMax,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (data.length < 2) return;

    final step = math.max(1, data.length ~/ 1000);

    double minV = yMin ?? double.infinity;
    double maxV = yMax ?? double.negativeInfinity;
    if (yMin == null || yMax == null) {
      for (int i = 0; i < data.length; i += step) {
        if (data[i] < minV) minV = data[i];
        if (data[i] > maxV) maxV = data[i];
      }
    }
    final range = maxV - minV;
    if (range < 1e-10) return;

    final paint = Paint()
      ..color       = color
      ..strokeWidth = 1.0
      ..style       = PaintingStyle.stroke;

    final path  = Path();
    bool  first = true;

    for (int i = 0; i < data.length; i += step) {
      final x = i / data.length * size.width;
      final y = size.height - (data[i] - minV) / range * size.height;
      if (first) { path.moveTo(x, y); first = false; }
      else        path.lineTo(x, y);
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(_MiniPlot old) =>
      old.data != data || old.color != color || old.yMin != yMin || old.yMax != yMax;
}
