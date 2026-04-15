import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show SystemChrome, DeviceOrientation;
import 'dart:math' as math;
import 'dart:async';
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;

// ── URLs du backend ────────────────────────────────────────────────────────
// Sur émulateur Android  : 'http://10.0.2.2:8000'
// Sur appareil physique  : 'http://192.168.x.x:8000'
// Sur desktop / web      : 'http://localhost:8000'
const String kBackendHttp = 'http://localhost:8000';
const String kBackendWs   = 'ws://localhost:8000/ws';

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
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF4F8EF7),
          surface: Color(0xFF1C2130),
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
  // ── WebSocket ──────────────────────────────────────────────────────────────
  WebSocketChannel?   _wsChannel;
  StreamSubscription? _wsSub;
  bool _connected  = false;
  bool _connecting = false;
  bool _uploading  = false;
  bool _paused     = false;

  // ── Métadonnées ────────────────────────────────────────────────────────────
  List<String> _channelNames = [];
  double       _samplingRate = 256.0;
  int          _numSamples   = 0;
  List<double> _channelMin   = [];
  List<double> _channelMax   = [];

  // ── Fenêtre courante ───────────────────────────────────────────────────────
  List<List<double>> _windowData      = [];
  List<List<double>> _cleanWindowData = [];
  int                _windowStart     = 0;

  // ── UI ─────────────────────────────────────────────────────────────────────
  String _status    = 'Lance le serveur Python puis importe un fichier EDF ou CSV';
  bool   _landscape = false;
  String _mode      = '';

  // ── Preprocessing ──────────────────────────────────────────────────────────
  bool   _showPreprocessing = false;
  bool   _notchEnabled      = false;
  double _notchFreq         = 50.0;
  bool   _lowpassEnabled    = false;
  double _lowpassCutoff     = 40.0;

  // ── ICA ────────────────────────────────────────────────────────────────────
  bool         _icaEnabled   = false;
  bool         _icaComputing = false;
  List<String> _icaLabels    = [];
  List<int>    _icaRemoved   = [];

  double _windowSec = 5.0;
  static const List<double> kWindowOptions = [1, 2, 3, 4, 5, 10, 20, 30, 60, 120, 300];

  int get _windowSize =>
      (_samplingRate * _windowSec).round().clamp(1, math.max(1, _numSamples));

  // ── Commandes WebSocket ───────────────────────────────────────────────────

  void _setWindowSec(double secs) {
    setState(() => _windowSec = secs);
    _wsChannel?.sink.add(jsonEncode({'type': 'set_window', 'seconds': secs}));
  }

  void _sendFilters() {
    _wsChannel?.sink.add(jsonEncode({
      'type':            'set_filters',
      'notch_enabled':   _notchEnabled,
      'notch_freq':      _notchFreq,
      'lowpass_enabled': _lowpassEnabled,
      'lowpass_cutoff':  _lowpassCutoff,
    }));
  }

  void _toggleIca() {
    final next = !_icaEnabled;
    setState(() {
      _icaEnabled   = next;
      _icaComputing = next;
      if (!next) {
        _cleanWindowData = [];
        _icaLabels       = [];
        _icaRemoved      = [];
      }
    });
    _wsChannel?.sink.add(jsonEncode({
      'type':         'set_ica',
      'enabled':      next,
      'n_components': 15,
    }));
  }

  // ── Import ────────────────────────────────────────────────────────────────

  Future<void> _importCsv() => _pickAndUpload('csv', 'offline');

  Future<void> _importEdf() async {
    final choice = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Mode EDF'),
        content: const Text(
          'Online — lazy loading, streaming en temps réel\n\n'
          'Offline — tout en RAM, navigation libre (◀ ▶)',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, 'online'),
            child: const Text('Online'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, 'offline'),
            child: const Text('Offline'),
          ),
        ],
      ),
    );
    if (choice == null) return;
    await _pickAndUpload('edf', choice);
  }

  Future<void> _pickAndUpload(String ext, String mode) async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      withData: true,
    );
    if (result == null) return;

    final file = result.files.first;
    setState(() {
      _uploading = true;
      _status    = 'Upload de ${file.name}…';
    });

    try {
      final request = http.MultipartRequest('POST', Uri.parse('$kBackendHttp/upload'));
      request.files.add(http.MultipartFile.fromBytes('file', file.bytes!, filename: file.name));
      request.fields['mode'] = mode;

      final response = await request.send();
      final body     = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        setState(() => _uploading = false);
        _connect();
      } else {
        setState(() { _uploading = false; _status = 'Erreur upload : $body'; });
      }
    } catch (e) {
      setState(() { _uploading = false; _status = 'Impossible de joindre le backend : $e'; });
    }
  }

  // ── WebSocket ─────────────────────────────────────────────────────────────

  void _connect() {
    if (_connecting || _connected) return;
    setState(() {
      _connecting = true;
      _status     = 'Connexion à $kBackendWs…';
    });

    _wsChannel = WebSocketChannel.connect(Uri.parse(kBackendWs));
    _wsSub = _wsChannel!.stream.listen(
      _onMessage,
      onError: (e) {
        if (!mounted) return;
        setState(() { _connected = false; _connecting = false; _status = 'Erreur WebSocket : $e'; });
      },
      onDone: () {
        if (!mounted) return;
        setState(() { _connected = false; _connecting = false; _status = 'Déconnecté du serveur'; });
      },
      cancelOnError: true,
    );

    setState(() { _connecting = false; _connected = true; _status = 'Connecté — en attente des données…'; });
  }

  void _disconnect() {
    _wsSub?.cancel();
    _wsChannel?.sink.close();
    setState(() {
      _connected       = false;
      _paused          = false;
      _mode            = '';
      _channelNames    = [];
      _windowData      = [];
      _cleanWindowData = [];
      _windowStart     = 0;
      _icaEnabled      = false;
      _icaComputing    = false;
      _icaLabels       = [];
      _icaRemoved      = [];
      _status          = 'Déconnecté';
    });
  }

  void _reset() {
    _wsSub?.cancel();
    _wsChannel?.sink.close();
    setState(() {
      _connected       = false;
      _connecting      = false;
      _paused          = false;
      _mode            = '';
      _channelNames    = [];
      _windowData      = [];
      _cleanWindowData = [];
      _windowStart     = 0;
      _windowSec       = 5.0;
      _icaEnabled      = false;
      _icaComputing    = false;
      _icaLabels       = [];
      _icaRemoved      = [];
      _status          = 'Lance le serveur Python puis importe un fichier EDF ou CSV';
    });
  }

  void _seek(double progress) {
    if (_mode != 'offline') return;
    final pos = (progress * (_numSamples - _windowSize)).round().clamp(0, _numSamples);
    _wsChannel?.sink.add(jsonEncode({'type': 'seek', 'position': pos}));
  }

  void _togglePause() {
    if (!_connected) return;
    final next = !_paused;
    _wsChannel?.sink.add(jsonEncode({'type': next ? 'pause' : 'resume'}));
    setState(() => _paused = next);
  }

  void _onMessage(dynamic raw) {
    final msg = jsonDecode(raw as String) as Map<String, dynamic>;
    switch (msg['type'] as String) {

      case 'meta':
        setState(() {
          _channelNames = List<String>.from(msg['channels'] as List);
          _samplingRate = (msg['sampling_rate'] as num).toDouble();
          _numSamples   = msg['total_samples'] as int;
          _channelMin   = (msg['channel_min'] as List).map((e) => (e as num).toDouble()).toList();
          _channelMax   = (msg['channel_max'] as List).map((e) => (e as num).toDouble()).toList();
          _mode         = msg['mode'] as String? ?? 'offline';
          _status       = '${_channelNames.length} canaux · '
              '${_samplingRate.toStringAsFixed(0)} Hz · '
              '${(_numSamples / _samplingRate).toStringAsFixed(0)} s · '
              '${_mode == 'online' ? 'Online' : 'Offline'}';
        });

      case 'window':
        setState(() {
          _windowStart     = msg['start'] as int;
          _windowData      = _parseChannels(msg['data'] as List);
          _cleanWindowData = [];
        });

      case 'ica_window':
        setState(() {
          _windowStart     = msg['start'] as int;
          _windowData      = _parseChannels(msg['raw'] as List);
          _cleanWindowData = _parseChannels(msg['clean'] as List);
          _icaRemoved      = List<int>.from(msg['removed_components'] as List);
          _icaLabels       = List<String>.from(msg['labels'] as List);
        });

      case 'ica_status':
        final status = msg['status'] as String;
        if (status == 'computing') {
          setState(() { _icaComputing = true; _status = 'ICA en cours de calcul…'; });
        } else if (status == 'done') {
          final removed = List<int>.from(msg['removed_components'] as List);
          final labels  = List<String>.from(msg['labels'] as List);
          final algo    = _mode == 'online' ? 'ORICA' : 'FastICA';
          setState(() {
            _icaComputing = false;
            _icaRemoved   = removed;
            _icaLabels    = labels;
            _status       = '$algo prête — ${removed.length} composante(s) retirée(s)';
          });
        } else if (status == 'error') {
          setState(() {
            _icaEnabled   = false;
            _icaComputing = false;
            _status       = 'Erreur ICA : ${msg['message']}';
          });
        }

      case 'done':
        setState(() { _connected = false; _paused = false; _status = '✓ Simulation terminée'; });
    }
  }

  List<List<double>> _parseChannels(List raw) =>
      raw.map((ch) => (ch as List).map((e) => (e as num).toDouble()).toList()).toList();

  void _toggleOrientation() {
    final next = !_landscape;
    SystemChrome.setPreferredOrientations(next
        ? [DeviceOrientation.landscapeLeft, DeviceOrientation.landscapeRight]
        : [DeviceOrientation.portraitUp, DeviceOrientation.portraitDown]);
    setState(() => _landscape = next);
  }

  @override
  void dispose() {
    _wsSub?.cancel();
    _wsChannel?.sink.close();
    super.dispose();
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final hasData   = _channelNames.isNotEmpty;
    final isOffline = _mode == 'offline';

    return Scaffold(
      appBar: AppBar(
        title: const Text('EEG ICA Pipeline'),
        actions: [
          IconButton(
            icon: Icon(_landscape ? Icons.stay_current_portrait : Icons.stay_current_landscape),
            tooltip: _landscape ? 'Mode portrait' : 'Mode paysage',
            onPressed: _toggleOrientation,
          ),
          if (hasData) ...[
            // ── Preprocessing ──────────────────────────────────────
            IconButton(
              icon: Icon(
                Icons.tune,
                color: (_notchEnabled || _lowpassEnabled) ? Colors.teal.shade300 : null,
              ),
              tooltip: 'Preprocessing',
              onPressed: () => setState(() => _showPreprocessing = !_showPreprocessing),
            ),
            // ── ICA : FastICA (offline) ou ORICA (online) ─────────
            _icaComputing
                ? const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 12),
                    child: SizedBox(
                      width: 20, height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  )
                : IconButton(
                    icon: Icon(
                      Icons.psychology,
                      color: _icaEnabled ? Colors.purple.shade300 : null,
                    ),
                    tooltip: _icaEnabled
                        ? 'Désactiver ICA'
                        : isOffline ? 'Activer ICA (FastICA)' : 'Activer ICA (ORICA)',
                    onPressed: _connected ? _toggleIca : null,
                  ),
            IconButton(
              icon: const Icon(Icons.close),
              tooltip: 'Fermer le dataset',
              onPressed: _reset,
            ),
          ],
          if (!_connected) ...[
            FilledButton.icon(
              onPressed: (_uploading || _connecting) ? null : _importCsv,
              icon: const Icon(Icons.table_chart),
              label: const Text('CSV'),
              style: FilledButton.styleFrom(backgroundColor: Colors.indigo.shade700),
            ),
            const SizedBox(width: 6),
            FilledButton.icon(
              onPressed: (_uploading || _connecting) ? null : _importEdf,
              icon: (_uploading || _connecting)
                  ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.upload_file),
              label: Text(_uploading ? 'Upload…' : _connecting ? 'Connexion…' : 'EDF'),
              style: FilledButton.styleFrom(backgroundColor: Colors.teal.shade700),
            ),
          ] else
            TextButton.icon(
              onPressed: _disconnect,
              icon: const Icon(Icons.sensors_off),
              label: const Text('Déconnecter'),
            ),
          const SizedBox(width: 8),
        ],
      ),
      body: Column(
        children: [
          _StatusBar(status: _status, running: _connecting || _icaComputing),
          if (_showPreprocessing && hasData)
            _PreprocessingBar(
              notchEnabled:    _notchEnabled,
              notchFreq:       _notchFreq,
              lowpassEnabled:  _lowpassEnabled,
              lowpassCutoff:   _lowpassCutoff,
              onNotchToggle:   (v) { setState(() => _notchEnabled = v);   _sendFilters(); },
              onNotchFreq:     (v) { setState(() => _notchFreq = v);      _sendFilters(); },
              onLowpassToggle: (v) { setState(() => _lowpassEnabled = v); _sendFilters(); },
              onLowpassCutoff: (v) { setState(() => _lowpassCutoff = v);  _sendFilters(); },
            ),
          Expanded(
            child: hasData
                ? _SignalPreview(
                    channelNames:    _channelNames,
                    samplingRate:    _samplingRate,
                    numSamples:      _numSamples,
                    channelMin:      _channelMin,
                    channelMax:      _channelMax,
                    windowData:      _windowData,
                    cleanWindowData: _cleanWindowData,
                    windowStart:     _windowStart,
                    windowSize:      _windowSize,
                    simulating:      _connected,
                    paused:          _paused,
                    onTogglePause:   _togglePause,
                    offline:         isOffline,
                    onSeek:          _seek,
                    windowSec:       _windowSec,
                    windowOptions:   kWindowOptions,
                    onWindowChanged: _setWindowSec,
                    icaLabels:       _icaLabels,
                    icaRemoved:      _icaRemoved,
                  )
                : const _EmptyView(),
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
  const _StatusBar({required this.status, required this.running});

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
                  const LinearProgressIndicator(),
                ],
              ),
            )
          else
            Expanded(child: Text(status, style: const TextStyle(fontSize: 12))),
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
          Text('Aucun signal chargé', style: TextStyle(fontSize: 18, color: Colors.grey)),
          SizedBox(height: 8),
          Text('Lance le serveur Python et importe un fichier EDF ou CSV',
              style: TextStyle(color: Colors.grey)),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Aperçu du signal (mono ou dual panel)
// ─────────────────────────────────────────────

class _SignalPreview extends StatelessWidget {
  final List<String>         channelNames;
  final double               samplingRate;
  final int                  numSamples;
  final List<double>         channelMin;
  final List<double>         channelMax;
  final List<List<double>>   windowData;
  final List<List<double>>   cleanWindowData;
  final int                  windowStart;
  final int                  windowSize;
  final bool                 simulating;
  final bool                 paused;
  final VoidCallback         onTogglePause;
  final bool                 offline;
  final void Function(double) onSeek;
  final double               windowSec;
  final List<double>         windowOptions;
  final void Function(double) onWindowChanged;
  final List<String>         icaLabels;
  final List<int>            icaRemoved;

  const _SignalPreview({
    required this.channelNames,
    required this.samplingRate,
    required this.numSamples,
    required this.channelMin,
    required this.channelMax,
    required this.windowData,
    required this.cleanWindowData,
    required this.windowStart,
    required this.windowSize,
    required this.simulating,
    required this.paused,
    required this.onTogglePause,
    required this.offline,
    required this.onSeek,
    required this.windowSec,
    required this.windowOptions,
    required this.onWindowChanged,
    required this.icaLabels,
    required this.icaRemoved,
  });

  bool get _dualMode => cleanWindowData.isNotEmpty;

  @override
  Widget build(BuildContext context) {
    final windowEnd = (windowStart + windowSize).clamp(0, numSamples);
    final tStart    = windowStart / samplingRate;
    final tEnd      = windowEnd   / samplingRate;
    final totalDur  = numSamples  / samplingRate;
    final progress  = numSamples > windowSize
        ? windowStart / (numSamples - windowSize)
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
                'Signal EEG — ${channelNames.length} canaux',
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              const SizedBox(width: 8),
              if (simulating || paused)
                IconButton(
                  icon: Icon(
                    paused ? Icons.play_arrow : Icons.pause,
                    color: paused ? Colors.teal.shade300 : Colors.orange.shade300,
                  ),
                  tooltip: paused ? 'Reprendre' : 'Pause',
                  onPressed: onTogglePause,
                ),
              const SizedBox(width: 4),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: paused
                      ? Colors.orange.shade900
                      : simulating ? Colors.teal.shade800 : Colors.grey.shade800,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '${tStart.toStringAsFixed(1)}s – ${tEnd.toStringAsFixed(1)}s  /  ${totalDur.toStringAsFixed(1)}s',
                  style: const TextStyle(fontSize: 11),
                ),
              ),
              const Spacer(),
              DropdownButton<double>(
                value: windowSec,
                underline: const SizedBox(),
                isDense: true,
                style: const TextStyle(fontSize: 12, color: Colors.white70),
                dropdownColor: const Color(0xFF1C2130),
                items: windowOptions.map((s) {
                  final label = s >= 60
                      ? '${(s ~/ 60)}min${s % 60 > 0 ? ' ${(s % 60).toInt()}s' : ''}'
                      : '${s.toInt()}s';
                  return DropdownMenuItem(value: s, child: Text(label));
                }).toList(),
                onChanged: (v) { if (v != null) onWindowChanged(v); },
              ),
              const SizedBox(width: 8),
            ],
          ),
        ),

        // ── Barre de progression ──────────────────────────────────
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 2),
          child: LayoutBuilder(
            builder: (_, constraints) {
              void onTap(double dx) =>
                  onSeek((dx / constraints.maxWidth).clamp(0.0, 1.0));
              final scrubber = CustomPaint(
                painter: _ProgressScrubber(progress: progress.clamp(0.0, 1.0)),
                size: const Size(double.infinity, 16),
              );
              if (!offline) return SizedBox(height: 16, child: scrubber);
              return GestureDetector(
                onTapDown:              (d) => onTap(d.localPosition.dx),
                onHorizontalDragUpdate: (d) => onTap(d.localPosition.dx),
                child: MouseRegion(
                  cursor: SystemMouseCursors.click,
                  child: SizedBox(height: 16, child: scrubber),
                ),
              );
            },
          ),
        ),

        // ── Panneaux de signal ────────────────────────────────────
        Expanded(
          child: _dualMode
              ? Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Panneau gauche — signal brut
                    Expanded(
                      child: _ChannelPanel(
                        label:        'Brut',
                        labelColor:   Colors.blue.shade300,
                        channelNames: channelNames,
                        windowData:   windowData,
                        channelMin:   channelMin,
                        channelMax:   channelMax,
                        signalColor:  Colors.blue,
                      ),
                    ),
                    const VerticalDivider(width: 1, color: Colors.white12),
                    // Panneau droit — signal nettoyé (sans mise en évidence)
                    Expanded(
                      child: _ChannelPanel(
                        label:        'Nettoyé (ICA)',
                        labelColor:   Colors.green.shade300,
                        channelNames: channelNames,
                        windowData:   cleanWindowData,
                        channelMin:   channelMin,
                        channelMax:   channelMax,
                        signalColor:  Colors.green,
                      ),
                    ),
                  ],
                )
              : _ChannelPanel(
                  channelNames: channelNames,
                  windowData:   windowData,
                  channelMin:   channelMin,
                  channelMax:   channelMax,
                  signalColor:  Colors.blue,
                ),
        ),

        // ── Légende ICA ───────────────────────────────────────────
        if (_dualMode && icaRemoved.isNotEmpty)
          _IcaLegend(removed: icaRemoved, labels: icaLabels),

        // ── Axe temps ────────────────────────────────────────────
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
//  Panneau de canaux (réutilisé pour raw / clean)
// ─────────────────────────────────────────────

class _ChannelPanel extends StatelessWidget {
  final String?            label;
  final Color?             labelColor;
  final List<String>       channelNames;
  final List<List<double>> windowData;
  final List<double>       channelMin;
  final List<double>       channelMax;
  final Color              signalColor;

  const _ChannelPanel({
    this.label,
    this.labelColor,
    required this.channelNames,
    required this.windowData,
    required this.channelMin,
    required this.channelMax,
    required this.signalColor,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (label != null)
          Padding(
            padding: const EdgeInsets.fromLTRB(8, 4, 8, 2),
            child: Text(
              label!,
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.bold,
                color: labelColor ?? Colors.grey,
              ),
            ),
          ),
        Expanded(
          child: ListView.builder(
            itemCount: channelNames.length,
            itemBuilder: (_, i) => _ChannelRow(
              name:        channelNames[i],
              data:        i < windowData.length ? windowData[i] : const [],
              yMin:        i < channelMin.length ? channelMin[i] : null,
              yMax:        i < channelMax.length ? channelMax[i] : null,
              signalColor: signalColor,
            ),
          ),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────
//  Légende ICA
// ─────────────────────────────────────────────

class _IcaLegend extends StatelessWidget {
  final List<int>    removed;
  final List<String> labels;
  const _IcaLegend({required this.removed, required this.labels});

  static const _labelColors = {
    'eye blink':      Colors.orange,
    'muscle artifact': Colors.red,
    'heart beat':     Colors.pink,
    'line noise':     Colors.yellow,
    'channel noise':  Colors.purple,
    'other':          Colors.grey,
    'brain':          Colors.teal,
  };

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      color: const Color(0xFF0D1117),
      child: Row(
        children: [
          Icon(Icons.psychology, size: 14, color: Colors.purple.shade300),
          const SizedBox(width: 6),
          Text(
            'Retirées : ',
            style: TextStyle(fontSize: 11, color: Colors.purple.shade200),
          ),
          Expanded(
            child: Wrap(
              spacing: 8,
              children: List.generate(removed.length, (i) {
                final lbl   = i < labels.length ? labels[i] : '?';
                final color = _labelColors[lbl] ?? Colors.grey;
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.2),
                    border: Border.all(color: color.withValues(alpha: 0.6)),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    'IC${removed[i]} · $lbl',
                    style: TextStyle(fontSize: 10, color: color),
                  ),
                );
              }),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Scrubber de progression
// ─────────────────────────────────────────────

class _ProgressScrubber extends CustomPainter {
  final double progress;
  const _ProgressScrubber({required this.progress});

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(0, 4, size.width, 8), const Radius.circular(4)),
      Paint()..color = Colors.white12,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(0, 4, size.width * progress, 8), const Radius.circular(4)),
      Paint()..color = Colors.teal.shade400,
    );
    final cx = size.width * progress;
    canvas.drawRRect(
      RRect.fromRectAndRadius(Rect.fromLTWH(cx - 3, 0, 6, 16), const Radius.circular(3)),
      Paint()..color = Colors.white70,
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

    const textStyle = TextStyle(color: Colors.grey, fontSize: 10);
    final tickPaint = Paint()..color = Colors.white24..strokeWidth = 1;

    final step = _niceStep(duration / 5);
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
    for (final s in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]) {
      if (raw <= s) return s;
    }
    return 60.0;
  }

  @override
  bool shouldRepaint(_TimeAxis old) => old.tStart != tStart || old.tEnd != tEnd;
}

// ─────────────────────────────────────────────
//  Ligne de canal
// ─────────────────────────────────────────────

class _ChannelRow extends StatelessWidget {
  final String       name;
  final List<double> data;
  final double?      yMin;
  final double?      yMax;
  final Color        signalColor;

  const _ChannelRow({
    required this.name,
    required this.data,
    required this.signalColor,
    this.yMin,
    this.yMax,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 48,
            child: Text(name, style: const TextStyle(fontSize: 11, color: Colors.grey)),
          ),
          Expanded(
            child: SizedBox(
              height: 40,
              child: CustomPaint(
                painter: _MiniPlot(data: data, yMin: yMin, yMax: yMax, color: signalColor),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
//  Mini tracé de signal (CustomPainter)
// ─────────────────────────────────────────────

class _MiniPlot extends CustomPainter {
  final List<double> data;
  final double?      yMin;
  final double?      yMax;
  final Color        color;

  const _MiniPlot({
    required this.data,
    required this.color,
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

    // ── Tracé du signal ──────────────────────────────────────────
    final paint = Paint()..color = color..strokeWidth = 1.0..style = PaintingStyle.stroke;
    final path  = Path();
    bool  first = true;

    for (int i = 0; i < data.length; i += step) {
      final x = i / data.length * size.width;
      final y = size.height - (data[i] - minV) / range * size.height;
      if (first) { path.moveTo(x, y); first = false; } else { path.lineTo(x, y); }
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(_MiniPlot old) =>
      old.data != data || old.yMin != yMin || old.yMax != yMax || old.color != color;
}

// ─────────────────────────────────────────────
//  Barre de preprocessing
// ─────────────────────────────────────────────

class _PreprocessingBar extends StatelessWidget {
  final bool   notchEnabled;
  final double notchFreq;
  final bool   lowpassEnabled;
  final double lowpassCutoff;
  final void Function(bool)   onNotchToggle;
  final void Function(double) onNotchFreq;
  final void Function(bool)   onLowpassToggle;
  final void Function(double) onLowpassCutoff;

  const _PreprocessingBar({
    required this.notchEnabled,
    required this.notchFreq,
    required this.lowpassEnabled,
    required this.lowpassCutoff,
    required this.onNotchToggle,
    required this.onNotchFreq,
    required this.onLowpassToggle,
    required this.onLowpassCutoff,
  });

  static const _notchFreqs     = [50.0, 60.0];
  static const _lowpassCutoffs = [10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0];

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      color: const Color(0xFF151B2A),
      child: Row(
        children: [
          const Text('Preprocessing', style: TextStyle(fontSize: 11, color: Colors.grey)),
          const SizedBox(width: 16),
          FilterChip(
            label: const Text('Notch'),
            selected: notchEnabled,
            onSelected: onNotchToggle,
            selectedColor: Colors.teal.shade800,
            labelStyle: TextStyle(fontSize: 11, color: notchEnabled ? Colors.white : Colors.grey),
          ),
          if (notchEnabled) ...[
            const SizedBox(width: 6),
            DropdownButton<double>(
              value: notchFreq,
              underline: const SizedBox(),
              isDense: true,
              style: const TextStyle(fontSize: 11, color: Colors.white70),
              dropdownColor: const Color(0xFF1C2130),
              items: _notchFreqs.map((f) => DropdownMenuItem(value: f, child: Text('${f.toInt()} Hz'))).toList(),
              onChanged: (v) { if (v != null) onNotchFreq(v); },
            ),
          ],
          const SizedBox(width: 16),
          FilterChip(
            label: const Text('Low-pass'),
            selected: lowpassEnabled,
            onSelected: onLowpassToggle,
            selectedColor: Colors.indigo.shade700,
            labelStyle: TextStyle(fontSize: 11, color: lowpassEnabled ? Colors.white : Colors.grey),
          ),
          if (lowpassEnabled) ...[
            const SizedBox(width: 6),
            DropdownButton<double>(
              value: lowpassCutoff,
              underline: const SizedBox(),
              isDense: true,
              style: const TextStyle(fontSize: 11, color: Colors.white70),
              dropdownColor: const Color(0xFF1C2130),
              items: _lowpassCutoffs.map((f) => DropdownMenuItem(value: f, child: Text('${f.toInt()} Hz'))).toList(),
              onChanged: (v) { if (v != null) onLowpassCutoff(v); },
            ),
          ],
        ],
      ),
    );
  }
}
