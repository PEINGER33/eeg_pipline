import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show SystemChrome, DeviceOrientation;
import 'dart:math' as math;
import 'dart:async';
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;

// ── Backend URLs ───────────────────────────────────────────────────────────
// Desktop / Chrome
const String kBackendHttp = 'http://localhost:8000';
const String kBackendWs   = 'ws://localhost:8000/ws';

// Mobile (same WiFi network)
// const String kBackendHttp = 'http://192.168.1.109:8000';
// const String kBackendWs   = 'ws://192.168.1.109:8000/ws';

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

  // ── Metadata ───────────────────────────────────────────────────────────────
  List<String> _channelNames = [];
  double       _samplingRate = 256.0;
  int          _numSamples   = 0;
  List<double> _channelMin   = [];
  List<double> _channelMax   = [];

  // ── Current window ─────────────────────────────────────────────────────────
  List<List<double>> _windowData  = [];
  int                _windowStart = 0;

  // ── IC panel (online ORICA) ────────────────────────────────────────────────
  List<List<double>> _icActivations = [];

  // ── UI ─────────────────────────────────────────────────────────────────────
  String _status    = 'Start the Python server then import an EDF or CSV file';
  bool   _landscape = false;
  String _mode      = '';

  // ── Preprocessing ──────────────────────────────────────────────────────────
  bool   _showPreprocessing = false;
  bool   _notchEnabled      = false;
  double _notchFreq         = 50.0;
  bool   _lowpassEnabled    = false;
  double _lowpassCutoff     = 40.0;
  bool   _fftEnabled        = false;
  double _fftLow            = 1.0;
  double _fftHigh           = 40.0;

  // ── Eye blink removal (Zhang 2017) ─────────────────────────────────────────
  bool _eyeBlinkEnabled   = false;
  bool _eyeBlinkComputing = false;
  int  _eyeBlinkNBlinks   = 0;

  // ── ICA ────────────────────────────────────────────────────────────────────
  bool               _icaEnabled      = false;
  bool               _icaComputing    = false;
  List<List<double>> _cleanWindowData = [];
  String             _cleanLabel      = 'Clean';
  List<String>       _icaLabels       = [];
  List<int>          _icaRemoved      = [];

  double _windowSec = 5.0;
  static const List<double> kWindowOptions = [1, 2, 3, 4, 5, 10, 20, 30, 60, 120, 300];

  int get _windowSize =>
      (_samplingRate * _windowSec).round().clamp(1, math.max(1, _numSamples));

  bool get _isOnline => _mode == 'online';

  // ── WebSocket commands ────────────────────────────────────────────────────

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
      'fft_enabled':     _fftEnabled,
      'fft_low':         _fftLow,
      'fft_high':        _fftHigh,
    }));
  }

  void _toggleEyeBlink() {
    final next = !_eyeBlinkEnabled;
    setState(() {
      _eyeBlinkEnabled   = next;
      _eyeBlinkComputing = next;
      if (!next) _eyeBlinkNBlinks = 0;
    });
    _wsChannel?.sink.add(jsonEncode({
      'type': 'set_direct_method', 'method': 'eyeblink', 'enabled': next,
    }));
  }

  void _toggleIca() {
    final next = !_icaEnabled;
    setState(() {
      _icaEnabled   = next;
      _icaComputing = next;
      if (!next) {
        _cleanWindowData = [];
        _icaLabels = []; _icaRemoved = [];
        _icActivations = [];
      }
    });
    _wsChannel?.sink.add(jsonEncode({
      'type': 'set_ica', 'enabled': next, 'n_components': 15,
    }));
  }


  // ── Import ────────────────────────────────────────────────────────────────

  Future<void> _importCsv() => _pickAndUpload('csv', 'offline');

  Future<void> _importEdf() async {
    final choice = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('EDF Mode'),
        content: const Text(
          'Online — lazy loading, real-time streaming\n\n'
          'Offline — full file in RAM, free navigation (◀ ▶)',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, 'online'), child: const Text('Online')),
          FilledButton(onPressed: () => Navigator.pop(ctx, 'offline'), child: const Text('Offline')),
        ],
      ),
    );
    if (choice == null) return;
    await _pickAndUpload('edf', choice);
  }

  Future<void> _pickAndUpload(String ext, String mode) async {
    final result = await FilePicker.platform.pickFiles(type: FileType.any, withData: true);
    if (result == null) return;
    final file = result.files.first;
    setState(() { _uploading = true; _status = 'Uploading ${file.name}…'; });
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
        setState(() { _uploading = false; _status = 'Upload error: $body'; });
      }
    } catch (e) {
      setState(() { _uploading = false; _status = 'Cannot reach backend: $e'; });
    }
  }

  // ── WebSocket ─────────────────────────────────────────────────────────────

  void _connect() {
    if (_connecting || _connected) return;
    setState(() { _connecting = true; _status = 'Connecting to $kBackendWs…'; });
    _wsChannel = WebSocketChannel.connect(Uri.parse(kBackendWs));
    _wsSub = _wsChannel!.stream.listen(
      _onMessage,
      onError: (e) {
        if (!mounted) return;
        setState(() { _connected = false; _connecting = false; _status = 'WebSocket error: $e'; });
      },
      onDone: () {
        if (!mounted) return;
        setState(() { _connected = false; _connecting = false; _status = 'Disconnected from server'; });
      },
      cancelOnError: true,
    );
    setState(() { _connecting = false; _connected = true; _status = 'Connected — waiting for data…'; });
  }

  void _disconnect() {
    _wsSub?.cancel();
    _wsChannel?.sink.close();
    setState(() {
      _connected = false; _paused = false; _mode = '';
      _channelNames = []; _windowData = []; _cleanWindowData = [];
      _windowStart = 0; _icaEnabled = false; _icaComputing = false;
      _icaLabels = []; _icaRemoved = [];
      _icActivations = [];
      _eyeBlinkEnabled = false; _eyeBlinkComputing = false; _eyeBlinkNBlinks = 0;
      _status = 'Disconnected';
    });
  }

  void _reset() {
    _wsSub?.cancel();
    _wsChannel?.sink.close();
    setState(() {
      _connected = false; _connecting = false; _paused = false; _mode = '';
      _channelNames = []; _windowData = []; _cleanWindowData = [];
      _windowStart = 0; _windowSec = 5.0;
      _icaEnabled = false; _icaComputing = false;
      _icaLabels = []; _icaRemoved = [];
      _icActivations = [];
      _eyeBlinkEnabled = false; _eyeBlinkComputing = false; _eyeBlinkNBlinks = 0;
      _status = 'Start the Python server then import an EDF or CSV file';
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
          _status       = '${_channelNames.length} channels · '
              '${_samplingRate.toStringAsFixed(0)} Hz · '
              '${(_numSamples / _samplingRate).toStringAsFixed(0)} s · '
              '${_mode == 'online' ? 'Online' : 'Offline'}';
        });

      case 'window':
        setState(() {
          _windowStart = msg['start'] as int;
          _windowData  = _parseChannels(msg['data'] as List);
          _cleanWindowData = [];
        });

      case 'ic_window':
        setState(() {
          _windowStart   = msg['start'] as int;
          _windowData    = _parseChannels(msg['raw'] as List);
          _icActivations = _parseChannels(msg['ic_activations'] as List);
        });

      case 'direct_window':
        setState(() {
          _windowStart     = msg['start'] as int;
          _windowData      = _parseChannels(msg['raw'] as List);
          _cleanWindowData = _parseChannels(msg['clean'] as List);
          _cleanLabel      = 'Eye Blink Removed';
        });

      case 'ica_window':
        setState(() {
          _windowStart     = msg['start'] as int;
          _windowData      = _parseChannels(msg['raw'] as List);
          _cleanWindowData = _parseChannels(msg['clean'] as List);
          _cleanLabel      = 'Clean (FastICA)';
          _icaRemoved      = List<int>.from(msg['removed_components'] as List);
          _icaLabels       = List<String>.from(msg['labels'] as List);
        });

      case 'ica_status':
        final st = msg['status'] as String;
        if (st == 'computing') {
          setState(() { _icaComputing = true; _status = 'ICA computing…'; });
        } else if (st == 'ready') {
          final n = msg['n_comp'] as int? ?? 0;
          setState(() {
            _icaComputing = false;
            _status       = 'ORICA ready — $n ICs';
          });
        } else if (st == 'done') {
          final removed = List<int>.from(msg['removed_components'] as List);
          final labels  = List<String>.from(msg['labels'] as List);
          setState(() {
            _icaComputing = false;
            _icaRemoved   = removed;
            _icaLabels    = labels;
            _status       = 'FastICA done — ${removed.length} component(s) removed';
          });
        } else if (st == 'error') {
          setState(() {
            _icaEnabled = false; _icaComputing = false;
            _status = 'ICA error: ${msg['message']}';
          });
        }

      case 'direct_method_status':
        final st = msg['status'] as String;
        if (st == 'computing') {
          setState(() { _eyeBlinkComputing = true; _status = 'Eye blink removal computing…'; });
        } else if (st == 'done') {
          final n         = msg['n_blinks'] as int? ?? 0;
          final effective = msg['effective'] as bool? ?? false;
          setState(() {
            _eyeBlinkComputing = false;
            _eyeBlinkNBlinks   = n;
            _status = effective
                ? 'Eye blink removal: $n blink(s) removed'
                : 'Eye blink removal: no blinks detected in Fp1';
          });
        } else if (st == 'error') {
          setState(() {
            _eyeBlinkEnabled   = false;
            _eyeBlinkComputing = false;
            _status = 'Eye blink error: ${msg['message']}';
          });
        }

      case 'done':
        setState(() { _connected = false; _paused = false; _status = '✓ Simulation complete'; });
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

  @override
  Widget build(BuildContext context) {
    final hasData   = _channelNames.isNotEmpty;
    final isOffline = _mode == 'offline';
    final icaOnline = _icaEnabled && _isOnline;

    return Scaffold(
      appBar: AppBar(
        title: const Text('EEG ICA Pipeline'),
        actions: [
          IconButton(
            icon: Icon(_landscape ? Icons.stay_current_portrait : Icons.stay_current_landscape),
            tooltip: _landscape ? 'Portrait' : 'Landscape',
            onPressed: _toggleOrientation,
          ),
          if (hasData) ...[
            IconButton(
              icon: Icon(Icons.tune,
                  color: (_notchEnabled || _lowpassEnabled || _fftEnabled)
                      ? Colors.teal.shade300 : null),
              tooltip: 'Preprocessing',
              onPressed: () => setState(() => _showPreprocessing = !_showPreprocessing),
            ),
            if (isOffline) ...[
              _eyeBlinkComputing
                  ? const Padding(
                      padding: EdgeInsets.symmetric(horizontal: 12),
                      child: SizedBox(width: 20, height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2,
                              color: Colors.orange)))
                  : Stack(
                      clipBehavior: Clip.none,
                      children: [
                        IconButton(
                          icon: Icon(Icons.remove_red_eye,
                              color: _eyeBlinkEnabled ? Colors.orange.shade300 : null),
                          tooltip: _eyeBlinkEnabled
                              ? 'Disable eye blink removal (Zhang 2017)'
                              : 'Enable eye blink removal (Zhang 2017)',
                          onPressed: _connected ? _toggleEyeBlink : null,
                        ),
                        if (_eyeBlinkEnabled && _eyeBlinkNBlinks > 0)
                          Positioned(
                            right: 4, top: 4,
                            child: Container(
                              padding: const EdgeInsets.all(3),
                              decoration: BoxDecoration(
                                color: Colors.orange.shade700,
                                shape: BoxShape.circle,
                              ),
                              child: Text('$_eyeBlinkNBlinks',
                                  style: const TextStyle(fontSize: 9, color: Colors.white)),
                            ),
                          ),
                      ],
                    ),
            ],
            _icaComputing
                ? const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 12),
                    child: SizedBox(width: 20, height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2)))
                : IconButton(
                    icon: Icon(Icons.psychology,
                        color: _icaEnabled ? Colors.purple.shade300 : null),
                    tooltip: _icaEnabled ? 'Disable ICA'
                        : isOffline ? 'Enable ICA (FastICA)' : 'Enable ICA (ORICA)',
                    onPressed: _connected ? _toggleIca : null,
                  ),
            IconButton(
              icon: const Icon(Icons.close),
              tooltip: 'Close dataset',
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
                  ? const SizedBox(width: 16, height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2))
                  : const Icon(Icons.upload_file),
              label: Text(_uploading ? 'Uploading…' : _connecting ? 'Connecting…' : 'EDF'),
              style: FilledButton.styleFrom(backgroundColor: Colors.teal.shade700),
            ),
          ] else
            TextButton.icon(
              onPressed: _disconnect,
              icon: const Icon(Icons.sensors_off),
              label: const Text('Disconnect'),
            ),
          const SizedBox(width: 8),
        ],
      ),
      body: Column(children: [
        _StatusBar(status: _status, running: _connecting || _icaComputing),
        if (_showPreprocessing && hasData)
          _PreprocessingBar(
            notchEnabled: _notchEnabled, notchFreq: _notchFreq,
            lowpassEnabled: _lowpassEnabled, lowpassCutoff: _lowpassCutoff,
            fftEnabled: _fftEnabled, fftLow: _fftLow, fftHigh: _fftHigh,
            onNotchToggle:   (v) { setState(() => _notchEnabled = v);   _sendFilters(); },
            onNotchFreq:     (v) { setState(() => _notchFreq = v);      _sendFilters(); },
            onLowpassToggle: (v) { setState(() => _lowpassEnabled = v); _sendFilters(); },
            onLowpassCutoff: (v) { setState(() => _lowpassCutoff = v);  _sendFilters(); },
            onFftToggle:     (v) { setState(() => _fftEnabled = v);     _sendFilters(); },
            onFftLow:        (v) { setState(() => _fftLow = v);         _sendFilters(); },
            onFftHigh:       (v) { setState(() => _fftHigh = v);        _sendFilters(); },
          ),
        Expanded(
          child: hasData
              ? _SignalView(
                  channelNames: _channelNames,
                  samplingRate: _samplingRate,
                  numSamples: _numSamples,
                  channelMin: _channelMin,
                  channelMax: _channelMax,
                  windowData: _windowData,
                  cleanWindowData: icaOnline ? [] : _cleanWindowData,
                  cleanLabel: _cleanLabel,
                  windowStart: _windowStart,
                  windowSize: _windowSize,
                  simulating: _connected,
                  paused: _paused,
                  onTogglePause: _togglePause,
                  offline: isOffline,
                  onSeek: _seek,
                  windowSec: _windowSec,
                  windowOptions: kWindowOptions,
                  onWindowChanged: _setWindowSec,
                  icaLabels: _icaLabels,
                  icaRemoved: _icaRemoved,
                  icaOnline: icaOnline,
                  icActivations: _icActivations,
                )
              : const _EmptyView(),
        ),
      ]),
    );
  }
}

// ─────────────────────────────────────────────
//  Status bar
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
      child: Row(children: [
        if (running)
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(status, style: const TextStyle(fontSize: 12)),
            const SizedBox(height: 4),
            const LinearProgressIndicator(),
          ]))
        else
          Expanded(child: Text(status, style: const TextStyle(fontSize: 12))),
      ]),
    );
  }
}

// ─────────────────────────────────────────────
//  Empty view
// ─────────────────────────────────────────────

class _EmptyView extends StatelessWidget {
  const _EmptyView();

  @override
  Widget build(BuildContext context) {
    return const Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
      Icon(Icons.show_chart, size: 64, color: Colors.grey),
      SizedBox(height: 16),
      Text('No signal loaded', style: TextStyle(fontSize: 18, color: Colors.grey)),
      SizedBox(height: 8),
      Text('Start the Python server and import an EDF or CSV file',
          style: TextStyle(color: Colors.grey)),
    ]));
  }
}

// ─────────────────────────────────────────────
//  Signal view (raw + optional IC panel / clean)
// ─────────────────────────────────────────────

class _SignalView extends StatelessWidget {
  final List<String>              channelNames;
  final double                    samplingRate;
  final int                       numSamples;
  final List<double>              channelMin;
  final List<double>              channelMax;
  final List<List<double>>        windowData;
  final List<List<double>>        cleanWindowData;
  final String                    cleanLabel;
  final int                       windowStart;
  final int                       windowSize;
  final bool                      simulating;
  final bool                      paused;
  final VoidCallback              onTogglePause;
  final bool                      offline;
  final void Function(double)     onSeek;
  final double                    windowSec;
  final List<double>              windowOptions;
  final void Function(double)     onWindowChanged;
  final List<String>              icaLabels;
  final List<int>                 icaRemoved;
  final bool                      icaOnline;
  final List<List<double>>        icActivations;
  const _SignalView({
    required this.channelNames,
    required this.samplingRate,
    required this.numSamples,
    required this.channelMin,
    required this.channelMax,
    required this.windowData,
    required this.cleanWindowData,
    required this.cleanLabel,
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
    required this.icaOnline,
    required this.icActivations,
  });

  bool get _offlineDual => cleanWindowData.isNotEmpty;

  @override
  Widget build(BuildContext context) {
    final windowEnd = (windowStart + windowSize).clamp(0, numSamples);
    final tStart    = windowStart / samplingRate;
    final tEnd      = windowEnd   / samplingRate;
    final totalDur  = numSamples  / samplingRate;
    final progress  = numSamples > windowSize
        ? windowStart / (numSamples - windowSize) : 0.0;

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      // ── Header ───────────────────────────────────────────────
      Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
        child: Row(children: [
          Text('EEG Signal — ${channelNames.length} channels',
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(width: 8),
          if (simulating || paused)
            IconButton(
              icon: Icon(paused ? Icons.play_arrow : Icons.pause,
                  color: paused ? Colors.teal.shade300 : Colors.orange.shade300),
              tooltip: paused ? 'Resume' : 'Pause',
              onPressed: onTogglePause,
            ),
          const SizedBox(width: 4),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: paused ? Colors.orange.shade900
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
        ]),
      ),

      // ── Progress bar ──────────────────────────────────────────
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 2),
        child: LayoutBuilder(builder: (_, constraints) {
          void onTap(double dx) => onSeek((dx / constraints.maxWidth).clamp(0.0, 1.0));
          final scrubber = CustomPaint(
            painter: _ProgressScrubber(progress: progress.clamp(0.0, 1.0)),
            size: const Size(double.infinity, 16),
          );
          if (!offline) return SizedBox(height: 16, child: scrubber);
          return GestureDetector(
            onTapDown:              (d) => onTap(d.localPosition.dx),
            onHorizontalDragUpdate: (d) => onTap(d.localPosition.dx),
            child: MouseRegion(cursor: SystemMouseCursors.click,
                child: SizedBox(height: 16, child: scrubber)),
          );
        }),
      ),

      // ── Signal panels ─────────────────────────────────────────
      Expanded(
        child: icaOnline
            // ORICA: raw (left) + IC panel (right)
            ? Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Expanded(child: _ChannelPanel(
                  label: 'Raw', labelColor: Colors.blue.shade300,
                  channelNames: channelNames, windowData: windowData,
                  channelMin: channelMin, channelMax: channelMax,
                  signalColor: Colors.blue,
                )),
                const VerticalDivider(width: 1, color: Colors.white12),
                Expanded(child: _IcPanel(icActivations: icActivations)),
              ])
            : _offlineDual
            // FastICA offline: raw (left) + clean (right)
            ? Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Expanded(child: _ChannelPanel(
                  label: 'Raw', labelColor: Colors.blue.shade300,
                  channelNames: channelNames, windowData: windowData,
                  channelMin: channelMin, channelMax: channelMax,
                  signalColor: Colors.blue,
                )),
                const VerticalDivider(width: 1, color: Colors.white12),
                Expanded(child: _ChannelPanel(
                  label: cleanLabel, labelColor: Colors.green.shade300,
                  channelNames: channelNames, windowData: cleanWindowData,
                  channelMin: channelMin, channelMax: channelMax,
                  signalColor: Colors.green,
                )),
              ])
            // No ICA: full width raw
            : _ChannelPanel(
                channelNames: channelNames, windowData: windowData,
                channelMin: channelMin, channelMax: channelMax,
                signalColor: Colors.blue,
              ),
      ),

      // ── ICA legend (offline only) ─────────────────────────────
      if (_offlineDual && icaRemoved.isNotEmpty)
        _IcaLegend(removed: icaRemoved, labels: icaLabels),

      // ── Time axis ─────────────────────────────────────────────
      Padding(
        padding: const EdgeInsets.only(left: 64, right: 16, bottom: 8),
        child: SizedBox(height: 20, child: CustomPaint(
          painter: _TimeAxis(tStart: tStart, tEnd: tEnd),
          size: const Size(double.infinity, 20),
        )),
      ),
    ]);
  }
}

// ─────────────────────────────────────────────
//  IC panel
// ─────────────────────────────────────────────

class _IcPanel extends StatelessWidget {
  final List<List<double>> icActivations;
  const _IcPanel({required this.icActivations});

  @override
  Widget build(BuildContext context) {
    final n = icActivations.length;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Padding(
        padding: const EdgeInsets.fromLTRB(8, 4, 8, 2),
        child: Text('IC Activations',
            style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold,
                color: Colors.purple.shade300)),
      ),
      Expanded(
        child: n == 0
            ? Center(child: Text('Waiting for ORICA…',
                style: TextStyle(fontSize: 11, color: Colors.grey.shade600)))
            : ListView.builder(
                itemCount: n,
                itemBuilder: (_, i) => Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  child: Row(children: [
                    SizedBox(width: 40,
                        child: Text('IC$i',
                            style: TextStyle(fontSize: 10,
                                color: Colors.purple.shade200))),
                    Expanded(child: SizedBox(height: 36, child: CustomPaint(
                      painter: _MiniPlot(data: icActivations[i],
                          color: Colors.purple.shade300),
                    ))),
                  ]),
                ),
              ),
      ),
    ]);
  }
}

// ─────────────────────────────────────────────
//  IC row (time series)

// ─────────────────────────────────────────────
//  Channel panel
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
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      if (label != null)
        Padding(
          padding: const EdgeInsets.fromLTRB(8, 4, 8, 2),
          child: Text(label!,
              style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold,
                  color: labelColor ?? Colors.grey)),
        ),
      Expanded(child: ListView.builder(
        itemCount: channelNames.length,
        itemBuilder: (_, i) => _ChannelRow(
          name: channelNames[i],
          data: i < windowData.length ? windowData[i] : const [],
          yMin: i < channelMin.length ? channelMin[i] : null,
          yMax: i < channelMax.length ? channelMax[i] : null,
          signalColor: signalColor,
        ),
      )),
    ]);
  }
}

// ─────────────────────────────────────────────
//  ICA legend (offline only)
// ─────────────────────────────────────────────

class _IcaLegend extends StatelessWidget {
  final List<int>    removed;
  final List<String> labels;
  const _IcaLegend({required this.removed, required this.labels});

  static const _labelColors = {
    'eye blink':       Colors.orange,
    'muscle artifact': Colors.red,
    'heart beat':      Colors.pink,
    'line noise':      Colors.yellow,
    'channel noise':   Colors.purple,
    'other':           Colors.grey,
    'brain':           Colors.teal,
  };

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      color: const Color(0xFF0D1117),
      child: Row(children: [
        Icon(Icons.psychology, size: 14, color: Colors.purple.shade300),
        const SizedBox(width: 6),
        Text('Removed: ', style: TextStyle(fontSize: 11, color: Colors.purple.shade200)),
        Expanded(child: Wrap(
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
              child: Text('IC${removed[i]} · $lbl',
                  style: TextStyle(fontSize: 10, color: color)),
            );
          }),
        )),
      ]),
    );
  }
}

// ─────────────────────────────────────────────
//  Progress scrubber
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
//  Time axis
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
//  Channel row
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
      child: Row(children: [
        SizedBox(width: 48,
            child: Text(name, style: const TextStyle(fontSize: 11, color: Colors.grey))),
        Expanded(child: SizedBox(height: 40, child: CustomPaint(
          painter: _MiniPlot(data: data, yMin: yMin, yMax: yMax, color: signalColor),
        ))),
      ]),
    );
  }
}

// ─────────────────────────────────────────────
//  Mini signal plot
// ─────────────────────────────────────────────

class _MiniPlot extends CustomPainter {
  final List<double> data;
  final double?      yMin;
  final double?      yMax;
  final Color        color;

  const _MiniPlot({required this.data, required this.color, this.yMin, this.yMax});

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
//  Preprocessing bar
// ─────────────────────────────────────────────

class _PreprocessingBar extends StatelessWidget {
  final bool   notchEnabled;
  final double notchFreq;
  final bool   lowpassEnabled;
  final double lowpassCutoff;
  final bool   fftEnabled;
  final double fftLow;
  final double fftHigh;
  final void Function(bool)   onNotchToggle;
  final void Function(double) onNotchFreq;
  final void Function(bool)   onLowpassToggle;
  final void Function(double) onLowpassCutoff;
  final void Function(bool)   onFftToggle;
  final void Function(double) onFftLow;
  final void Function(double) onFftHigh;

  const _PreprocessingBar({
    required this.notchEnabled,   required this.notchFreq,
    required this.lowpassEnabled, required this.lowpassCutoff,
    required this.fftEnabled,     required this.fftLow, required this.fftHigh,
    required this.onNotchToggle,  required this.onNotchFreq,
    required this.onLowpassToggle, required this.onLowpassCutoff,
    required this.onFftToggle,    required this.onFftLow, required this.onFftHigh,
  });

  static const _notchFreqs     = [50.0, 60.0];
  static const _lowpassCutoffs = [10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0];
  static const _fftLowOptions  = [0.5, 1.0, 2.0, 4.0, 8.0];
  static const _fftHighOptions = [20.0, 30.0, 40.0, 50.0, 70.0, 100.0];

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      color: const Color(0xFF151B2A),
      child: Row(children: [
        const Text('Preprocessing', style: TextStyle(fontSize: 11, color: Colors.grey)),
        const SizedBox(width: 16),
        FilterChip(
          label: const Text('Notch'), selected: notchEnabled, onSelected: onNotchToggle,
          selectedColor: Colors.teal.shade800,
          labelStyle: TextStyle(fontSize: 11, color: notchEnabled ? Colors.white : Colors.grey),
        ),
        if (notchEnabled) ...[
          const SizedBox(width: 6),
          DropdownButton<double>(
            value: notchFreq, underline: const SizedBox(), isDense: true,
            style: const TextStyle(fontSize: 11, color: Colors.white70),
            dropdownColor: const Color(0xFF1C2130),
            items: _notchFreqs.map((f) =>
                DropdownMenuItem(value: f, child: Text('${f.toInt()} Hz'))).toList(),
            onChanged: (v) { if (v != null) onNotchFreq(v); },
          ),
        ],
        const SizedBox(width: 16),
        FilterChip(
          label: const Text('Low-pass'), selected: lowpassEnabled, onSelected: onLowpassToggle,
          selectedColor: Colors.indigo.shade700,
          labelStyle: TextStyle(fontSize: 11, color: lowpassEnabled ? Colors.white : Colors.grey),
        ),
        if (lowpassEnabled) ...[
          const SizedBox(width: 6),
          DropdownButton<double>(
            value: lowpassCutoff, underline: const SizedBox(), isDense: true,
            style: const TextStyle(fontSize: 11, color: Colors.white70),
            dropdownColor: const Color(0xFF1C2130),
            items: _lowpassCutoffs.map((f) =>
                DropdownMenuItem(value: f, child: Text('${f.toInt()} Hz'))).toList(),
            onChanged: (v) { if (v != null) onLowpassCutoff(v); },
          ),
        ],
        const SizedBox(width: 16),
        FilterChip(
          label: const Text('FFT bandpass'), selected: fftEnabled, onSelected: onFftToggle,
          selectedColor: Colors.orange.shade800,
          labelStyle: TextStyle(fontSize: 11, color: fftEnabled ? Colors.white : Colors.grey),
        ),
        if (fftEnabled) ...[
          const SizedBox(width: 6),
          DropdownButton<double>(
            value: fftLow, underline: const SizedBox(), isDense: true,
            style: const TextStyle(fontSize: 11, color: Colors.white70),
            dropdownColor: const Color(0xFF1C2130),
            items: _fftLowOptions.map((f) =>
                DropdownMenuItem(value: f, child: Text('${f}Hz'))).toList(),
            onChanged: (v) { if (v != null) onFftLow(v); },
          ),
          const Padding(padding: EdgeInsets.symmetric(horizontal: 4),
              child: Text('–', style: TextStyle(fontSize: 11, color: Colors.white54))),
          DropdownButton<double>(
            value: fftHigh, underline: const SizedBox(), isDense: true,
            style: const TextStyle(fontSize: 11, color: Colors.white70),
            dropdownColor: const Color(0xFF1C2130),
            items: _fftHighOptions.map((f) =>
                DropdownMenuItem(value: f, child: Text('${f.toInt()}Hz'))).toList(),
            onChanged: (v) { if (v != null) onFftHigh(v); },
          ),
        ],
      ]),
    );
  }
}
