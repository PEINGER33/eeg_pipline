// Lecteur EDF par chunks — plateforme native (Android, iOS, Desktop)
// Utilise dart:io RandomAccessFile pour ne charger qu'une fenêtre à la fois.
//
// Séquence d'utilisation :
//   final reader = await EDFChunkedReader.open(path);
//   final window = await reader.getWindow(startSample, count);
//   await reader.close();

import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;
import 'eeg_signal.dart';

class EDFChunkedReader extends EEGDataSource {
  final RandomAccessFile _raf;

  @override final List<String> channelNames;
  @override final double       samplingRate;
  @override final int          numSamples;

  final int          _totalHeaderBytes;
  final int          _numRecords;
  final int          _samplesPerRecord;   // échantillons par record pour les canaux EEG
  final int          _bytesPerRecord;     // octets totaux par data record (tous canaux)
  final List<int>    _chanByteOffset;     // offset en octets du canal j dans un record
  final List<int>    _chanSampCount;      // nb échantillons par record pour canal j
  final List<double> _gains;
  final List<double> _offsets;
  final List<double> _mins;
  final List<double> _maxs;

  EDFChunkedReader._({
    required RandomAccessFile raf,
    required this.channelNames,
    required this.samplingRate,
    required this.numSamples,
    required int totalHeaderBytes,
    required int numRecords,
    required int samplesPerRecord,
    required int bytesPerRecord,
    required List<int> chanByteOffset,
    required List<int> chanSampCount,
    required List<double> gains,
    required List<double> offsets,
    required List<double> mins,
    required List<double> maxs,
  })  : _raf              = raf,
        _totalHeaderBytes = totalHeaderBytes,
        _numRecords       = numRecords,
        _samplesPerRecord = samplesPerRecord,
        _bytesPerRecord   = bytesPerRecord,
        _chanByteOffset   = chanByteOffset,
        _chanSampCount    = chanSampCount,
        _gains            = gains,
        _offsets          = offsets,
        _mins             = mins,
        _maxs             = maxs;

  // ── Ouverture + parse header + scan min/max ────────────────────────────────
  static Future<EDFChunkedReader> open(
    String path, {
    void Function(double progress)? onProgress,
  }) async {
    final raf = await File(path).open();

    // ── Helpers ASCII ───────────────────────────────────────────────────────
    String readAscii(Uint8List buf, int off, int len) {
      final end = math.min(off + len, buf.length);
      return String.fromCharCodes(
        buf.sublist(off, end).where((b) => b >= 32 && b < 127),
      ).trim();
    }

    int    ri(Uint8List b, int o, int l) => int.tryParse(readAscii(b, o, l))    ?? 0;
    double rd(Uint8List b, int o, int l) => double.tryParse(readAscii(b, o, l)) ?? 0.0;

    // ── Header fixe (256 octets) ────────────────────────────────────────────
    final fixedHeader = Uint8List(256);
    await raf.readInto(fixedHeader);

    final totalHeaderBytes = ri(fixedHeader, 184, 8);
    final numRecords       = ri(fixedHeader, 236, 8);
    final recordDuration   = rd(fixedHeader, 244, 8);
    final ns               = ri(fixedHeader, 252, 4);

    if (ns <= 0 || numRecords <= 0 || recordDuration <= 0) {
      await raf.close();
      throw Exception('EDF header invalide (ns=$ns, records=$numRecords, dur=$recordDuration)');
    }

    // ── Headers par signal (ns × 256 octets) ───────────────────────────────
    final sigHeader = Uint8List(ns * 256);
    await raf.readInto(sigHeader);

    // Tailles des champs par signal (même ordre que la spec EDF)
    const fieldSizes = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32];

    int fieldOff(int fi) {
      int o = 0;
      for (int f = 0; f < fi; f++) o += ns * fieldSizes[f];
      return o;
    }

    String sigField(int fi, int si) =>
        readAscii(sigHeader, fieldOff(fi) + si * fieldSizes[fi], fieldSizes[fi]);

    final labels   = [for (int i = 0; i < ns; i++) sigField(0, i)];
    final physMins = [for (int i = 0; i < ns; i++) double.tryParse(sigField(3, i)) ?? 0.0];
    final physMaxs = [for (int i = 0; i < ns; i++) double.tryParse(sigField(4, i)) ?? 0.0];
    final digMins  = [for (int i = 0; i < ns; i++) double.tryParse(sigField(5, i)) ?? 0.0];
    final digMaxs  = [for (int i = 0; i < ns; i++) double.tryParse(sigField(6, i)) ?? 0.0];
    final nSampRec = [for (int i = 0; i < ns; i++) int.tryParse(sigField(8, i)) ?? 0];

    // ── Filtrage canaux EEG (exclusion annotations EDF+) ───────────────────
    var eegIdx = <int>[];
    for (int i = 0; i < ns; i++) {
      final lbl = labels[i].toUpperCase();
      if (!lbl.contains('ANNOTATION') && !lbl.contains('STIM') && nSampRec[i] > 0) {
        eegIdx.add(i);
      }
    }
    if (eegIdx.isEmpty) {
      eegIdx = [for (int i = 0; i < ns; i++) if (nSampRec[i] > 0) i];
    }
    if (eegIdx.isEmpty) {
      await raf.close();
      throw Exception('EDF: aucun canal EEG trouvé');
    }

    // ── Facteurs de conversion digital → physique ───────────────────────────
    final gains   = <double>[];
    final offsets = <double>[];
    for (final i in eegIdx) {
      final dr   = digMaxs[i] - digMins[i];
      final gain = dr.abs() > 1e-10 ? (physMaxs[i] - physMins[i]) / dr : 1.0;
      gains.add(gain);
      offsets.add(physMins[i] - digMins[i] * gain);
    }

    // ── Offsets en octets de chaque canal dans un data record ───────────────
    final bytesPerRecord = nSampRec.fold(0, (a, b) => a + b) * 2;
    final chanByteOffset = <int>[];
    final chanSampCount  = <int>[];
    for (final idx in eegIdx) {
      int off = 0;
      for (int k = 0; k < idx; k++) off += nSampRec[k] * 2;
      chanByteOffset.add(off);
      chanSampCount.add(nSampRec[idx]);
    }

    final samplesPerRecord = nSampRec[eegIdx[0]];
    final samplingRate     = samplesPerRecord / recordDuration;
    final numSamples       = samplesPerRecord * numRecords;
    final channelNames     = [for (final i in eegIdx) labels[i]];
    final actualHeader     =
        totalHeaderBytes > 0 ? totalHeaderBytes : 256 + ns * 256;

    // ── Scan séquentiel pour calculer min/max par canal ─────────────────────
    // Lecture record par record (séquentielle = rapide), aucune donnée stockée.
    final mins   = List<double>.filled(eegIdx.length, double.infinity);
    final maxs   = List<double>.filled(eegIdx.length, double.negativeInfinity);
    final recBuf = Uint8List(bytesPerRecord);

    await raf.setPosition(actualHeader);
    for (int rec = 0; rec < numRecords; rec++) {
      await raf.readInto(recBuf);
      for (int j = 0; j < eegIdx.length; j++) {
        final off = chanByteOffset[j];
        final n   = chanSampCount[j];
        for (int s = 0; s < n; s++) {
          int raw = recBuf[off + s * 2] | (recBuf[off + s * 2 + 1] << 8);
          if (raw >= 0x8000) raw -= 0x10000;
          final phys = raw * gains[j] + offsets[j];
          if (phys < mins[j]) mins[j] = phys;
          if (phys > maxs[j]) maxs[j] = phys;
        }
      }
      onProgress?.call((rec + 1) / numRecords);
    }

    return EDFChunkedReader._(
      raf:              raf,
      channelNames:     channelNames,
      samplingRate:     samplingRate,
      numSamples:       numSamples,
      totalHeaderBytes: actualHeader,
      numRecords:       numRecords,
      samplesPerRecord: samplesPerRecord,
      bytesPerRecord:   bytesPerRecord,
      chanByteOffset:   chanByteOffset,
      chanSampCount:    chanSampCount,
      gains:            gains,
      offsets:          offsets,
      mins:             mins,
      maxs:             maxs,
    );
  }

  // ── EEGDataSource ──────────────────────────────────────────────────────────

  @override int    get numChannels   => channelNames.length;
  @override double channelMin(int ch) => _mins[ch];
  @override double channelMax(int ch) => _maxs[ch];

  // allData reste null : les données ne sont pas gardées en RAM

  /// Lit uniquement les data records couvrant [startSample, startSample+count).
  /// Une seule lecture séquentielle par record concerné.
  @override
  Future<List<List<double>>> getWindow(int startSample, int count) async {
    final endSample = (startSample + count).clamp(0, numSamples);
    if (endSample <= startSample) return List.generate(numChannels, (_) => []);

    final startRec = startSample ~/ _samplesPerRecord;
    final endRec   = ((endSample - 1) ~/ _samplesPerRecord).clamp(0, _numRecords - 1);

    final result = List.generate(numChannels, (_) => <double>[]);
    final recBuf = Uint8List(_bytesPerRecord);

    for (int rec = startRec; rec <= endRec; rec++) {
      await _raf.setPosition(_totalHeaderBytes + rec * _bytesPerRecord);
      await _raf.readInto(recBuf);

      for (int j = 0; j < numChannels; j++) {
        final off = _chanByteOffset[j];
        final n   = _chanSampCount[j];
        for (int s = 0; s < n; s++) {
          final sampleIdx = rec * _samplesPerRecord + s;
          if (sampleIdx < startSample || sampleIdx >= endSample) continue;
          int raw = recBuf[off + s * 2] | (recBuf[off + s * 2 + 1] << 8);
          if (raw >= 0x8000) raw -= 0x10000;
          result[j].add(raw * _gains[j] + _offsets[j]);
        }
      }
    }

    return result;
  }

  @override
  Future<void> close() async => _raf.close();
}
