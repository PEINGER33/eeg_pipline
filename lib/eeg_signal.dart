import 'dart:typed_data';
import 'dart:math' as math;

class EEGSignal {
  final List<String> channelNames;
  final double samplingRate;
  final List<List<double>> data;

  EEGSignal({
    required this.channelNames,
    required this.samplingRate,
    required this.data,
  });

  int get numChannels => data.length;
  int get numSamples  => data.isEmpty ? 0 : data[0].length;
}

// ── CSV Parser ────────────────────────────────────────────────────────────────

class EEGCSVParser {
  static EEGSignal parse(String content) {
    final lines = content
        .replaceAll('\r\n', '\n')
        .replaceAll('\r', '\n')
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty)
        .toList();

    if (lines.isEmpty) throw Exception('Fichier CSV vide');

    double samplingRate = 128.0;
    final dataLines = <String>[];

    for (final line in lines) {
      if (line.startsWith('#')) {
        final match = RegExp(r'sampling_rate=(\d+)').firstMatch(line);
        if (match != null) samplingRate = double.parse(match.group(1)!);
      } else {
        dataLines.add(line);
      }
    }

    if (dataLines.isEmpty) throw Exception('Aucune donnee trouvee');

    final firstCells = dataLines[0].split(',');
    final hasHeader  = firstCells.any((c) => double.tryParse(c.trim()) == null);

    List<String> channelNames;
    int startRow;

    if (hasHeader) {
      channelNames = firstCells
          .map((c) => c.trim())
          .where((c) => c.toLowerCase() != 'eyedetection')
          .toList();
      startRow = 1;
    } else {
      channelNames = List.generate(firstCells.length, (i) => 'CH${i + 1}');
      startRow = 0;
    }

    final numCh   = channelNames.length;
    final rawData = List.generate(numCh, (_) => <double>[]);

    for (int i = startRow; i < dataLines.length; i++) {
      final cells = dataLines[i].split(',');
      for (int ch = 0; ch < numCh && ch < cells.length; ch++) {
        final v = double.tryParse(cells[ch].trim());
        if (v != null) rawData[ch].add(v);
      }
    }

    if (rawData.isEmpty || rawData[0].isEmpty) {
      throw Exception('Aucun echantillon numerique trouve');
    }

    return EEGSignal(
      channelNames: channelNames,
      samplingRate: samplingRate,
      data: rawData,
    );
  }
}

// ── EDF / EDF+ Parser ─────────────────────────────────────────────────────────
//
// Spec complète : https://www.edfplus.info/specs/edf.html
//
// Header fixe (256 octets) :
//   0   : 8   version
//   8   : 80  patient info
//   88  : 80  recording info
//   168 : 8   date (dd.mm.yy)
//   176 : 8   time (hh.mm.ss)
//   184 : 8   nb octets header total
//   192 : 44  reserved ("EDF+C" pour EDF+)
//   236 : 8   nb data records
//   244 : 8   durée data record (secondes)
//   252 : 4   nb signaux (ns)
//
// Headers par signal (ns × 256 octets, après le header fixe) :
//   ns × 16  label
//   ns × 80  transducer type
//   ns × 8   physical dimension
//   ns × 8   physical minimum
//   ns × 8   physical maximum
//   ns × 8   digital minimum
//   ns × 8   digital maximum
//   ns × 44  prefiltering
//   ns × 8   nb samples par record   ← champ clé
//   ns × 32  reserved

class EEGEDFParser {
  static EEGSignal parse(Uint8List bytes) {

    // ── Helpers ───────────────────────────────────────────────────
    String readField(int offset, int length) {
      if (offset < 0 || offset >= bytes.length) return '';
      final end = math.min(offset + length, bytes.length);
      return String.fromCharCodes(
        bytes.sublist(offset, end).where((b) => b >= 32 && b < 127),
      ).trim();
    }

    int    ri(int o, int l) => int.tryParse(readField(o, l))    ?? 0;
    double rd(int o, int l) => double.tryParse(readField(o, l)) ?? 0.0;

    // ── Validation minimale ───────────────────────────────────────
    if (bytes.length < 256) {
      throw Exception('Fichier trop petit (${bytes.length} o). EDF invalide.');
    }

    // ── Lecture header fixe ───────────────────────────────────────
    final totalHeaderBytes = ri(184, 8);
    final numRecords       = ri(236, 8);
    final recordDuration   = rd(244, 8);
    final ns               = ri(252, 4);

    if (ns <= 0)             throw Exception('EDF: ns=$ns invalide.');
    if (numRecords <= 0)     throw Exception('EDF: numRecords=$numRecords invalide.');
    if (recordDuration <= 0) throw Exception('EDF: recordDuration=$recordDuration invalide.');

    final expectedHeaderSize = 256 + ns * 256;
    if (bytes.length < expectedHeaderSize) {
      throw Exception(
        'Header incomplet: attendu $expectedHeaderSize o, recu ${bytes.length} o.'
      );
    }

    // ── Lecture headers par signal ────────────────────────────────
    //
    // IMPORTANT : les offsets sont calculés en accumulant
    // ns × fieldSize pour chaque champ précédent.
    // C'est la seule façon correcte — ne pas hardcoder.
    //
    // Tailles des champs (en octets par signal) :
    const List<int> fieldSizes = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32];
    // Index :                     0   1  2  3  4  5  6   7  8   9
    // Champs:
    //   0 = label
    //   1 = transducer
    //   2 = physDimension
    //   3 = physMin
    //   4 = physMax
    //   5 = digMin
    //   6 = digMax
    //   7 = prefiltering
    //   8 = nSamplesPerRecord  ← celui qui était mal calculé
    //   9 = reserved

    // Calcule l'offset absolu du début du champ fieldIndex
    int fieldStart(int fieldIndex) {
      int off = 256; // après le header fixe
      for (int f = 0; f < fieldIndex; f++) {
        off += ns * fieldSizes[f];
      }
      return off;
    }

    final labels   = <String>[];
    final physMins = <double>[];
    final physMaxs = <double>[];
    final digMins  = <double>[];
    final digMaxs  = <double>[];
    final nSampRec = <int>[];

    for (int i = 0; i < ns; i++) {
      labels.add(readField(fieldStart(0) + i * fieldSizes[0], fieldSizes[0]));
      physMins.add(rd(fieldStart(3) + i * fieldSizes[3], fieldSizes[3]));
      physMaxs.add(rd(fieldStart(4) + i * fieldSizes[4], fieldSizes[4]));
      digMins.add(rd(fieldStart(5)  + i * fieldSizes[5], fieldSizes[5]));
      digMaxs.add(rd(fieldStart(6)  + i * fieldSizes[6], fieldSizes[6]));
      nSampRec.add(ri(fieldStart(8) + i * fieldSizes[8], fieldSizes[8]));
    }

    // ── Filtrage des canaux annotations EDF+ ─────────────────────
    // Le canal annotation a un label "EDF Annotations" et
    // un nSamplesPerRecord différent des canaux EEG.
    // On le détecte par son label ET par nSampRec = 0 ou différent.
    final eegIdx = <int>[];
    for (int i = 0; i < ns; i++) {
      final lbl = labels[i].toUpperCase();
      final isAnnotation = lbl.contains('ANNOTATION') || lbl.contains('STIM');
      final hasSamples   = nSampRec[i] > 0;
      if (!isAnnotation && hasSamples) {
        eegIdx.add(i);
      }
    }

    // Si le filtre est trop strict, on prend tous les canaux avec nSampRec > 0
    if (eegIdx.isEmpty) {
      for (int i = 0; i < ns; i++) {
        if (nSampRec[i] > 0) eegIdx.add(i);
      }
    }

    if (eegIdx.isEmpty) {
      // Affiche les infos pour le debug
      final info = List.generate(ns, (i) =>
        '  [$i] "${labels[i]}" nSamp=${nSampRec[i]}'
      ).join('\n');
      throw Exception(
        'EDF: aucun canal avec des echantillons.\n'
        'ns=$ns, numRecords=$numRecords, recordDuration=$recordDuration\n'
        'Canaux:\n$info'
      );
    }

    // ── Facteurs de conversion digital → physique ─────────────────
    final gains   = <double>[];
    final offsets = <double>[];
    for (final i in eegIdx) {
      final dr   = digMaxs[i] - digMins[i];
      final gain = dr.abs() > 1e-10
          ? (physMaxs[i] - physMins[i]) / dr
          : 1.0;
      gains.add(gain);
      offsets.add(physMins[i] - digMins[i] * gain);
    }

    // ── Allocation des buffers ────────────────────────────────────
    final nValid     = eegIdx.length;
    final totSamples = eegIdx.map((i) => nSampRec[i] * numRecords).toList();
    final data       = List.generate(nValid, (j) => List<double>.filled(totSamples[j], 0.0));
    final writePos   = List<int>.filled(nValid, 0);

    // ── Décodage des data records ─────────────────────────────────
    // Les données commencent après le header complet
    int bytePos = totalHeaderBytes > 0 ? totalHeaderBytes : expectedHeaderSize;

    for (int rec = 0; rec < numRecords; rec++) {
      for (int sig = 0; sig < ns; sig++) {
        final n      = nSampRec[sig];
        final eegJ   = eegIdx.indexOf(sig);

        for (int s = 0; s < n; s++) {
          if (bytePos + 1 >= bytes.length) break;

          // Chaque sample = int16 little-endian signé
          final lo = bytes[bytePos];
          final hi = bytes[bytePos + 1];
          bytePos += 2;

          if (eegJ >= 0) {
            int raw = lo | (hi << 8);
            if (raw >= 0x8000) raw -= 0x10000; // extension de signe

            final phys = raw * gains[eegJ] + offsets[eegJ];
            final wp   = writePos[eegJ];
            if (wp < data[eegJ].length) {
              data[eegJ][wp] = phys;
              writePos[eegJ]++;
            }
          }
        }
      }
    }

    // ── Rognage et validation finale ──────────────────────────────
    final trimmed = List.generate(nValid, (j) => data[j].sublist(0, writePos[j]));

    // Retire les canaux qui n'ont finalement rien reçu
    final nonEmpty = <int>[];
    for (int j = 0; j < nValid; j++) {
      if (trimmed[j].isNotEmpty) nonEmpty.add(j);
    }

    if (nonEmpty.isEmpty) {
      throw Exception('EDF: aucune donnee decodee. Fichier peut-etre corrompu.');
    }

    final finalData     = [for (final j in nonEmpty) trimmed[j]];
    final finalLabels   = [for (final j in nonEmpty) labels[eegIdx[j]]];
    final samplingRate  = nSampRec[eegIdx[nonEmpty[0]]] / recordDuration;

    return EEGSignal(
      channelNames: finalLabels,
      samplingRate: samplingRate,
      data: finalData,
    );
  }
}

// ── Interface commune EEGDataSource ──────────────────────────────────────────
//
// Abstraite : CSV chargé en RAM (InMemoryEEGDataSource)
//             EDF lu par chunks via RandomAccessFile (EDFChunkedReader)

abstract class EEGDataSource {
  List<String> get channelNames;
  double       get samplingRate;
  int          get numChannels;
  int          get numSamples;

  double channelMin(int ch);
  double channelMax(int ch);

  // null pour les sources qui ne gardent pas tout en RAM (ex: EDF chunked)
  List<List<double>>? get allData => null;

  Future<List<List<double>>> getWindow(int startSample, int count);

  Future<void> close() async {}
}

// ── Source en mémoire (CSV, ou EDF web) ──────────────────────────────────────

class InMemoryEEGDataSource extends EEGDataSource {
  final EEGSignal _signal;
  late final List<double> _mins;
  late final List<double> _maxs;

  InMemoryEEGDataSource(this._signal) {
    _mins = List.generate(
      _signal.numChannels,
      (i) => _signal.data[i].isEmpty ? 0.0 : _signal.data[i].reduce(math.min),
    );
    _maxs = List.generate(
      _signal.numChannels,
      (i) => _signal.data[i].isEmpty ? 1.0 : _signal.data[i].reduce(math.max),
    );
  }

  @override List<String> get channelNames => _signal.channelNames;
  @override double       get samplingRate  => _signal.samplingRate;
  @override int          get numChannels   => _signal.numChannels;
  @override int          get numSamples    => _signal.numSamples;

  @override double channelMin(int ch) => _mins[ch];
  @override double channelMax(int ch) => _maxs[ch];

  @override List<List<double>>? get allData => _signal.data;

  @override
  Future<List<List<double>>> getWindow(int startSample, int count) {
    final end = (startSample + count).clamp(0, _signal.numSamples);
    return Future.value([
      for (int ch = 0; ch < _signal.numChannels; ch++)
        _signal.data[ch].sublist(startSample, end),
    ]);
  }
}

// ── Détection automatique CSV / EDF ──────────────────────────────────────────

class EEGParser {
  static EEGSignal parseFile({
    required String    filename,
    String?    csvContent,
    Uint8List? rawBytes,
  }) {
    final ext = filename.toLowerCase().split('.').last;

    if (ext == 'edf') {
      if (rawBytes == null) throw Exception('Donnees binaires manquantes pour EDF.');
      return EEGEDFParser.parse(rawBytes);
    }
    if (ext == 'csv') {
      if (csvContent == null) throw Exception('Contenu texte manquant pour CSV.');
      return EEGCSVParser.parse(csvContent);
    }

    throw Exception('Format non supporte: .$ext  (acceptes: .csv .edf)');
  }
}