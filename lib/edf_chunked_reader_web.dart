// Stub web — dart:io n'est pas disponible sur Flutter Web.
// Sur web, l'EDF est chargé entièrement en RAM via EEGEDFParser (comportement précédent).
// EDFChunkedReader n'est donc jamais instancié sur web ; ce stub satisfait
// l'import conditionnel sans provoquer d'erreur de compilation.

import 'eeg_signal.dart';

class EDFChunkedReader extends EEGDataSource {
  EDFChunkedReader._();

  static Future<EDFChunkedReader> open(
    String path, {
    void Function(double progress)? onProgress,
  }) {
    throw UnsupportedError('EDFChunkedReader non disponible sur Web.');
  }

  @override List<String> get channelNames => throw UnimplementedError();
  @override double       get samplingRate  => throw UnimplementedError();
  @override int          get numChannels   => throw UnimplementedError();
  @override int          get numSamples    => throw UnimplementedError();
  @override double channelMin(int ch)      => throw UnimplementedError();
  @override double channelMax(int ch)      => throw UnimplementedError();

  @override
  Future<List<List<double>>> getWindow(int startSample, int count) =>
      throw UnimplementedError();
}
