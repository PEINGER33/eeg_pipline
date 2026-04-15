// Export conditionnel : native → implémentation dart:io, web → stub
export 'edf_chunked_reader_native.dart'
    if (dart.library.html) 'edf_chunked_reader_web.dart';
