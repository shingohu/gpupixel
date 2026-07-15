TFLite integration

This branch adds a prototype TFLite-based face detector implementation and CMake hooks.

How it works
- New source: src/face_detector/tflite_face_detector.cc/.h
  - Implements a PoC BlazeFace-style decode + simple NMS and maps landmarks to the
    existing FaceDetector::Detect return format (normalized [x,y,...]).
- CMake option: GPUPIXEL_ENABLE_TFLITE_DETECTOR
  - When ON, the new detector is compiled. You must provide TensorFlow Lite headers
    and libraries on your system (see below).
- Models: A helper script third_party/models/download_models.sh will fetch the
  official models from the face_detection_tflite repository into third_party/models.

How to enable and build (Linux / Android host)
1. Fetch models:
   ./third_party/models/download_models.sh

2. Install or build TensorFlow Lite C++ runtime on your system. Make sure you have
   tensorflow/lite/interpreter.h available under an include dir and libtflite available
   as a library (name may be tensorflowlite / tensorflow-lite / tflite). On Debian-like
   systems you may need to build from source.

3. Configure CMake with the option enabled and point to TFLite locations if CMake
   cannot auto-discover them:

cmake -B build -S . -DGPUPIXEL_ENABLE_TFLITE_DETECTOR=ON \
  -DTFLITE_INCLUDE_DIR=/path/to/tflite/include \
  -DTFLITE_LIB=/path/to/libtensorflowlite.so

4. Build:
   cmake --build build -j

Notes & Caveats
- This is a prototype implementation intended to demonstrate integration. The image
  preprocessing (resize/letterbox) uses a simple nearest-like approach and should be
  replaced with a proper high-quality resize + padding for production.
- The anchor generation and decode logic are adapted to match the face_detection_tflite
  expectations but may require tuning for exact parity.
- For mobile platforms (Android/iOS) you'll need to provide platform builds of
  TensorFlow Lite (cross-compiled) and link them in the platform-specific CMake toolchain.
