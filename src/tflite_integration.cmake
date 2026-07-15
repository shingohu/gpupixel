# CMake option to enable TFLite-based face detector
option(GPUPIXEL_ENABLE_TFLITE_DETECTOR "Build with TFLite-based face detector (requires TensorFlow Lite runtime installed)" OFF)

if(GPUPIXEL_ENABLE_TFLITE_DETECTOR)
  list(APPEND common_source_files
       ${CMAKE_CURRENT_SOURCE_DIR}/face_detector/tflite_face_detector.cc)

  # Try to find TensorFlow Lite headers and library. Users should point
  # TFLITE_INCLUDE_DIR and TFLITE_LIBRARIES to their local TFLite build if
  # automatic discovery fails.
  find_path(TFLITE_INCLUDE_DIR
    NAMES tensorflow/lite/interpreter.h
    HINTS /usr/include /usr/local/include ${PROJECT_SOURCE_DIR}/third_party/tflite/include)

  find_library(TFLITE_LIB
    NAMES tensorflowlite tensorflow-lite tflite
    HINTS /usr/lib /usr/local/lib ${PROJECT_SOURCE_DIR}/third_party/tflite/lib)

  if(NOT TFLITE_INCLUDE_DIR OR NOT TFLITE_LIB)
    message(FATAL_ERROR "GPUPIXEL_ENABLE_TFLITE_DETECTOR is ON but TensorFlow Lite headers or library were not found. Set TFLITE_INCLUDE_DIR and TFLITE_LIBRARIES or install TensorFlow Lite.")
  endif()

  # Add include dir & link library to the gpupixel target later (target exists below)
  list(APPEND EXTRA_TFLITE_INCLUDES ${TFLITE_INCLUDE_DIR})
  list(APPEND EXTRA_TFLITE_LIBS ${TFLITE_LIB})
endif()

# -- later after add_library(gpupixel...) we will append includes & link libs
# Append below where target is defined (we add code near the existing target_link_libraries blocks)
