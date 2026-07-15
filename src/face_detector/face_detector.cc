#include "gpupixel/face_detector/face_detector.h"
#include <cassert>
#include "mars_vision/mars_defines.h"
#include "mars_vision/mars_face_landmarker.h"
#include "utils/filesystem.h"
#include "utils/logging.h"
#ifdef GPUPIXEL_ENABLE_TFLITE_DETECTOR
#include "face_detector/tflite_face_detector.h"
#endif

std::vector<float> FaceDetector::Detect(const uint8_t* data,
                                        int width,
                                        int height,
                                        int stride,
                                        GPUPIXEL_MODE_FMT fmt,
                                        GPUPIXEL_FRAME_TYPE type) {
#ifdef GPUPIXEL_ENABLE_TFLITE_DETECTOR
  // When TFLite detector is enabled at build time, route detection to the TFLite implementation.
  return RunTFLiteFaceDetector(data, width, height, stride, (int)fmt, (int)type);
#else
  mars_vision::MarsImage image;
  image.data = (uint8_t*)data;
  image.width = width == stride / 4 ? width : stride / 4;
  image.height = height;
  if (type == GPUPIXEL_FRAME_TYPE_RGBA) {
    image.format = mars_vision::MarsImageFormat::RGBA;
  } else if (type == GPUPIXEL_FRAME_TYPE_BGRA) {
    image.format = mars_vision::MarsImageFormat::BGRA;
  }
  image.stride = stride;
  image.rotate_type = mars_vision::RotateType::CLOCKWISE_0;
  image.timestamp = 0;

  std::vector<mars_vision::FaceLandmarkerResult> face_results;
  std::vector<float> landmarks;

  mars_face_detector_->Detect(image, face_results);
  // only support one face
  for (auto& result : face_results) {
    for (auto& point : result.key_points) {
      landmarks.push_back(point.x / width);
      landmarks.push_back(point.y / height);
    }
  }

  return landmarks;
#endif
}
