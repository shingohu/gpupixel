// TFLite-based face detector bridge (PoC)
// NOTE: This implementation is a prototype port of the BlazeFace-style
// decode flow used by the `face_detection_tflite` package. It requires a
// TensorFlow Lite C++ runtime to be available and linked by CMake when
// GPUPIXEL_ENABLE_TFLITE_DETECTOR is ON.

#include "face_detector/tflite_face_detector.h"
#include "utils/util.h"
#include "utils/logging.h"

#ifdef GPUPIXEL_ENABLE_TFLITE_DETECTOR

#include <memory>
#include <cmath>
#include <string>
#include <algorithm>
#include <numeric>

// TFLite headers
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

using namespace std;

namespace gpupixel {

// Simple SSD anchor options (subset of flutter_litert options used by the
// face_detection_tflite models). These values mirror the Dart constants.
struct SSDAnchorOptions {
  int num_layers;
  double min_scale;
  double max_scale;
  int input_size_width;
  int input_size_height;
  double anchor_offset_x;
  double anchor_offset_y;
  std::vector<int> strides;
};

static SSDAnchorOptions GetDefaultOptionsForModel() {
  // Use back-camera (full-range) by default (input 256x256, strides [16,32,32,32])
  return SSDAnchorOptions{
    4,        // num_layers
    0.1464,   // min_scale
    0.9,      // max_scale
    256,      // input_size_width
    256,      // input_size_height
    0.5,      // anchor_offset_x
    0.5,      // anchor_offset_y
    {16,32,32,32}
  };
}

// Generate SSD anchors (centers only). Returns vector of {cx, cy} pairs.
static std::vector<std::pair<double,double>> generateAnchors(const SSDAnchorOptions& opts) {
  std::vector<std::pair<double,double>> anchors;
  const int num_layers = opts.num_layers;
  const double min_scale = opts.min_scale;
  const double max_scale = opts.max_scale;

  for (int layer = 0; layer < num_layers; ++layer) {
    int stride = opts.strides[layer];
    int grid_w = opts.input_size_width / stride;
    int grid_h = opts.input_size_height / stride;

    for (int y = 0; y < grid_h; ++y) {
      for (int x = 0; x < grid_w; ++x) {
        double cx = (x + opts.anchor_offset_x) / (double)grid_w;
        double cy = (y + opts.anchor_offset_y) / (double)grid_h;
        anchors.emplace_back(cx, cy);
      }
    }
  }
  return anchors;
}

struct DecodedCandidate {
  double xmin, ymin, xmax, ymax;
  std::vector<double> keypoints; // flattened xy
  double score;
};

static double sigmoid_clipped(double x, double limit = 80.0) {
  if (x > limit) x = limit;
  if (x < -limit) x = -limit;
  return 1.0 / (1.0 + std::exp(-x));
}

// Very small IoU helper
static double iou(const DecodedCandidate& a, const DecodedCandidate& b) {
  double inter_x1 = std::max(a.xmin, b.xmin);
  double inter_y1 = std::max(a.ymin, b.ymin);
  double inter_x2 = std::min(a.xmax, b.xmax);
  double inter_y2 = std::min(a.ymax, b.ymax);
  double w = std::max(0.0, inter_x2 - inter_x1);
  double h = std::max(0.0, inter_y2 - inter_y1);
  double inter = w * h;
  double areaA = std::max(0.0, a.xmax - a.xmin) * std::max(0.0, a.ymax - a.ymin);
  double areaB = std::max(0.0, b.xmax - b.xmin) * std::max(0.0, b.ymax - b.ymin);
  double uni = areaA + areaB - inter;
  if (uni <= 0) return 0.0;
  return inter / uni;
}

// Weighted NMS (simple greedy NMS with IOU threshold)
static std::vector<DecodedCandidate> weighted_nms(std::vector<DecodedCandidate>& dets, double iou_thresh=0.3) {
  std::vector<DecodedCandidate> out;
  // Sort by score desc
  std::sort(dets.begin(), dets.end(), [](const DecodedCandidate& a, const DecodedCandidate& b){
    return a.score > b.score;
  });

  std::vector<char> removed(dets.size(), 0);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (removed[i]) continue;
    DecodedCandidate cur = dets[i];
    // Merge any overlapping detections into cur (simple average weighted by score)
    double totalScore = cur.score;
    std::vector<double> accumBox = {cur.xmin * cur.score, cur.ymin * cur.score, cur.xmax * cur.score, cur.ymax * cur.score};
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (removed[j]) continue;
      if (iou(cur, dets[j]) > iou_thresh) {
        removed[j] = 1;
        totalScore += dets[j].score;
        accumBox[0] += dets[j].xmin * dets[j].score;
        accumBox[1] += dets[j].ymin * dets[j].score;
        accumBox[2] += dets[j].xmax * dets[j].score;
        accumBox[3] += dets[j].ymax * dets[j].score;
      }
    }
    cur.xmin = accumBox[0] / totalScore;
    cur.ymin = accumBox[1] / totalScore;
    cur.xmax = accumBox[2] / totalScore;
    cur.ymax = accumBox[3] / totalScore;
    out.push_back(cur);
  }
  return out;
}

// Decode boxes & keypoints similar to decodeBlazeFaceCandidates from the Dart code
static std::vector<DecodedCandidate> decodeBlazeFaceCandidates(
    const float* scoresRaw,
    int scoresCount,
    const float* boxesRaw,
    int boxesElems,
    const std::vector<std::pair<double,double>>& anchors,
    int valuesPerBox,
    double scale,
    double minScore=0.5) {
  std::vector<DecodedCandidate> out;
  int anchorCount = scoresCount;
  for (int i = 0; i < anchorCount; ++i) {
    double s = sigmoid_clipped(scoresRaw[i]);
    if (!(s >= minScore)) continue;
    int base = i * valuesPerBox;
    // tmp values
    std::vector<double> tmp(valuesPerBox);
    for (int j = 0; j < valuesPerBox; ++j) tmp[j] = boxesRaw[base + j] / scale;
    double ax = anchors[i].first;
    double ay = anchors[i].second;
    tmp[0] += ax;
    tmp[1] += ay;
    for (int j = 4; j < valuesPerBox; j += 2) {
      tmp[j] += ax;
      tmp[j+1] += ay;
    }
    double xc = tmp[0], yc = tmp[1], w = tmp[2], h = tmp[3];
    if (w <= 0 || h <= 0) continue;
    DecodedCandidate cand;
    cand.xmin = xc - w * 0.5;
    cand.ymin = yc - h * 0.5;
    cand.xmax = xc + w * 0.5;
    cand.ymax = yc + h * 0.5;
    cand.score = s;
    // keypoints
    for (int j = 4; j < valuesPerBox; j += 2) {
      cand.keypoints.push_back(tmp[j]);
      cand.keypoints.push_back(tmp[j+1]);
    }
    out.push_back(cand);
  }
  return out;
}

// Helper to convert RGBA bytes to normalized float tensor [-1,1]
static void convertRGBAtoInputTensor(const uint8_t* rgba,
                                     int srcW, int srcH,
                                     int dstW, int dstH,
                                     float* outTensor) {
  // Simple letterbox: resize by nearest-neighbor into center and pad black
  // For PoC we do a naive center-crop + resize ignoring high-quality resize.
  // NOTE: For production use, replace with proper bilinear resize and padding.
  double scaleW = double(dstW) / srcW;
  double scaleH = double(dstH) / srcH;
  double scale = std::min(scaleW, scaleH);
  int newW = std::max(1, int(std::round(srcW * scale)));
  int newH = std::max(1, int(std::round(srcH * scale)));
  int padX = (dstW - newW) / 2;
  int padY = (dstH - newH) / 2;

  // Zero the output
  std::fill(outTensor, outTensor + dstW * dstH * 3, 0.0f);

  for (int y = 0; y < newH; ++y) {
    int srcY = std::min(srcH - 1, int(y / scale));
    for (int x = 0; x < newW; ++x) {
      int srcX = std::min(srcW - 1, int(x / scale));
      int srcIdx = (srcY * srcW + srcX) * 4;
      int dstX = x + padX;
      int dstY = y + padY;
      int dstIdx = (dstY * dstW + dstX) * 3;
      // Normalize to [-1,1]
      outTensor[dstIdx + 0] = (rgba[srcIdx + 0] / 127.5f) - 1.0f;
      outTensor[dstIdx + 1] = (rgba[srcIdx + 1] / 127.5f) - 1.0f;
      outTensor[dstIdx + 2] = (rgba[srcIdx + 2] / 127.5f) - 1.0f;
    }
  }
}

std::vector<float> RunTFLiteFaceDetector(const uint8_t* data,
                                         int width,
                                         int height,
                                         int stride,
                                         int fmt,
                                         int type) {
  std::vector<float> landmarks;

  // Determine model path from resources
  auto resourcePath = Util::GetResourcePath();
  std::string modelPath = (resourcePath / "models" / "face_detection_back.tflite").string();

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
  if (!model) {
    LOG_ERROR("TFLite model not found: {}", modelPath);
    return landmarks;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG_ERROR("Failed to construct TFLite interpreter");
    return landmarks;
  }

  // Allocate tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG_ERROR("AllocateTensors failed");
    return landmarks;
  }

  // Find input dims
  int input = interpreter->inputs()[0];
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int inH = dims->data[1];
  int inW = dims->data[2];
  int inC = dims->data[3];
  // Prepare input buffer
  std::vector<float> inputBuffer(inW * inH * 3);

  // data is expected RGBA (stride may be width*4)
  convertRGBAtoInputTensor(data, width, height, inW, inH, inputBuffer.data());

  // Copy to interpreter input (assuming float input)
  float* inTensor = interpreter->typed_tensor<float>(input);
  // Model might expect NHWC floats; copy
  std::copy(inputBuffer.begin(), inputBuffer.end(), inTensor);

  if (interpreter->Invoke() != kTfLiteOk) {
    LOG_ERROR("TFLite invoke failed");
    return landmarks;
  }

  // Find outputs: assume two outputs: boxes and scores
  const std::vector<int> outputs = interpreter->outputs();
  if (outputs.size() < 2) {
    LOG_ERROR("Unexpected model outputs size: {}", outputs.size());
    return landmarks;
  }

  // Heuristic: larger output is boxes, smaller is scores
  int boxesIdx = 0;
  for (size_t i = 1; i < outputs.size(); ++i) {
    int count0 = interpreter->tensor(outputs[boxesIdx])->bytes / sizeof(float);
    int counti = interpreter->tensor(outputs[i])->bytes / sizeof(float);
    if (counti > count0) boxesIdx = i;
  }
  int scoresIdx = (boxesIdx == 0) ? 1 : 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if ((int)i == boxesIdx) continue;
    int c = interpreter->tensor(outputs[i])->bytes / sizeof(float);
    int s = interpreter->tensor(outputs[scoresIdx])->bytes / sizeof(float);
    if (c < s) scoresIdx = (int)i;
  }

  float* boxesRaw = interpreter->typed_output_tensor<float>(boxesIdx);
  float* scoresRaw = interpreter->typed_output_tensor<float>(scoresIdx);
  int boxesElems = interpreter->tensor(outputs[boxesIdx])->bytes / sizeof(float);
  int scoresElems = interpreter->tensor(outputs[scoresIdx])->bytes / sizeof(float);

  // K = valuesPerBox
  int valuesPerBox = boxesElems / scoresElems;

  // Generate anchors
  SSDAnchorOptions opts = GetDefaultOptionsForModel();
  auto anchors = generateAnchors(opts);
  if ((int)anchors.size() != scoresElems) {
    LOG_ERROR("Anchor count {} does not match scores {}", anchors.size(), scoresElems);
    // continue anyway
  }

  // Decode candidates
  auto candidates = decodeBlazeFaceCandidates(scoresRaw, scoresElems, boxesRaw, boxesElems, anchors, valuesPerBox, (double)inW, 0.5);

  // NMS
  auto detections = weighted_nms(candidates, 0.3);

  if (detections.empty()) return landmarks;

  // For parity with existing GPUPixel detector, return first face's keypoints normalized to image dims
  const auto& chosen = detections[0];
  for (size_t i = 0; i < chosen.keypoints.size(); i += 2) {
    double nx = chosen.keypoints[i];
    double ny = chosen.keypoints[i+1];
    // Map from model input normalized coords to original image normalized coords
    // This PoC assumes letterbox with simple scale used in convertRGBAtoInputTensor; compute inverse transform
    double scaleW = double(inW) / width;
    double scaleH = double(inH) / height;
    double scale = std::min(scaleW, scaleH);
    int newW = std::max(1, int(std::round(width * scale)));
    int newH = std::max(1, int(std::round(height * scale)));
    double padX = (inW - newW) / 2.0;
    double padY = (inH - newH) / 2.0;
    double x_in_model = nx * inW;
    double y_in_model = ny * inH;
    double x_src = (x_in_model - padX) / scale;
    double y_src = (y_in_model - padY) / scale;
    landmarks.push_back((float)(x_src / width));
    landmarks.push_back((float)(y_src / height));
  }

  return landmarks;
}

} // namespace gpupixel

#else

// If TFLite not enabled, provide empty stub implementation
#include <vector>
namespace gpupixel {
std::vector<float> RunTFLiteFaceDetector(const uint8_t* data,
                                         int width,
                                         int height,
                                         int stride,
                                         int fmt,
                                         int type) {
  // TFLite detector not enabled at compile time
  return {};
}
}

#endif
