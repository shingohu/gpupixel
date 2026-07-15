#pragma once

#include <vector>
#include <cstdint>

namespace gpupixel {

std::vector<float> RunTFLiteFaceDetector(const uint8_t* data,
                                         int width,
                                         int height,
                                         int stride,
                                         int fmt,
                                         int type);

} // namespace gpupixel
