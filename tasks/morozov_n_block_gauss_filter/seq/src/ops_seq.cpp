#include "morozov_n_block_gauss_filter/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "morozov_n_block_gauss_filter/common/include/common.hpp"

namespace morozov_n_block_gauss_filter {

MorozovNBlockGaussFilterSEQ::MorozovNBlockGaussFilterSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MorozovNBlockGaussFilterSEQ::ValidationImpl() {
  return true;
}

bool MorozovNBlockGaussFilterSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovNBlockGaussFilterSEQ::RunImpl() {
  const std::vector<uint8_t> &src = std::get<0>(GetInput());
  int width = std::get<1>(GetInput());
  int height = std::get<2>(GetInput());

  std::vector<uint8_t> res(width * height * 3, 0);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      Color new_color = CalculatePixelColor(src, x, y, width, height);
      int pix_idx = 3 * ((y * width) + x);
      res[pix_idx + 0] = new_color.r;
      res[pix_idx + 1] = new_color.g;
      res[pix_idx + 2] = new_color.b;
    }
  }
  GetOutput() = res;
  return true;
}

bool MorozovNBlockGaussFilterSEQ::PostProcessingImpl() {
  return true;
}

Color MorozovNBlockGaussFilterSEQ::CalculatePixelColor(const std::vector<uint8_t> &src, int x, int y, int width,
                                                       int height) {
  constexpr uint8_t ch_max = 255;
  constexpr uint8_t ch_min = 0;
  constexpr int rad_x = 1;
  constexpr int rad_y = 1;
  // Calculate kernel sum once (1+2+1+2+4+2+1+2+1 = 16, so normalization is 1/16 = 0.0625)
  constexpr float kernel_sum = 16.0f;
  constexpr float kernel_inv = 1.0f / kernel_sum;

  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;

  for (int l = -rad_y; l <= rad_y; l++) {
    for (int k = -rad_x; k <= rad_x; k++) {
      int idX = std::clamp(x + k, 0, width - 1);
      int idY = std::clamp(y + l, 0, height - 1);
      int pix_id = 3 * ((idY * width) + idX);
      float kernel_val = kernel[l + rad_y][k + rad_x] * kernel_inv;
      r += static_cast<float>(src[pix_id + 0]) * kernel_val;
      g += static_cast<float>(src[pix_id + 1]) * kernel_val;
      b += static_cast<float>(src[pix_id + 2]) * kernel_val;
    }
  }

  Color res;
  res.r = std::clamp(static_cast<uint8_t>(r), ch_min, ch_max);
  res.g = std::clamp(static_cast<uint8_t>(g), ch_min, ch_max);
  res.b = std::clamp(static_cast<uint8_t>(b), ch_min, ch_max);
  return res;
}

}  // namespace morozov_n_block_gauss_filter
