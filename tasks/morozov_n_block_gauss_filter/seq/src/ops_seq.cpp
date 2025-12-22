#include "morozov_n_block_gauss_filter/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "morozov_n_block_gauss_filter/common/include/common.hpp"
#include "ops_seq.hpp"

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
  int width = std::get<1>(GetInput());
  int height = std::get<2>(GetInput());
  std::vector<uint8_t> src = std::get<0>(GetInput());

  std::vector<uint8_t> res(src.size(), 0);
  for(int y = 0;y < height ; y++) {
    for(int x = 0; x < width; x++) {
      Color new_color =  CalculatePixelColor(src, x, y, width, height);
      res[3 * ((y * width) + x) + 0] = new_color.r;
      res[3 * ((y * width) + x) + 1] = new_color.g;
      res[3 * ((y * width) + x) + 2] = new_color.b;
    }
  }

  GetOutput() = res;
  return true;
}

bool MorozovNBlockGaussFilterSEQ::PostProcessingImpl() {
  return true;
}

Color MorozovNBlockGaussFilterSEQ::CalculatePixelColor(std::vector<uint8_t> src, int x, int y, int width , int height) {
  Color res {0, 0, 0};
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;

  const uint8_t ch_max = 255;
  const uint8_t ch_min = 0;
  const int rad_x = 1;
  const int rad_y = 1;

  for(int l = -1; l <= rad_y; l++) {
    for(int k = -1; k <= rad_x; k++) {
      int idX = std::clamp(x + k, 0, width - 1);
      int idY = std::clamp(y + l, 0, height - 1);

      r = static_cast<float>(src[3 * ((idY * width) + idX) + 0]) * kernel[k + rad_x][l + rad_y] / 0.0625f;
      g = static_cast<float>(src[3 * ((idY * width) + idX) + 1]) * kernel[k + rad_x][l + rad_y] / 0.0625f;
      b = static_cast<float>(src[3 * ((idY * width) + idX) + 2]) * kernel[k + rad_x][l + rad_y] / 0.0625f;
    }
  }

  res.r = std::clamp(static_cast<uint8_t>(r), ch_min, ch_max);
  res.g = std::clamp(static_cast<uint8_t>(g), ch_min, ch_max);
  res.b = std::clamp(static_cast<uint8_t>(b), ch_min, ch_max);
  return res;
}

}  // namespace morozov_n_block_gauss_filter
