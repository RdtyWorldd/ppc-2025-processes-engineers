#include "morozov_n_block_gauss_filter/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

  std::vector<uint8_t> res(static_cast<size_t>(width * height * 3), 0);
  for (int row_idx = 0; row_idx < height; row_idx++) {
    for (int col_idx = 0; col_idx < width; col_idx++) {
      Color new_color = CalculatePixelColor(src, col_idx, row_idx, width, height);
      int pix_idx = 3 * ((row_idx * width) + col_idx);
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
                                                       int height) const {
  constexpr uint8_t kChMax = 255;
  constexpr uint8_t kChMin = 0;
  constexpr int kRadX = 1;
  constexpr int kRadY = 1;
  constexpr float kKernelSum = 16.0F;
  constexpr float kKernelInv = 1.0F / kKernelSum;

  float r = 0.0F;
  float g = 0.0F;
  float b = 0.0F;

  for (int row_offset = -kRadY; row_offset <= kRadY; row_offset++) {
    for (int col_offset = -kRadX; col_offset <= kRadX; col_offset++) {
      int id_x = std::clamp(x + col_offset, 0, width - 1);
      int id_y = std::clamp(y + row_offset, 0, height - 1);
      int pix_id = 3 * ((id_y * width) + id_x);
      const int kernel_row = row_offset + kRadY;
      const int kernel_col = col_offset + kRadX;
      float kernel_val = kernel_.at(static_cast<size_t>(kernel_row)).at(static_cast<size_t>(kernel_col)) * kKernelInv;
      r += static_cast<float>(src[pix_id + 0]) * kernel_val;
      g += static_cast<float>(src[pix_id + 1]) * kernel_val;
      b += static_cast<float>(src[pix_id + 2]) * kernel_val;
    }
  }

  Color res{};
  res.r = std::clamp(static_cast<uint8_t>(r), kChMin, kChMax);
  res.g = std::clamp(static_cast<uint8_t>(g), kChMin, kChMax);
  res.b = std::clamp(static_cast<uint8_t>(b), kChMin, kChMax);
  return res;
}

}  // namespace morozov_n_block_gauss_filter
