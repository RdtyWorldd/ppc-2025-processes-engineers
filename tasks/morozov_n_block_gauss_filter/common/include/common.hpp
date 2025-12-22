#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace morozov_n_block_gauss_filter {
struct Color {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct Tile {
  int x;
  int y;
  int width;
  int height;
  int w_over;
  int h_over;
  std::vector<uint8_t> img;
};

using InType = std::tuple<std::vector<uint8_t>, int, int>;
using OutType = std::vector<uint8_t>;
using TestType = std::tuple<int, std::string, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace morozov_n_block_gauss_filter
