#pragma once

#include <cstddef>
#include <vector>

#include "morozov_n_block_gauss_filter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozov_n_block_gauss_filter {

class MorozovNBlockGaussFilterSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MorozovNBlockGaussFilterSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  const float kernel[3][3] = {{1.0f, 2.0f, 1.0f}, {2.0f, 4.0f, 2.0f}, {1.0f, 2.0f, 1.0f}};
  Color CalculatePixelColor(std::vector<uint8_t> src, int x, int y, int width, int height);
  void print_pic(int h, int w, int c, uint8_t *img);
};
}  // namespace morozov_n_block_gauss_filter
