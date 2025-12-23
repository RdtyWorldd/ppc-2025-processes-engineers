#pragma once

#include <array>
#include <cstdint>
#include <tuple>
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

  const std::array<std::array<float, 3>, 3> kernel_ = {{{1.0F, 2.0F, 1.0F}, {2.0F, 4.0F, 2.0F}, {1.0F, 2.0F, 1.0F}}};
  Color CalculatePixelColor(const std::vector<uint8_t> &src, int x, int y, int width, int height) const;
};
}  // namespace morozov_n_block_gauss_filter
