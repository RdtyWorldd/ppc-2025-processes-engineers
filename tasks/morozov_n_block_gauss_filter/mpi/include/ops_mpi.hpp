#pragma once

#include <cstddef>
#include <vector>

#include "morozov_n_block_gauss_filter/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozov_n_block_gauss_filter {

class MorozovNBlockGaussFilterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MorozovNBlockGaussFilterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  const float kernel[3][3] = {{1.0f, 2.0f, 1.0f}, {2.0f, 4.0f, 2.0f}, {1.0f, 2.0f, 1.0f}};
  Color CalculatePixelColor(std::vector<uint8_t> &src, int x, int y, int width, int height);
  std::vector<Tile> ParseImageToTiles(std::vector<uint8_t> &src, int width, int height, 
    int rows, int cols, int tile_w, int tile_h);
  void SendTiles();
};

}  // namespace morozov_n_block_gauss_filter
