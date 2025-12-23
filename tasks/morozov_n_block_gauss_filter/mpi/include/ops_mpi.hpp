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
  std::vector<uint8_t> new_img;

  Color CalculatePixelColor(uint8_t *src, int x, int y, int width, int height);
  std::tuple<std::vector<uint8_t>, std::vector<int>> ParseImageToTiles(const std::vector<uint8_t> &src, int width,
                                                                       int height, int rows, int cols, int tile_w,
                                                                       int tile_h);
  int CalculateTileSize(int width, int height, int mpi_size);
  int GetTileColsRowsCount(int src_wh, int tile_wh);
  int GetTilesDataSize(const std::vector<int> &shifts, int tile_start_id, int tiles_count);
  void ScatterTiles(const std::vector<int> &proc_tile_count, const std::vector<uint8_t> &tiles_imgs,
                    const std::vector<int> &shifts, int mpi_size);
  std::vector<uint8_t> SimpleMergeTiles(const std::vector<uint8_t> &tiles_data, const std::vector<int> &tiles_attr,
                                        int image_width, int image_height, int tile_w, int tile_h, int cols, int rows);
};

}  // namespace morozov_n_block_gauss_filter
