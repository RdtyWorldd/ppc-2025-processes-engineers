#pragma once

#include <array>
#include <cstdint>
#include <tuple>
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

  const std::array<std::array<float, 3>, 3> kernel_ = {{{1.0F, 2.0F, 1.0F}, {2.0F, 4.0F, 2.0F}, {1.0F, 2.0F, 1.0F}}};
  std::vector<uint8_t> new_img_;

  Color CalculatePixelColor(const uint8_t *src, int x, int y, int width, int height) const;
  static std::tuple<std::vector<uint8_t>, std::vector<int>> ParseImageToTiles(const std::vector<uint8_t> &src,
                                                                              int width, int height, int rows, int cols,
                                                                              int tile_w, int tile_h);
  static void CopyTileWithBorders(const std::vector<uint8_t> &src, int width, int height, int x, int y, int t_width,
                                  int t_height, int l, int r, int u, int d, int w_over, int /* h_over */,
                                  int tile_start, std::vector<uint8_t> &tiles_imgs);
  static int CalculateTileSize(int width, int height, int mpi_size);
  static int GetTileColsRowsCount(int src_wh, int tile_wh);
  static int GetTilesDataSize(const std::vector<int> &shifts, int tile_start_id, int tiles_count);
  void ScatterTiles(const std::vector<int> &proc_tile_count, const std::vector<uint8_t> &tiles_imgs,
                    const std::vector<int> &shifts, int mpi_size);
  static std::vector<uint8_t> SimpleMergeTiles(const std::vector<uint8_t> &tiles_data,
                                               const std::vector<int> &tiles_attr, int image_width, int image_height,
                                               int tile_w, int tile_h, int cols, int rows);
  static void CopyTileToImage(const std::vector<uint8_t> &tiles_data, int tile_data_offset, int tile_width,
                              int tile_height, int tile_start_x, int tile_start_y, int image_width, int image_height,
                              std::vector<uint8_t> &image);
  int ProcessTiles(const std::vector<uint8_t> &tiles_data, const std::vector<int> &tiles_attr, int tiles_count,
                   std::vector<uint8_t> &res);
  void ReceiveTiles(int &tiles_count, std::vector<int> &tiles_attr, int &tiles_data_size,
                    std::vector<uint8_t> &tiles_data);
  void PrepareGatherData(int /* rank */, int mpi_size, const std::vector<int> &proc_tile_count,
                         const std::vector<int> &tiles_attr, std::vector<int> &recv_counts,
                         std::vector<int> &recv_displ);
};

}  // namespace morozov_n_block_gauss_filter
