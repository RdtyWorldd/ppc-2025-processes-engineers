#include "morozov_n_block_gauss_filter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "morozov_n_block_gauss_filter/common/include/common.hpp"

namespace morozov_n_block_gauss_filter {

MorozovNBlockGaussFilterMPI::MorozovNBlockGaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool MorozovNBlockGaussFilterMPI::ValidationImpl() {
  return true;
}

bool MorozovNBlockGaussFilterMPI::PreProcessingImpl() {
  return true;
}

bool MorozovNBlockGaussFilterMPI::RunImpl() {
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int in_width = 0;
  int in_height = 0;
  int tiles_count = 0;
  std::vector<uint8_t> tiles_data;
  std::vector<int> proc_tile_count;
  int tiles_data_size = 0;
  std::vector<int> tiles_attr;
  int tile_w = 0;
  int tile_h = 0;
  int cols = 0;
  int rows = 0;

  if (rank == 0) {
    const std::vector<uint8_t> &src = std::get<0>(GetInput());
    int width = std::get<1>(GetInput());
    int height = std::get<2>(GetInput());
    in_width = width;
    in_height = height;

    tile_w = CalculateTileSize(width, height, mpi_size);
    tile_h = tile_w;
    cols = GetTileColsRowsCount(width, tile_w);
    rows = GetTileColsRowsCount(height, tile_h);

    int global_tile_count = cols * rows;
    int tile_count_per_proc = global_tile_count / mpi_size;
    int rem = global_tile_count % mpi_size;

    proc_tile_count.resize(mpi_size, tile_count_per_proc);
    for (int i = 0; i < rem; i++) {
      proc_tile_count[i]++;
    }

    std::tuple<std::vector<uint8_t>, std::vector<int>> tiles =
        ParseImageToTiles(src, width, height, rows, cols, tile_w, tile_h);
    tiles_data = std::get<0>(tiles);
    tiles_attr = std::get<1>(tiles);

    ScatterTiles(proc_tile_count, tiles_data, tiles_attr, mpi_size);

    tiles_count = proc_tile_count[0];
    tiles_data_size = GetTilesDataSize(tiles_attr, 0, tiles_count);
  } else {
    ReceiveTiles(tiles_count, tiles_attr, tiles_data_size, tiles_data);
  }

  // Process tiles
  std::vector<uint8_t> res(tiles_data_size, 0);
  int real_res_size = ProcessTiles(tiles_data, tiles_attr, tiles_count, res);

  // Gather results
  std::vector<int> recv_counts(mpi_size, 0);
  std::vector<int> recv_displ(mpi_size, 0);

  if (rank == 0) {
    PrepareGatherData(rank, mpi_size, proc_tile_count, tiles_attr, recv_counts, recv_displ);
    int total_size = 0;
    for (int count : recv_counts) {
      total_size += count;
    }
    new_img_.resize(total_size);
  }

  MPI_Gatherv(res.data(), real_res_size, MPI_BYTE, (rank == 0) ? new_img_.data() : nullptr, recv_counts.data(),
              recv_displ.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

  // Merge tiles back into final image
  if (rank == 0) {
    new_img_ = SimpleMergeTiles(new_img_, tiles_attr, in_width, in_height, tile_w, tile_h, cols, rows);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool MorozovNBlockGaussFilterMPI::PostProcessingImpl() {
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int new_img_size = 0;
  if (rank == 0) {
    new_img_size = static_cast<int>(new_img_.size());
  }
  MPI_Bcast(&new_img_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    new_img_.resize(new_img_size, 0);
  }
  MPI_Bcast(new_img_.data(), new_img_size, MPI_BYTE, 0, MPI_COMM_WORLD);

  GetOutput() = new_img_;
  return true;
}

std::tuple<std::vector<uint8_t>, std::vector<int>> MorozovNBlockGaussFilterMPI::ParseImageToTiles(
    const std::vector<uint8_t> &src, int width, int height, int rows, int cols, int tile_w, int tile_h) {
  int shifts_len = 6 * cols * rows;
  std::vector<int> whrlud(shifts_len, 0);
  std::vector<uint8_t> tiles_imgs;

  int tile_ind_shifts = 0;
  for (int id_y = 0; id_y < rows; id_y++) {
    for (int id_x = 0; id_x < cols; id_x++) {
      int x = id_x * tile_w;
      int y = id_y * tile_h;
      int t_width = std::min(tile_w, width - x);
      int t_height = std::min(tile_h, height - y);

      // Determine border sizes
      int l = (x == 0) ? 0 : 1;
      int r = (x + t_width == width) ? 0 : 1;
      int u = (y == 0) ? 0 : 1;
      int d = (y + t_height == height) ? 0 : 1;

      whrlud[tile_ind_shifts + 0] = t_width;
      whrlud[tile_ind_shifts + 1] = t_height;
      whrlud[tile_ind_shifts + 2] = r;
      whrlud[tile_ind_shifts + 3] = l;
      whrlud[tile_ind_shifts + 4] = u;
      whrlud[tile_ind_shifts + 5] = d;
      tile_ind_shifts += 6;

      // Allocate space for this tile with borders
      int w_over = t_width + r + l;
      int h_over = t_height + u + d;
      int tile_start = static_cast<int>(tiles_imgs.size());
      tiles_imgs.resize(tiles_imgs.size() + static_cast<size_t>(w_over * h_over * 3), 0);

      // Copy tile data with borders
      CopyTileWithBorders(src, width, height, x, y, t_width, t_height, l, r, u, d, w_over, h_over, tile_start,
                          tiles_imgs);
    }
  }
  return std::make_tuple(tiles_imgs, whrlud);
}

int MorozovNBlockGaussFilterMPI::CalculateTileSize(int width, int height, int mpi_size) {
  if (mpi_size <= 0 || width <= 0 || height <= 0) {
    return 1;
  }
  int img_sqr = width * height;
  int tiles_sqr_per_proc = img_sqr / mpi_size;
  if (tiles_sqr_per_proc <= 0) {
    return 1;
  }
  int tile_wh = static_cast<int>(round(sqrt(static_cast<double>(tiles_sqr_per_proc))));
  return std::max(1, tile_wh);
}

int MorozovNBlockGaussFilterMPI::GetTileColsRowsCount(int src_wh, int tile_wh) {
  int count = src_wh / tile_wh;
  if (src_wh % tile_wh != 0) {
    count++;
  }
  return count;
}

int MorozovNBlockGaussFilterMPI::GetTilesDataSize(const std::vector<int> &shifts, int tile_start_id, int tiles_count) {
  int tiles_data_size = 0;
  int tile_displ = tile_start_id;
  for (int i = 0; i < tiles_count; i++) {
    int w = shifts[(tile_displ * 6) + 0];
    int h = shifts[(tile_displ * 6) + 1];
    int r = shifts[(tile_displ * 6) + 2];
    int l = shifts[(tile_displ * 6) + 3];
    int u = shifts[(tile_displ * 6) + 4];
    int d = shifts[(tile_displ * 6) + 5];
    tiles_data_size += (w + r + l) * (h + u + d);
    tile_displ++;
  }
  return 3 * tiles_data_size;
}

void MorozovNBlockGaussFilterMPI::ScatterTiles(const std::vector<int> &proc_tile_count,
                                               const std::vector<uint8_t> &tiles_imgs, const std::vector<int> &shifts,
                                               int mpi_size) {
  int img_displ = GetTilesDataSize(shifts, 0, proc_tile_count[0]);
  int tile_displ = proc_tile_count[0];

  const int req_size = (mpi_size - 1) * 3;
  std::vector<MPI_Request> requests(req_size);
  std::vector<MPI_Status> statuses(req_size);

  for (int i = 1; i < mpi_size; i++) {
    int req_ind = (i - 1) * 3;
    MPI_Isend(&proc_tile_count[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[req_ind]);

    int shifts_displ = tile_displ * 6;
    int shifts_send_count = proc_tile_count[i] * 6;
    MPI_Isend(shifts.data() + shifts_displ, shifts_send_count, MPI_INT, i, 1, MPI_COMM_WORLD, &requests[req_ind + 1]);

    int img_send_count = GetTilesDataSize(shifts, tile_displ, proc_tile_count[i]);
    MPI_Isend(tiles_imgs.data() + img_displ, img_send_count, MPI_BYTE, i, 2, MPI_COMM_WORLD, &requests[req_ind + 2]);

    tile_displ += proc_tile_count[i];
    img_displ += img_send_count;
  }

  MPI_Waitall(req_size, requests.data(), statuses.data());
}

std::vector<uint8_t> MorozovNBlockGaussFilterMPI::SimpleMergeTiles(const std::vector<uint8_t> &tiles_data,
                                                                   const std::vector<int> &tiles_attr, int image_width,
                                                                   int image_height, int tile_w, int tile_h, int cols,
                                                                   int rows) {
  std::vector<uint8_t> image(static_cast<size_t>(image_width * image_height * 3), 0);
  int tile_data_offset = 0;

  // Tiles are stored in row-major order (same as created in ParseImageToTiles)
  for (int id_y = 0; id_y < rows; id_y++) {
    for (int id_x = 0; id_x < cols; id_x++) {
      int tile_idx = (id_y * cols) + id_x;
      if (std::cmp_greater_equal(tile_idx, static_cast<int>(tiles_attr.size() / 6))) {
        break;
      }

      int attr_idx = tile_idx * 6;
      int tile_width = tiles_attr[attr_idx + 0];
      int tile_height = tiles_attr[attr_idx + 1];

      // Calculate original tile position in the image
      int tile_start_x = id_x * tile_w;
      int tile_start_y = id_y * tile_h;

      // Copy tile data to the correct position in the image
      CopyTileToImage(tiles_data, tile_data_offset, tile_width, tile_height, tile_start_x, tile_start_y, image_width,
                      image_height, image);

      tile_data_offset += 3 * tile_width * tile_height;
    }
  }

  return image;
}

Color MorozovNBlockGaussFilterMPI::CalculatePixelColor(const uint8_t *src, int x, int y, int width, int height) const {
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

int MorozovNBlockGaussFilterMPI::ProcessTiles(const std::vector<uint8_t> &tiles_data, const std::vector<int> &tiles_attr,
                                              int tiles_count, std::vector<uint8_t> &res) {
  int tiles_data_displ = 0;
  int res_displ = 0;

  for (int k = 0; k < tiles_count; k++) {
    int tile_ind = k * 6;
    int w = tiles_attr[tile_ind + 0];
    int h = tiles_attr[tile_ind + 1];
    int r = tiles_attr[tile_ind + 2];
    int l = tiles_attr[tile_ind + 3];
    int u = tiles_attr[tile_ind + 4];
    int d = tiles_attr[tile_ind + 5];

    int w_over = (w + r + l);
    int h_over = (h + u + d);

    // Process only the valid region (excluding borders used for filtering)
    for (int row_idx = u; row_idx < (h_over - d); row_idx++) {
      for (int col_idx = l; col_idx < (w_over - r); col_idx++) {
        int res_x = col_idx - l;
        int res_y = row_idx - u;
        int pix_id = res_displ + (3 * ((res_y * w) + res_x));
        Color new_color = CalculatePixelColor(tiles_data.data() + tiles_data_displ, col_idx, row_idx, w_over, h_over);
        res[pix_id + 0] = new_color.r;
        res[pix_id + 1] = new_color.g;
        res[pix_id + 2] = new_color.b;
      }
    }
    tiles_data_displ += 3 * w_over * h_over;
    res_displ += 3 * w * h;
  }
  return res_displ;
}

void MorozovNBlockGaussFilterMPI::ReceiveTiles(int &tiles_count, std::vector<int> &tiles_attr, int &tiles_data_size,
                                                std::vector<uint8_t> &tiles_data) {
  MPI_Recv(&tiles_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  tiles_attr.resize(static_cast<size_t>(tiles_count) * 6);
  MPI_Recv(tiles_attr.data(), static_cast<int>(tiles_attr.size()), MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  tiles_data_size = GetTilesDataSize(tiles_attr, 0, tiles_count);
  tiles_data.resize(tiles_data_size);
  MPI_Recv(tiles_data.data(), tiles_data_size, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void MorozovNBlockGaussFilterMPI::PrepareGatherData(int /* rank */, int mpi_size, const std::vector<int> &proc_tile_count,
                                                     const std::vector<int> &tiles_attr, std::vector<int> &recv_counts,
                                                     std::vector<int> &recv_displ) {
  int tile_count_displ = 0;
  for (int i = 0; i < mpi_size; i++) {
    for (int j = 0; j < proc_tile_count[i]; j++) {
      int attr_ind = (tile_count_displ + j) * 6;
      int w = tiles_attr[attr_ind + 0];
      int h = tiles_attr[attr_ind + 1];
      recv_counts[i] += h * w;
    }
    tile_count_displ += proc_tile_count[i];
    recv_counts[i] = recv_counts[i] * 3;
  }
  for (int i = 1; i < mpi_size; i++) {
    recv_displ[i] = recv_displ[i - 1] + recv_counts[i - 1];
  }
}

void MorozovNBlockGaussFilterMPI::CopyTileWithBorders(const std::vector<uint8_t> &src, int width, int height, int x,
                                                       int y, int t_width, int t_height, int l, int r, int u, int d,
                                                       int w_over, int /* h_over */, int tile_start,
                                                       std::vector<uint8_t> &tiles_imgs) {
  for (int j = -u; j < t_height + d; j++) {
    for (int i = -l; i < t_width + r; i++) {
      int src_x = std::clamp(i + x, 0, width - 1);
      int src_y = std::clamp(j + y, 0, height - 1);
      int src_pix_id = 3 * ((src_y * width) + src_x);
      int tile_pix_id = tile_start + (3 * (((j + u) * w_over) + (i + l)));
      tiles_imgs[tile_pix_id + 0] = src[src_pix_id + 0];
      tiles_imgs[tile_pix_id + 1] = src[src_pix_id + 1];
      tiles_imgs[tile_pix_id + 2] = src[src_pix_id + 2];
    }
  }
}

void MorozovNBlockGaussFilterMPI::CopyTileToImage(const std::vector<uint8_t> &tiles_data, int tile_data_offset,
                                                   int tile_width, int tile_height, int tile_start_x, int tile_start_y,
                                                   int image_width, int image_height, std::vector<uint8_t> &image) {
  for (int row_idx = 0; row_idx < tile_height; row_idx++) {
    for (int col_idx = 0; col_idx < tile_width; col_idx++) {
      int image_x = tile_start_x + col_idx;
      int image_y = tile_start_y + row_idx;

      if (image_x >= image_width || image_y >= image_height) {
        continue;
      }

      int tile_pixel_idx = tile_data_offset + (3 * ((row_idx * tile_width) + col_idx));
      int image_pixel_idx = 3 * ((image_y * image_width) + image_x);

      image[image_pixel_idx + 0] = tiles_data[tile_pixel_idx + 0];
      image[image_pixel_idx + 1] = tiles_data[tile_pixel_idx + 1];
      image[image_pixel_idx + 2] = tiles_data[tile_pixel_idx + 2];
    }
  }
}
}  // namespace morozov_n_block_gauss_filter
