#include "morozov_n_block_gauss_filter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "morozov_n_block_gauss_filter/common/include/common.hpp"
#include "ops_mpi.hpp"

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
  int tiles_count = 0;
  std::vector<uint8_t> tiles_data;
  std::vector<int> tiles_attr;

  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (rank == 0) {
    std::vector<uint8_t> src = std::get<0>(GetInput());
    width = std::get<1>(GetInput());
    height = std::get<2>(GetInput());

    int tile_w = CalculateTileSize(src, width, height);
    int tile_h = tile_w;
    int cols = GetTileColsRowsCount(width, tile_w);
    int rows = GetTileColsRowsCount(height, tile_h);

    int tile_count = cols * rows;
    int tile_count_per_proc = tile_count / mpi_size;
    int rem = tile_count % mpi_size;

    std::vector<int> proc_tile_count(mpi_size, tile_count_per_proc);
    for (int i = 0; i < rem; i++) {
      proc_tile_count[i]++;
    }

    //добавить ссылки
    std::tuple<std::vector<uint8_t>, std::vector<int>> tiles =
        ParseImageToTiles(src, width, height, rows, cols, tile_w, tile_h);
    tiles_data = std::get<0>(tiles);
    tiles_attr = std::get<1>(tiles);

    ScatterTiles(tiles_data, tiles_attr, mpi_size);
  } else {
    MPI_Recv(&tiles_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tiles_attr.resize(tiles_count * 6);
    MPI_Recv(tiles_attr.data(), static_cast<int>(tiles_attr.size()), MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int tiles_data_size = GetTilesDataSize(tiles_attr, 0, tiles_count);
    tiles_data.resize(tiles_data_size);
    Mpi_Recv(tiles_data.data(), tiles_data_size, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  std::vector<uint8_t> res(tiles_data.size(), 0);
  int tiles_data_displ = 0; //смещение по данным тайла (изображениям)
  int res_displ = 0; // смещение по записанным новым изображениям
  for(int k = 0; k < tiles_count; i++) {
    int w = tiles_attr[(k * 6) + 0];
    int h = tiles_attr[(k * 6) + 1];
    int r = tiles_attr[(k * 6) + 2];
    int l = tiles_attr[(k * 6) + 3];
    int u = tiles_attr[(k * 6) + 4];
    int d  = tiles_attr[(k * 6) + 5];

    int w_over = (w + r + l);
    int h_over = (h + u + d);
    for(int y = u; y < (h - d); y++) {
      for(int x = l; x < (w - r); x++) {
        int res_x = x - l;
        int res_y = y - u;
        int pix_id = res_displ + (3 * ((res_y * w) + res_x));
        Color new_color =  CalculatePixelColor(tiles_data.data() + tiles_data_displ, x, y, w_over, h_over);
        res[pix_id + 0] = new_color.r;
        res[pix_id + 1] = new_color.g;
        res[pix_id + 2] = new_color.b;
      }
    }
    tiles_data_displ += 3 * w_over * h_over;
    res_displ += 3 * w * h;
  }

  //сбор данных на 0
  return true;
}

bool MorozovNBlockGaussFilterMPI::PostProcessingImpl() {
  return true;
}

std::tuple<std::vector<uint8_t>, std::vector<int>> MorozovNBlockGaussFilterMPI::ParseImageToTiles(
    std::vector<uint8_t> &src, int width, int height, int rows, int cols, int tile_w, int tile_h) {
  int len = 3 * (tile_w + 1) * (tile_h + 1) * cols * rows;
  int shifts_len = 6 * cols * rows;
  std::vector<uint8_t> tiles_imgs(len, 0);  // вектор изображений для всех тайлов
  // вектор значний тайла h- выссота w - ширина rlud - доп границы со всех сторон
  std::vector<int> whrlud(shifts_len, 0);

  int tile_ind_img = 0;
  int tile_ind_shfits = 0;
  for (int idY = 0; idY < rows; idY++) {
    for (int idX = 0; idX < cols; idX++) {
      int x = idX * tile_w;
      int y = idY * tile_h;
      int t_width = std::min(tile_w, width - x);
      int t_height = std::min(tile_h, height - y);

      int r = 1, l = 1;
      int up = 1, down = 1;
      if (x == 0) {  // Левый  Край
        l = 0
      }
      if (x + t_width == width) {  // Правый край
        r = 0;
      }
      if (y == 0) {  // Верхний край
        up = 0;
      }
      if (y + t_height == height) {  // Нижний край
        down = 0;
      }
      whrlud[tile_ind_shfits] = t_width;
      whrlud[tile_ind_shfits + 1] = t_height;
      whrlud[tile_ind_shfits + 2] = r;
      whrlud[tile_ind_shfits + 3] = l;
      whrlud[tile_ind_shfits + 4] = up;
      whrlud[tile_ind_shfits + 5] = down;
      tile_ind_shfits += 6;
      // t.img = std::vector<uint8_t>((t_width + r + l) * (t_height + up + down) * 3);
      // заполнение изображения тайла
      for (int j = -up; j < t.height + down; j++) {
        for (int i = -l; i < t.weight + r; i++) {
          int src_x = i + x;
          int src_y = j + y;
          tiles_imgs[tile_ind_img + 0] = src[3 * ((src_y * width) + src_x) + 0];
          tiles_imgs[tile_ind_img + 1] = src[3 * ((src_y * width) + src_x) + 1];
          tiles_imgs[tile_ind_img + 2] = src[3 * ((src_y * width) + src_x) + 2];
          tile_ind_img += 3;
        }
      }
    }
  }
  return std::make_tuple(tiles_imgs, whrlud);
}

int MorozovNBlockGaussFilterMPI::CalculateTileSize(std::vector<uint8_t> &src, int width, int height, int mpi_size) {
  int img_sqr = width * height;
  int tiles_sqr_per_proc = img_sqr / mpi_size;
  int tile_wh = static_cast<int>(sqrt(tiles_sqr_per_proc));
  return 0;
}

int MorozovNBlockGaussFilterMPI::GetTileColsRowsCount(int src_wh, int tile_wh) {
  int count = src_wh / tile_wh;
  if (src_wh % tile_wh != 0) {
    count++;
  }
  return count;
}

int MorozovNBlockGaussFilterMPI::GetTilesDataSize(std::vector<int> &shifts, int tile_start_id, int tiles_count) {
  int tiles_data_size = 0;
  int tile_displ = tile_start_id;
  for(int i = 0; i < tiles_count; i++) {
    img_send_count += (shifts[(tile_displ * 6) + 0] + shifts[(tile_displ * 6) + 2] + shifts[(tile_displ * 6) + 3]) *
                        (shifts[(tile_displ * 6) + 1] + shifts[(tile_displ * 6) + 4] + shifts[(tile_displ * 6) + 5]);
    tile_displ++;
  }
  return tiles_data_size;
}

void MorozovNBlockGaussFilterMPI::ScatterTiles(std::vector<uint8_t> &tiles_imgs, std::vector<int> &shifts,
                                               int mpi_size) {
  int img_displ = 0; //сколько бит из вектора с изображениями всех тайлов отправлено
  int tile_displ = 0; //сколько тайлов посчитано
  MPI_Request requests[(mpi_size - 1) * 3];
  MPI_Status statuses[(mpi_size - 1) * 3];
  for (int i = 1; i < mpi_size; i++) {
    MPI_ISend(&proc_tile_count[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, requests + (i * 3));
    // расчет смещения для количества тайлов
    int shifts_displ = tile_displ * 6;
    // расчет отправляемых данных;
    int shifts_send_count = proc_tile_count[i] * 6;
    int img_send_count = 0;
    MPI_ISend(shifts.data() + shifts_displ, shifts_send_count, MPI_INT, i, 1, MPI_COMM_WORLD, requests + ((i * 3) + 1));
    // for (int j = 0; j < proc_tile_count[i]; j++) {
    //   img_send_count += (shifts[(tile_displ * 6) + 0] + shifts[(tile_displ * 6) + 2] + shifts[(tile_displ * 6) + 3]) *
    //                     (shifts[(tile_displ * 6) + 1] + shifts[(tile_displ * 6) + 4] + shifts[(tile_displ * 6) + 5]);
    //   tile_displ++;
    // }
    img_send_count = 3 * GetTilesDataSize(shifts, tile_displ, proc_tile_count[i]);  // усножаем так как 3 цвета
    MPI_ISend(tiles_imgs.data() + img_displ, img_send_count, MPI_BYTE, i, 2, MPI_COMM_WORLD, requests + ((i * 3) + 1));
    tile_displ += proc_tile_count[i];
    img_displ += img_send_count;
  }

  MPI_Waitall((mpi_size - 1) * 3, requests, statuses);
  return true;
}

void MorozovNBlockGaussFilterMPI::SendTiles() {}

Color MorozovNBlockGaussFilterMPI::CalculatePixelColor(uint8_t *src, int x, int y, int width, int height) {
  Color res{0, 0, 0};
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;

  const uint8_t ch_max = 255;
  const uint8_t ch_min = 0;
  const int rad_x = 1;
  const int rad_y = 1;

  for (int l = -1; l <= rad_y; l++) {
    for (int k = -1; k <= rad_x; k++) {
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
