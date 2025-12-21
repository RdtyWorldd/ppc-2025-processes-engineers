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
  int width = 0;
  int height = 0;

  int rank = 0;
  int mpi_size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if(rank == 0) {
    std::vector<uint8_t> src = std::get<0>(GetInput());
    width = std::get<1>(GetInput());
    height = std::get<2>(GetInput());
    int img_sqr = width * height;
    int tiles_sqr_per_proc = img_sqr / mpi_size;

    int tile_w = static_cast<int>(sqrt(tiles_sqr_per_proc));
    int tile_h = tile_w; 
    int tile_square = tile_h * tile_w;

    int cols = width /tile_w;
    int rows = height / tile_h;
    if(width % tile_w != 0) {
      cols++;
    }
    if(height % tile_h != 0) {
      height++;
    }
    int tile_count = cols * rows;
    int tile_count_per_proc = tile_count / mpi_size;
    int rem = tile_count % mpi_size;

    std::vector<int> proc_tile_count(mpi_size, tile_count_per_proc);
    for(int i = 0; i < rem; i++) {
      proc_tile_count[i]++;
    }

    std::vector<Tile> tiles = ParseImageToTiles(src, width, height, rows, cols, tile_w, tile_h); 
    //пересылка тайлов
    for(int i = 1; i < mpi_size; i++) {
      MPI_ISend(proc_tile_count[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, );
      for(int j)
    }
  }
  return true;
}

bool MorozovNBlockGaussFilterMPI::PostProcessingImpl() {
  return true;
}

std::vector<Tile> MorozovNBlockGaussFilterMPI::ParseImageToTiles(std::vector<uint8_t> &src, int width, int height,
                                                                 int rows, int cols, int tile_w, int tile_h) {
  std::vector<Tile> tiles;
  int len = 3 * (tile_w + 1) * (tile_h +1) * cols * rows;
  std::vector<uint8_t> tiles_imgs(len, 0); //вектор изображений для всех тайлов
  std::vector<int> hwrlud(6 * cols * rows, 0); //вектор значний тайла h- выссота w - ширина rlud - доп границы со всех сторон
  //цикл рассчета 
  for(int idY = 0; idY < rows; idY++) {
    for(int idX = 0; idX < cols; idX++) {
      Tile t {0};
      t.x = idX * tile_w;
      t.y = idY * tile_h;
      t.width = std::min(tile_w, width - t.x);
      t.height = std::min(tile_h, height - t.h);
      int r = 1, l = 1;
      int up = 1, down = 1; 
      if (t.x == 0) { //Левый  Край
          l = 0
      }
      if (t.X + t.width == width) { //Правый край
          r = 0;
      }
      if (t.y == 0) { //Верхний край
          up = 0;
      }
      if (t.y + t.height == height) { //Нижний край
          down = 0; 
      }
      t.img = std::vector<uint8_t>((t.width + r + l) * (t.height + up + down) * 3);
      //заполнение изображения тайла
      int tile_ind = 0;
      for(int j = -up; j < t.height + down; j++) {
        for(int i = -l; i < t.weight + r; i++) {
          int src_x = i + t.x;
          int src_y = j + t.y;
          t.img[tile_ind + 0] = src[3 * ((src_y * width) + src_x) + 0];
          t.img[tile_ind + 1] = src[3 * ((src_y * width) + src_x) + 1];
          t.img[tile_ind + 2] = src[3 * ((src_y * width) + src_x) + 2];
          tile_ind += 3;
        }
      }
      tiles.push_back(tile);
    }
  }
  return tiles;
}

void MorozovNBlockGaussFilterMPI::SendTiles() {

}

Color MorozovNBlockGaussFilterMPI::CalculatePixelColor(std::vector<uint8_t> src, int x, int y, int width, int height) {
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
