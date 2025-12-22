#include "morozov_n_block_gauss_filter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "morozov_n_block_gauss_filter/common/include/common.hpp"

namespace morozov_n_block_gauss_filter {

void MorozovNBlockGaussFilterMPI::print_pic(int h, int w, int c, uint8_t *img) {
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      for (int k = 0; k < c; k++) {
        std::cout << (int)img[3 * ((i * w) + j) + k] << " ";
      }
      std::cout << "| ";
    }
    std::cout << "\n";
  }
}

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
  int in_width = 0;
  int in_height = 0;
  int tiles_count = 0;
  std::vector<uint8_t> tiles_data;
  std::vector<int> proc_tile_count;
  int tiles_data_size = 0;
  std::vector<int> tiles_attr;
  std::vector<uint8_t> new_img;
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (rank == 0) {
    std::vector<uint8_t> src = std::get<0>(GetInput());
    int width = std::get<1>(GetInput());
    int height = std::get<2>(GetInput());
    in_height = height;
    in_width = width;

    int tile_w = CalculateTileSize(width, height, mpi_size);
    int tile_h = tile_w;
    int cols = GetTileColsRowsCount(width, tile_w);
    int rows = GetTileColsRowsCount(height, tile_h);

    int global_tile_count = cols * rows;
    int tile_count_per_proc = global_tile_count / mpi_size;
    int rem = global_tile_count % mpi_size;

    proc_tile_count.resize(mpi_size, tile_count_per_proc);
    for (int i = 0; i < rem; i++) {
      proc_tile_count[i]++;
    }

    // debug
    //  {
    //    std::cout << width << " " << height << "\n";
    //    print_pic(height, width, 3, src.data());
    //    std::cout<<"-------\n" << tile_w << " " << tile_h << " cols:" << cols <<" rows:" << rows <<"\n";
    //  }

    // добавить ссылки в кортеже
    std::tuple<std::vector<uint8_t>, std::vector<int>> tiles =
        ParseImageToTiles(src, width, height, rows, cols, tile_w, tile_h);
    tiles_data = std::get<0>(tiles);
    tiles_attr = std::get<1>(tiles);

    // debug
    //  if(rank == 0){
    //    int tile_data_displ = 0;
    //    for(int i = 0; i < tile_count; i++) {
    //      int w = tiles_attr[(i * 6) + 0] + tiles_attr[(i * 6) + 2] + tiles_attr[(i * 6) + 3];
    //      int h = tiles_attr[(i * 6) + 1] + tiles_attr[(i * 6) + 4] + tiles_attr[(i * 6) + 5];
    //      print_pic(h, w, 3, tiles_data.data() + tile_data_displ); //вывод каждого тайла
    //      tile_data_displ += w * h * 3;
    //      std::cout <<"\n";
    //    }
    //  }

    ScatterTiles(proc_tile_count, tiles_data, tiles_attr, mpi_size);

    tiles_count = proc_tile_count[0];
    tiles_data_size = GetTilesDataSize(tiles_attr, 0, tiles_count);
  } else {  // прием данных на остальных процессах
    MPI_Recv(&tiles_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tiles_attr.resize(tiles_count * 6);
    MPI_Recv(tiles_attr.data(), static_cast<int>(tiles_attr.size()), MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    tiles_data_size = GetTilesDataSize(tiles_attr, 0, tiles_count);
    tiles_data.resize(tiles_data_size);
    // debug
    //  {
    //    std::cout << "rank:" << rank << " " <<tiles_count << "\n";
    //    std::cout << "rank:" << rank << " " << "tile_data_size:" << tiles_data_size << "\n";
    //  }
    MPI_Recv(tiles_data.data(), tiles_data_size, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  // MPI_Barrier(MPI_COMM_WORLD);
  // debug print
  //  if(rank == 1) {
  //    std::cout << "in print" << std::endl;
  //    int tile_data_displ = 0;
  //    for(int i = 0; i < tiles_count; i++) {
  //        int w = tiles_attr[(i * 6) + 0] + tiles_attr[(i * 6) + 2] + tiles_attr[(i * 6) + 3];
  //        int h = tiles_attr[(i * 6) + 1] + tiles_attr[(i * 6) + 4] + tiles_attr[(i * 6) + 5];
  //        print_pic(h, w, 3, tiles_data.data() + tile_data_displ); //вывод каждого тайла
  //        tile_data_displ += w * h * 3;
  //        std::cout << "\n";
  //    }
  //    std::cout << std::endl;
  //  }
  std::cout << " rank: " << rank << "t_count:" << tiles_count << "\n";
  std::vector<uint8_t> res(tiles_data_size, 0);
  int tiles_data_displ = 0;  // смещение по данным тайла (изображениям)
  int res_displ = 0;         // смещение по записанным новым изображениям
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
    for (int y = u; y < (h_over - d); y++) {
      for (int x = l; x < (w_over - r); x++) {
        int res_x = x - l;
        int res_y = y - u;
        int pix_id = res_displ + (3 * ((res_y * w) + res_x));
        Color new_color = CalculatePixelColor(tiles_data.data() + tiles_data_displ, x, y, w_over, h_over);
        res[pix_id + 0] = new_color.r;
        res[pix_id + 1] = new_color.g;
        res[pix_id + 2] = new_color.b;
      }
    }
    tiles_data_displ += 3 * w_over * h_over;
    res_displ += 3 * w * h;
  }
  int real_res_size = res_displ;

  // debug
  {
    res_displ = 0;
    for (int i = 0; i < tiles_count; i++) {
      int tile_ind = i * 6;
      int w = tiles_attr[tile_ind + 0];
      int h = tiles_attr[tile_ind + 1];
      print_pic(h, w, 3, res.data() + res_displ);
      std::cout << "\n";
      res_displ += 3 * w * h;
    }
  }
  // сбор данных на 0
  std::vector<int> recv_counts;
  std::vector<int> recv_displ;
  if (rank == 0) {
    recv_counts.resize(mpi_size, 0);
    recv_displ.resize(mpi_size, 0);
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

    int total_size = 0;
    for (int count : recv_counts) {
      total_size += count;
    }
    new_img.resize(total_size);

    // debug
    //  for(int i = 0; i < mpi_size; i++) {
    //    std::cout << "rank: " << i << " exp_size:" << recv_counts[i] << " exp_displ: " << recv_displ[i] <<"\n";
    //  }
    //  std::cout << "\n";
  }
  MPI_Gatherv(res.data(), real_res_size, MPI_BYTE, new_img.data(), recv_counts.data(), recv_displ.data(), MPI_BYTE, 0,
              MPI_COMM_WORLD);

  // для проверки с последовательной версией
  if (rank == 0) {
    //  res_displ = 0;
    //  for(int i = 0; i < tiles_count; i ++) {
    //    int tile_ind = i * 6;
    //    int w = tiles_attr[tile_ind + 0];
    //    int h = tiles_attr[tile_ind + 1];
    //    print_pic(h, w, 3, res.data() + res_displ);
    //    std::cout << "\n";
    //    res_displ += 3 * w * h;
    //  }

    print_pic(in_height, in_width, 3, SimpleMergeTiles(new_img, tiles_attr, in_width, in_height).data());
  }

  MPI_Barrier(MPI_COMM_WORLD);
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
        l = 0;
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
      for (int j = -up; j < t_height + down; j++) {
        for (int i = -l; i < t_width + r; i++) {
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

int MorozovNBlockGaussFilterMPI::CalculateTileSize(int width, int height, int mpi_size) {
  int img_sqr = width * height;
  int tiles_sqr_per_proc = img_sqr / mpi_size;
  int tile_wh = static_cast<int>(round(sqrt(tiles_sqr_per_proc)));
  return tile_wh;
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
  for (int i = 0; i < tiles_count; i++) {
    tiles_data_size += (shifts[(tile_displ * 6) + 0] + shifts[(tile_displ * 6) + 2] + shifts[(tile_displ * 6) + 3]) *
                       (shifts[(tile_displ * 6) + 1] + shifts[(tile_displ * 6) + 4] + shifts[(tile_displ * 6) + 5]);
    // debug
    //  {
    //    std::cout << "tile_statt: "<< tile_start_id << " count: "<< tiles_count <<
    //    " tiles_data_size: "<< tiles_data_size << "\n";
    //  }
    tile_displ++;
  }
  return 3 * tiles_data_size;
}

void MorozovNBlockGaussFilterMPI::ScatterTiles(std::vector<int> &proc_tile_count, std::vector<uint8_t> &tiles_imgs,
                                               std::vector<int> &shifts, int mpi_size) {
  // сколько бит из вектора с изображениями всех тайлов отправлено  (изначально смещено на длину всех тайлов процесса
  // 0);
  int img_displ = GetTilesDataSize(shifts, 0, proc_tile_count[0]);
  int tile_displ = proc_tile_count[0];  // сколько тайлов посчитано

  const int req_size = (mpi_size - 1) * 3;
  MPI_Request *requests = new MPI_Request[req_size];
  MPI_Status *statuses = new MPI_Status[req_size];

  for (int i = 1; i < mpi_size; i++) {
    int req_ind = (i - 1) * 3;  // индекс позиции реквеста
    MPI_Isend(&proc_tile_count[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, requests + req_ind);
    // расчет смещения для количества тайлов
    int shifts_displ = tile_displ * 6;
    // расчет отправляемых данных;
    int shifts_send_count = proc_tile_count[i] * 6;
    int img_send_count = 0;
    MPI_Isend(shifts.data() + shifts_displ, shifts_send_count, MPI_INT, i, 1, MPI_COMM_WORLD, requests + (req_ind + 1));

    img_send_count = GetTilesDataSize(shifts, tile_displ, proc_tile_count[i]);
    MPI_Isend(tiles_imgs.data() + img_displ, img_send_count, MPI_BYTE, i, 2, MPI_COMM_WORLD, requests + (req_ind + 2));
    // debug
    //  {
    //    std::cout << "to_rank: " << i << "\n";
    //    std::cout << "attr_displ: "<< shifts_displ << " attrs_len: " << shifts_send_count << "\n";
    //    std::cout << "imgs_displ:" << img_displ << " img_send_len: " << img_send_count << std::endl;
    //    std::cout << "Sended tiles:\n";
    //    int print_displ = img_displ;
    //    for(int j = 0; j < proc_tile_count[i]; j++) {
    //      int tile_ind = (j + tile_displ) * 6;
    //      int w = shifts[tile_ind + 0] + shifts[tile_ind + 2] + shifts[tile_ind + 3];
    //      int h = shifts[tile_ind + 1] + shifts[tile_ind + 4] + shifts[tile_ind + 5];
    //      std::cout << "tile ind:"<< tile_ind << " w:" << w << " h:" << h << std::endl;
    //      print_pic(h, w, 3, tiles_imgs.data() + print_displ); //вывод каждого тайла
    //      print_displ += w * h * 3;
    //      std::cout <<"\n";
    //    }
    //  }
    tile_displ += proc_tile_count[i];  // считаем количество уже пройденных тайлов
    img_displ += img_send_count;       // считаем количество пересланныъ пикселей
  }

  MPI_Waitall((mpi_size - 1) * 3, requests, statuses);
  delete[] requests;
  delete[] statuses;
}

std::vector<uint8_t> MorozovNBlockGaussFilterMPI::SimpleMergeTiles(const std::vector<uint8_t> &tiles_data,
                                                                   const std::vector<int> &tiles_attr, int image_width,
                                                                   int image_height) {
  std::vector<uint8_t> image(image_width * image_height * 3, 0);
  int tile_data_offset = 0;

  // Предполагаем, что тайлы идут в порядке слева направо, сверху вниз
  int current_y = 0;
  int current_x = 0;

  for (size_t tile_idx = 0; tile_idx < tiles_attr.size() / 6; tile_idx++) {
    int attr_idx = tile_idx * 6;
    int tile_width = tiles_attr[attr_idx + 0];
    int tile_height = tiles_attr[attr_idx + 1];

    // Копируем данные тайла
    for (int y = 0; y < tile_height; y++) {
      for (int x = 0; x < tile_width; x++) {
        if (current_y + y >= image_height || current_x + x >= image_width) {
          continue;
        }

        int tile_pixel_idx = tile_data_offset + 3 * (y * tile_width + x);
        int image_pixel_idx = 3 * (((current_y + y) * image_width) + (current_x + x));

        image[image_pixel_idx + 0] = tiles_data[tile_pixel_idx + 0];
        image[image_pixel_idx + 1] = tiles_data[tile_pixel_idx + 1];
        image[image_pixel_idx + 2] = tiles_data[tile_pixel_idx + 2];
      }
    }

    tile_data_offset += 3 * tile_width * tile_height;

    // Переходим к следующей позиции
    current_x += tile_width;
    if (current_x >= image_width) {
      current_x = 0;
      current_y += tile_height;
    }
  }

  return image;
}

Color MorozovNBlockGaussFilterMPI::CalculatePixelColor(uint8_t *src, int x, int y, int width, int height) {
  Color res{0, 0, 0};
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;

  const uint8_t ch_max = 255;
  const uint8_t ch_min = 0;
  const int rad_x = 1;
  const int rad_y = 1;

  for (int l = -rad_y; l <= rad_y; l++) {
    for (int k = -rad_x; k <= rad_x; k++) {
      int idX = std::clamp(x + k, 0, width - 1);
      int idY = std::clamp(y + l, 0, height - 1);
      int pix_id = 3 * ((idY * width) + idX);
      r += static_cast<float>(src[pix_id + 0]) * kernel[l + rad_y][k + rad_x] / 0.0625f;
      g += static_cast<float>(src[pix_id + 1]) * kernel[l + rad_y][k + rad_x] / 0.0625f;
      b += static_cast<float>(src[pix_id + 2]) * kernel[l + rad_y][k + rad_x] / 0.0625f;
    }
  }

  res.r = std::clamp(static_cast<uint8_t>(r), ch_min, ch_max);
  res.g = std::clamp(static_cast<uint8_t>(g), ch_min, ch_max);
  res.b = std::clamp(static_cast<uint8_t>(b), ch_min, ch_max);
  return res;
}
}  // namespace morozov_n_block_gauss_filter
