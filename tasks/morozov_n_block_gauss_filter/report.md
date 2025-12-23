# Линейная фильтрация изображений (блочное разбиение). Ядро Гаусса 3x3.

- **Студент**: Морозов Никита Александрович, группа 3823Б1ПР1
- **Технология**: SEQ | MPI
- **Вариант**: 28

## 1. Введение
Обработка изображений одна из популярных работ производимых при помощи компьютеров. Но пользователи не задумываются о внутренней работе алгоритмов, им важен результат и скорость работы обработки.

Одна из задач данной сфере - фильтрация изображений. Существует множество различных фильтров, один из самых популярных - фильтр Гаусса. 

Обработка больших изображений может занимать долгое время, один из придуманных способов ускорить этот процесс - блочное разбиение (tiling).

Цель моей работы - реализовать фильтр Гауса с ядром 3х3, алгоритм разбиения изображения на блоки,распараллелить фильтрацию при помощи MPI.

## 2. Постановка задачи
На вход программе подается изображение `img` и его характеристики: ширина - `width` и выосота - `height`
- `img` - матрица коэфициентов неизвестных размера `n * n`
- `width` - вектор неизвестных размера `n`
- `height` - вектор свободных членов размера `n`

Требуется сгладить изображение применив фильтр Гауса с ядром размера `3х3`.

Тип входных данных:
```cpp
using InType = std::tuple<std::vector<uint8_t>, int, int>;
using OutType = std::vector<uint8_t>;
using TestType = std::tuple<std::string, int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;
```

Тип выходных данных:
```cpp
using OutType = std::vector<uint8_t>;
```

## 3. Базовый алгоритм (последовательная версия) 
Алгоритм фильтрации представляет собой проход по всем пикселям изображения с вычислением нового цвета по следующему правилу:
1. инициализируется цикл прохода по каждому пикселю с радиусом заданного ядра, вокруг выбранного пикселя
2. значение каждого цвета пикселя умножается на соответвующий позиции элемент ядра
3. полученные значения аккумулируются
4. по оконачии цикла каждое значение проверяется на допустимость значений и сохраняется в новый пиксель

Код Алгоритма вычисление цвета нового пикселя:
``` cpp
  float kernel_sum = 16.0f;
  float kernel_inv = 1.0f / kernel_sum;

  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;

  for (int l = -rad_y; l <= rad_y; l++) {
    for (int k = -rad_x; k <= rad_x; k++) {
      int idX = std::clamp(x + k, 0, width - 1);
      int idY = std::clamp(y + l, 0, height - 1);
      int pix_id = 3 * ((idY * width) + idX);
      float kernel_val = kernel[l + rad_y][k + rad_x] * kernel_inv;
      r += static_cast<float>(src[pix_id + 0]) * kernel_val;
      g += static_cast<float>(src[pix_id + 1]) * kernel_val;
      b += static_cast<float>(src[pix_id + 2]) * kernel_val;
    }
  }

  Color res;
  res.r = std::clamp(static_cast<uint8_t>(r), ch_min, ch_max);
  res.g = std::clamp(static_cast<uint8_t>(g), ch_min, ch_max);
  res.b = std::clamp(static_cast<uint8_t>(b), ch_min, ch_max);
```

Код Алгоритма Фильтрации:
```cpp
 std::vector<uint8_t> res(width * height * 3, 0);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      Color new_color = CalculatePixelColor(src, x, y, width, height);
      int pix_idx = 3 * ((y * width) + x);
      res[pix_idx + 0] = new_color.r;
      res[pix_idx + 1] = new_color.g;
      res[pix_idx + 2] = new_color.b;
    }
  }
```

**Характеристики:**
Сложность алгоритма равна O(M * N * (k * k)), где:
- `M` - ширина изображения
- `N` - высота изображения
- `k` - размер ядра 

## 4. Схема распараллеливания

Для распараллеливания алгоритма, изображение разбивается на части, каждая часть обрабатывается своим процессом. Один из типов разбиения - **блочное** подразумевает, что изображение разбивается на равные участки, учаастки равномерно распределяются между всеми процессами. После фильтрации всех участков изображения, оно собирается обратно.

### 4.1 Алгоритм Распределения
Алгоритм распределения состоит из нескольких этапов
1. Процессы узнаю свой ранг и кол-во процессов исполняющих программу
2. Процесс с `рангом 0` получает входные данные
3. Вычисление размера тайла
4. Получение числа строк и столбцов, на которые тайлы разобьют изображение
5. Вычисление общего числа тайлов
6. Подсчет количества тайлов для каждого процесса
7. Получение информации о тайле (размеры, пиксели из исходного изображения, позиция)
8. Распределение с между всеми процессами

Код алгоритмов рассчета информации о тайлах представен в приложении [1, 2, 3]

**Код Алгоритма распредления**
```cpp
if(rank == 0) {
  //вычисление данных о тайлах
  ScatterTiles(proc_tile_count, tiles_data, tiles_attr, mpi_size);
} else { 
  //получение данных на остальных потоках
  MPI_Recv(&tiles_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  tiles_attr.resize(tiles_count * 6);
  MPI_Recv(tiles_attr.data(), static_cast<int>(tiles_attr.size()), MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  tiles_data_size = GetTilesDataSize(tiles_attr, 0, tiles_count);
  tiles_data.resize(tiles_data_size);
  MPI_Recv(tiles_data.data(), tiles_data_size, MPI_BYTE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

//Алгоритм распрделения
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
```


## 4.2 Алгоритм Подсчета
Алгоритм подсчета абсолютно индентичен последовательной версии за исключением того, что он производится для каждого тайла на процессе

**Код**
```cpp
for(int k = 0; k < tiles_count; k++) {
  //получение аттрибутов тайла
  //подсчет фильтра для тайла
  //сохранение обработанного блока
}
```
Код алгоритма в приложении `[5]`

## 4.3 Получение конечного результата
По окончании фильтрации каждым процессам, все обработанные участки пересылаются на процесс с `рангом 0` при помощи `MPI_Gatherv()`.
Как только все данные им получены начианется алгоритм слияния тайлов в единое изображение.

**Описание алгоритма слияния**
1. Начинается обход по строкам на которые разбилось изображение
2. Каждый блок, входящий в строку, записывает свои данные в новое изображение   

Код алгоритма в приложении [6]

## 4.4 Схема работы программы
```
┌──────────────────────────────────────────────────────────┐
│  Входные данные   (Изображение и его размеры)           │
└────────┬─────────────────────────────────────────────────┘
         │ (передача входных аргументов программе)
         ↓
 ┌──────────────────────┬──────────────────────┬────────────────────────┐
 │  Процесс 0           │  Процесс 1           │  Процесс 2             │  
 │ получает входные     │ Ждут данных от       | Ждут данных от         |
 | данные, разбивает    | нулевого             |             нулевого   |
 | изображение на блоки |                      |                        |
 └───────┬──────────────┴──────────────────────┴────────────────────────┘
         │ передача данных MPI_Isend, MPI_Recv
         ↓
 ┌────────────────┬────────────────┬────────────────┐
 │ Фильтрация     │ Фильтрация     │ Фильтрация     │
 |  тайлов        │ тайлов         │  тайлов        │
 └───────┬────────┴────────────────┴────────────────┘
         │ Отправка данных на процесс 0, MPI_Gatherv()
         ↓                                                    
 ┌────────────────────────────────────────────────┐         
 │   Сбор результирующиего изображения из блоков  |
 |    на процессе с рангом 0                      |
 └───────┬────────────────────────────────────────┘        
         │ MPI_Bcast()  рассылка результата                    
         ↓                                                    
         │ 
         ↓
┌───────────────────────────────────────┐  
│  Сохранение: GetOutput() = new_img    │
└───────────────────────────────────────┘
```

## 5. Детали реализации
**Структура проекта**
|          Файл          |                 Назначение                  |
|------------------------|---------------------------------------------|
| `common.hpp`           | Определение входных и выходных типов задачи |
| `ops_seq.hpp/.cpp`     |         Последовательная реализация         |
| `ops_mpi.hpp/.cpp`     |                MPI-реализация               |
| `functional/main.cpp`  |             Функциональные тесты            |
| `performance/main.cpp` |       Тестирование производительности       |

## 6. Экспериментальная среда

|  Компонент |               Значение                       |
|------------|----------------------------------------------|
|     CPU    |           Apple M2 (8 cores)                 |
|     RAM    |                 16 GB                        |
|     ОС     | OS: Ubuntu 24.04 (DevContainer / macOs 15.6) |
| Компилятор | GCC 13.3.0 (g++), C++20, CMake, Release      |
|     MPI    |        mpirun (Open MPI) 4.1.6               |

Тестовые данные: 
1. Функциональные тесты: 
    - тестовые данные из фалов
    - сгенерированные изображения
2. Перформанс тесты:
    - сгенериорванные изображения больших размеров

## 7. Результаты и обсуждение

### 7.1 Корректность
Функциональные тесты используют входные файлы или генерацию в зависимости от параметра теста.

Струкутра параметра теста:
- строка
- ширина изображения (только для генерации)
- высота изображения (только для генерации)

Тест считывает первый аргумент параметра и, если он не равен `"gen"`, 
считывает входной файл, в котором хранятся: тестовое изображение
Входные файлы имеют следующие наименования:
```
img_1.jpg
img_2.jpg
...
```
Если тест во втором аргументе параметра считал строку `"gen"`, то тестовое изображение будет сгенерированно с заданнами размерами.

Корректность работы теста проверяется только для параллельной версии. Тест пройден успешно, если выходные данные равны выхдным данным последовательного алгоритма

### 7.2 Производительность

Размер генерируемого изображения для теста `2000 x 2000`
| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 0.041   | 1.00    | N/A        |
| MPI         | 2     | 0.041   | 1.00    | 50%        |
| MPI         | 4     | 0.024   | 1.71    | 42%        |

## 8. Заключение
В ходе выполнения работы:
- реализовал фильтр Гауса с ядром 3х3
- алгоритм разбиения изображения на блоки
- реализовал параллельный алгоритм фильтра Гауса с блочным разбиением при помощи MPI
- рассчитал характекристик ускорение и эффективности для распараллеленного алгоритма
- выяснил, что параллельная обработка позволяет ускорить алгоритм

## 9. Источники
1. А.В. Сысоев               Курс лекций по параллельному программированию
2. В.П. Гергель, Р.Г.        Стронгин Основы параллельных вычислений для многопроцессорных вычислительных систем
3. В.Е. Турпалов             Курс лекций по обработке изображений
4. Документация Open MPI     https://www.open-mpi.org/doc/
5. Microsoft Функции MPI     https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-functions

## Приложения

### 1. Вычисление размера блока
```cpp
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
```

### 2. Подсчет числа строк/столбцов
```cpp
int MorozovNBlockGaussFilterMPI::GetTileColsRowsCount(int src_wh, int tile_wh) {
  int count = src_wh / tile_wh;
  if (src_wh % tile_wh != 0) {
    count++;
  }
  return count;
}
```

### 3. Получение данных для каждого блока
```cpp
std::tuple<std::vector<uint8_t>, std::vector<int>> MorozovNBlockGaussFilterMPI::ParseImageToTiles(
    const std::vector<uint8_t> &src, int width, int height, int rows, int cols, int tile_w, int tile_h) {
  int shifts_len = 6 * cols * rows;
  std::vector<int> whrlud(shifts_len, 0);
  std::vector<uint8_t> tiles_imgs;

  int tile_ind_shifts = 0;
  for (int idY = 0; idY < rows; idY++) {
    for (int idX = 0; idX < cols; idX++) {
      int x = idX * tile_w;
      int y = idY * tile_h;
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
      int tile_start = tiles_imgs.size();
      tiles_imgs.resize(tiles_imgs.size() + w_over * h_over * 3, 0);

      // Copy tile data with borders
      for (int j = -u; j < t_height + d; j++) {
        for (int i = -l; i < t_width + r; i++) {
          int src_x = std::clamp(i + x, 0, width - 1);
          int src_y = std::clamp(j + y, 0, height - 1);
          int src_pix_id = 3 * ((src_y * width) + src_x);
          int tile_pix_id = tile_start + 3 * (((j + u) * w_over) + (i + l));
          tiles_imgs[tile_pix_id + 0] = src[src_pix_id + 0];
          tiles_imgs[tile_pix_id + 1] = src[src_pix_id + 1];
          tiles_imgs[tile_pix_id + 2] = src[src_pix_id + 2];
        }
      }
    }
  }
  return std::make_tuple(tiles_imgs, whrlud);
}
```

### 4. Получение размера одного блока
```cpp
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
```

### 5. Обработка всех тайлов процесса
```cpp
  // Process tiles
  std::vector<uint8_t> res(tiles_data_size, 0);
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
```

### 6. Алгоритм слияния блоков в результирующее изображение
```cpp
std::vector<uint8_t> MorozovNBlockGaussFilterMPI::SimpleMergeTiles(const std::vector<uint8_t> &tiles_data,
                                                                   const std::vector<int> &tiles_attr, int image_width,
                                                                   int image_height, int tile_w, int tile_h, int cols,
                                                                   int rows) {
  std::vector<uint8_t> image(image_width * image_height * 3, 0);
  int tile_data_offset = 0;

  // Tiles are stored in row-major order (same as created in ParseImageToTiles)
  for (int idY = 0; idY < rows; idY++) {
    for (int idX = 0; idX < cols; idX++) {
      int tile_idx = idY * cols + idX;
      if (tile_idx >= static_cast<int>(tiles_attr.size() / 6)) {
        break;
      }

      int attr_idx = tile_idx * 6;
      int tile_width = tiles_attr[attr_idx + 0];
      int tile_height = tiles_attr[attr_idx + 1];

      // Calculate original tile position in the image
      int tile_start_x = idX * tile_w;
      int tile_start_y = idY * tile_h;

      // Copy tile data to the correct position in the image
      for (int y = 0; y < tile_height; y++) {
        for (int x = 0; x < tile_width; x++) {
          int image_x = tile_start_x + x;
          int image_y = tile_start_y + y;

          if (image_x >= image_width || image_y >= image_height) {
            continue;
          }

          int tile_pixel_idx = tile_data_offset + 3 * (y * tile_width + x);
          int image_pixel_idx = 3 * ((image_y * image_width) + image_x);

          image[image_pixel_idx + 0] = tiles_data[tile_pixel_idx + 0];
          image[image_pixel_idx + 1] = tiles_data[tile_pixel_idx + 1];
          image[image_pixel_idx + 2] = tiles_data[tile_pixel_idx + 2];
        }
      }

      tile_data_offset += 3 * tile_width * tile_height;
    }
  }

  return image;
}
```