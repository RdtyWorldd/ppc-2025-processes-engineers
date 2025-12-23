#include "morozov_n_siedels_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "morozov_n_siedels_method/common/include/common.hpp"

namespace morozov_n_siedels_method {

MorozovNSiedelsMethodMPI::MorozovNSiedelsMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput();
}

bool MorozovNSiedelsMethodMPI::ValidationImpl() {
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  bool matrix_correct = false;
  if (rank == 0) {
    std::size_t n = std::get<0>(GetInput());
    std::vector<double> a = std::get<1>(GetInput());
    std::vector<double> b = std::get<2>(GetInput());
    if ((a.size() == (n * n)) && (b.size() == n)) {
      int rank_a = CalcMatrixRank(n, n, a);
      // вычисление расширенной матрицы
      std::vector<double> ext_a;
      for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) {
          ext_a.push_back(a[(i * n) + j]);
        }
        ext_a.push_back(b[i]);
      }
      int rank_ext_a = CalcMatrixRank(n, n + 1, ext_a);
      if (rank_a >= rank_ext_a) {
        matrix_correct = true;
      }
    }
  }
  MPI_Bcast(&matrix_correct, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
  return matrix_correct;
}

bool MorozovNSiedelsMethodMPI::PreProcessingImpl() {
  return true;
}

bool MorozovNSiedelsMethodMPI::RunImpl() {
  double eps = std::get<3>(GetInput());

  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  double *a = nullptr;
  double *b = nullptr;

  std::size_t n = 0;
  if (rank == 0) {
    n = std::get<0>(GetInput());
    a = std::get<1>(GetInput()).data();
    b = std::get<2>(GetInput()).data();

    // debug
    // {
    //   std::cout << n << "\n";
    //   std::cout << "a: " << "\n";
    //   for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //       std::cout << a[i * n + j] << " ";
    //     }
    //     std::cout << "\n";
    //   }
    //   std::cout << "b: " << "\n";
    //   for (int i = 0; i < n; i++) {
    //     std::cout << b[i] << " ";
    //   }
    //   std::cout << "\n--------------\n";
    // }
  }
  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  int step = static_cast<int>(n) / mpi_size;
  int remainder = static_cast<int>(n) % mpi_size;

  std::vector<int> send_counts(mpi_size, step);
  std::vector<int> displacements(mpi_size, 0);
  std::size_t displacement = 0;
  for (int i = 0; i < remainder; ++i) {
    send_counts[i]++;
  }

  displacements[0] = 0;
  int disp_sum = 0;
  for (int i = 1; i < mpi_size; ++i) {
    disp_sum += send_counts[i - 1];
    displacements[i] = disp_sum;
  }
  displacement = static_cast<std::size_t>(displacements[rank]);

  // debug
  // {
  //   if (rank == 0) {
  //     std::cout << "send_size: " << "\n";
  //     for (int i = 0; i < mpi_size; i++) {
  //       std::cout << send_counts[i] << " ";
  //     }
  //     std::cout << "disp: " << "\n";
  //     for (int i = 0; i < mpi_size; i++) {
  //       std::cout << displacements[i] << " ";
  //     }
  //     std::cout << "\n--------------\n";
  //   }
  //   std::cout << "rank " << rank << " l_b size: " << send_counts[rank] << "\n";
  // }

  std::vector<double> local_b(send_counts[rank]);
  MPI_Scatterv(b, send_counts.data(), displacements.data(), MPI_DOUBLE, local_b.data(),
               static_cast<int>(local_b.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // //debug
  // {
  //   std::cout << "rank: " << rank << "\n";
  //   for(int i = 0; i < send_counts[rank]; i++) {
  //       std::cout << local_b[i] << " ";
  //   }
  //   std::cout <<"\n--------------\n";
  // }

  for (int i = 0; i < mpi_size; i++) {
    send_counts[i] *= static_cast<int>(n);
    displacements[i] *= static_cast<int>(n);
  }

  // debug
  //  if(rank == 1) {
  //    std::cout << "send_size: " << "\n";
  //    for(int i = 0; i < mpi_size; i++) {
  //      std::cout << send_counts[i] << " ";
  //    }
  //    std::cout <<"\n";
  //    std::cout << "disp: " << "\n";
  //    for(int i = 0; i < mpi_size; i++) {
  //      std::cout << displacements[i] << " ";
  //    }
  //    std::cout <<"\n--------------\n";
  //  }

  std::vector<double> local_a(send_counts[rank]);
  MPI_Scatterv(a, send_counts.data(), displacements.data(), MPI_DOUBLE, local_a.data(),
               static_cast<int>(local_a.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int i = 0; i < mpi_size; i++) {
    send_counts[i] /= static_cast<int>(n);
    displacements[i] /= static_cast<int>(n);
  }

  // debug
  // if (rank == 0) {
  //   std::cout << "send_size: " << "\n";
  //   for (int i = 0; i < mpi_size; i++) {
  //     std::cout << send_counts[i] << " ";
  //   }
  //   std::cout << "\ndisp: " << "\n";
  //   for (int i = 0; i < mpi_size; i++) {
  //     std::cout << displacements[i] << " ";
  //   }
  //   std::cout << "\n--------------\n";
  // }

  std::vector<double> x(n, 0);                           // вектор ответа
  std::vector<double> x_new(local_b.size(), 0);          // вектор для рассылки
  std::vector<double> iter_eps(n, -1);                   // вектор погрешности
  std::vector<double> iter_eps_new(local_b.size(), -1);  // вектор для рассылки
  bool complete = false;
  while (!complete) {
    for (std::size_t i = 0; i < local_b.size(); i++) {
      std::size_t g_row = displacement + i;  // позиция строки в общей матрице
      double iter_x = local_b[i];            // результат на итерации
      // циклы с суммой без элмента диагонали
      for (std::size_t j = 0; j < g_row; j++) {
        iter_x = iter_x - (local_a[(i * n) + j] * x[j]);
      }
      for (std::size_t j = g_row + 1; j < n; j++) {
        iter_x = iter_x - (local_a[(i * n) + j] * x[j]);
      }

      iter_x = iter_x / local_a[(i * n) + g_row];  // вычисление корня стоящего на диагонали
      // обновление погрешности
      iter_eps[g_row] = std::fabs(iter_x - x[g_row]);
      iter_eps_new[i] = iter_eps[g_row];
      // обновление полученного корня
      x[g_row] = iter_x;
      x_new[i] = iter_x;
    }

    // debug
    // if (rank == 1) {
    //   std::cout << "local iteration rank: " << rank << "\n";
    //   for (size_t i = 0; i < local_b.size(); i++) {
    //     std::cout << x_new[i] << " ";
    //   }
    //   std::cout << "\n";
    // }

    MPI_Allgatherv(x_new.data(), static_cast<int>(x_new.size()), MPI_DOUBLE, x.data(), send_counts.data(),
                   displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgatherv(iter_eps_new.data(), static_cast<int>(iter_eps_new.size()), MPI_DOUBLE, iter_eps.data(),
                   send_counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    complete = !EpsOutOfBound(iter_eps, eps);
    // debug
    //  if (rank == 0) {
    //    std::cout << "rank: " << rank << " iteration: " << iter_count <<" X:\n";
    //    for(int i = 0; i < n; i++) {
    //      std::cout << x[i] << " ";
    //    }
    //    std::cout << "rank: " << rank << " iteration: " << iter_count <<" EPS:\n";
    //    for(int i = 0; i < n; i++) {
    //      std::cout << iter_eps[i] << " ";
    //    }
    //    std::cout <<"\n--------\n";
    //  }
    // debug
    // if (rank == 0) {
    //   std::cout << "answer X:\n";
    //   for (int i = 0; i < n; i++) {
    //     std::cout << x[i] << " ";
    //   }
    //   std::cout << "\n";
    // }
  }

  // debug
  // if (rank == 0) {
  //   std::string result = "answer X:\n";
  //   for (int i = 0; i < n; i++) {
  //     result += std::to_string(x[i]) + " ";
  //   }
  //   result += '\n';
  //   std::cout << result;
  // }

  GetOutput() = x;
  // std::cout << "rank:" << rank << " end of calc\n";
  if (rank != 0) {
    delete[] a;
    delete[] b;
  }
  return true;
}

bool MorozovNSiedelsMethodMPI::PostProcessingImpl() {
  return true;
}

bool MorozovNSiedelsMethodMPI::EpsOutOfBound(std::vector<double> &iter_eps, double correct_eps) {
  double max_in_iter = *std::ranges::max_element(iter_eps);
  // debug
  // std::cout << max_in_iter << "\n";
  return max_in_iter > correct_eps;
}

int MorozovNSiedelsMethodMPI::CalcMatrixRank(std::size_t n, std::size_t m, std::vector<double> &a) {
  const double e = 1e-9;
  std::vector<std::vector<double>> mat(n, std::vector<double>(m));
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < m; j++) {
      mat[i][j] = a[(i * n) + j];
    }
  }

  int rank = 0;
  std::vector<bool> row_selected(n, false);

  for (std::size_t col = 0; col < n; col++) {
    std::size_t pivot_row = 0;
    if (!GetPivotRow(&pivot_row, row_selected, col, mat, e)) {
      continue;
    }
    rank++;
    row_selected[pivot_row] = true;
    // Нормализация строки и вычитание строки из других строк
    //-> приведение к степнчатому виду
    SubRow(pivot_row, col, mat, e);
  }
  return rank;
}

bool MorozovNSiedelsMethodMPI::GetPivotRow(std::size_t *pivot_row, std::vector<bool> &row_selected, std::size_t col,
                                           std::vector<std::vector<double>> &mat, double e) {
  for (std::size_t row = 0; row < mat.size(); row++) {
    if (!row_selected[row] && std::fabs(mat[row][col]) > e) {
      *pivot_row = row;
      return true;
    }
  }
  return false;
}

void MorozovNSiedelsMethodMPI::SubRow(std::size_t pivot_row, std::size_t col, std::vector<std::vector<double>> &mat,
                                      double e) {
  std::size_t n = mat.size();
  std::size_t m = mat[0].size();
  // Нормализация строки
  double pivot = mat[pivot_row][col];
  for (std::size_t j = col; j < n; j++) {
    mat[pivot_row][j] /= pivot;
  }
  // Вычитание текущей строки из других строк
  for (std::size_t row = 0; row < n; row++) {
    if (row != pivot_row && std::fabs(mat[row][col]) > e) {
      double factor = mat[row][col];
      for (std::size_t j = col; j < m; j++) {
        mat[row][j] -= factor * mat[pivot_row][j];
      }
    }
  }
}

}  // namespace morozov_n_siedels_method
