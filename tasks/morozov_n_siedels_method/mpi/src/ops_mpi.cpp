#include "morozov_n_siedels_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "morozov_n_siedels_method/common/include/common.hpp"
#include "util/include/util.hpp"

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
    int n = std::get<0>(GetInput());
    std::vector<double> a = std::get<1>(GetInput());
    std::vector<double> b = std::get<2>(GetInput());
    if ((a.size() == static_cast<std::size_t>(n * n)) && (b.size() == static_cast<std::size_t>(n))) {
      int rank_a = CalcMatrixRank(n, n, a);

      std::vector<double> ext_a;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
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

  double *a = 0;
  double *b = 0;

  int n = 0;
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
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int step = n / mpi_size;
  int remainder = n % mpi_size;

  std::vector<int> send_counts(mpi_size, step);
  std::vector<int> displacements(mpi_size, 0);
  int displacement = 0;
  for (int i = 0; i < remainder; ++i) {
    send_counts[i]++;
  }

  displacements[0] = 0;
  int disp_sum = 0;
  for (int i = 1; i < mpi_size; ++i) {
    disp_sum += send_counts[i - 1];
    displacements[i] = disp_sum;
  }
  displacement = displacements[rank];

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
  MPI_Scatterv(b, send_counts.data(), displacements.data(), MPI_DOUBLE, local_b.data(), local_b.size(), MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  // //debug
  // {
  //   std::cout << "rank: " << rank << "\n";
  //   for(int i = 0; i < send_counts[rank]; i++) {
  //       std::cout << local_b[i] << " ";
  //   }
  //   std::cout <<"\n--------------\n";
  // }

  for (int i = 0; i < mpi_size; i++) {
    send_counts[i] *= n;
    displacements[i] *= n;
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
  MPI_Scatterv(a, send_counts.data(), displacements.data(), MPI_DOUBLE, local_a.data(), local_a.size(), MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  for (int i = 0; i < mpi_size; i++) {
    send_counts[i] /= n;
    displacements[i] /= n;
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
  do {
    for (int i = 0; i < static_cast<int>(local_b.size()); i++) {
      int g_row = displacement + i;  // позиция строки в общей матрице
      double iter_x = local_b[i];    // результат на итерации
      // циклы с суммой без элмента диагонали
      for (int j = 0; j < g_row; j++) {
        iter_x = iter_x - local_a[(i * n) + j] * x[j];
      }
      for (int j = g_row + 1; j < n; j++) {
        iter_x = iter_x - local_a[(i * n) + j] * x[j];
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

    MPI_Allgatherv(x_new.data(), x_new.size(), MPI_DOUBLE, x.data(), send_counts.data(), displacements.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgatherv(iter_eps_new.data(), iter_eps_new.size(), MPI_DOUBLE, iter_eps.data(), send_counts.data(),
                   displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
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
  } while (InEpsBound(iter_eps, eps));

  // debug
  if (rank == 0) {
    std::string result = "answer X:\n";
    for (int i = 0; i < n; i++) {
      result += std::to_string(x[i]) + " ";
    }
    result += '\n';
    std::cout << result;
  }

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

bool MorozovNSiedelsMethodMPI::InEpsBound(std::vector<double> &iter_eps, double correct_eps) {
  double max_in_iter = *(std::max_element(begin(iter_eps), end(iter_eps)));
  // debug
  // std::cout << max_in_iter << "\n";
  return max_in_iter > correct_eps;
}

int MorozovNSiedelsMethodMPI::CalcMatrixRank(int n, int m, std::vector<double> &a) {
  const double EPS = 1e-9;
  std::vector<std::vector<double>> mat(n, std::vector<double>(m));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat[i][j] = a[i * n + j];
    }
  }

  int rank = 0;
  std::vector<bool> row_selected(n, false);

  for (int col = 0; col < n; col++) {
    int pivot_row = -1;
    for (int row = 0; row < n; row++) {
      if (!row_selected[row] && abs(mat[row][col]) > EPS) {
        pivot_row = row;
        break;
      }
    }
    if (pivot_row == -1) {
      continue;
    }

    rank++;
    row_selected[pivot_row] = true;

    // Нормализация строки
    double pivot = mat[pivot_row][col];
    for (int j = col; j < n; j++) {
      mat[pivot_row][j] /= pivot;
    }

    // Вычитание текущей строки из других строк
    for (int row = 0; row < n; row++) {
      if (row != pivot_row && abs(mat[row][col]) > EPS) {
        double factor = mat[row][col];
        for (int j = col; j < n; j++) {
          mat[row][j] -= factor * mat[pivot_row][j];
        }
      }
    }
  }

  return rank;
}

}  // namespace morozov_n_siedels_method
