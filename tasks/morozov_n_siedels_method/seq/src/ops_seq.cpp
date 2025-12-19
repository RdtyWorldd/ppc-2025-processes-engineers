#include "morozov_n_siedels_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

#include "morozov_n_siedels_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace morozov_n_siedels_method {

MorozovNSiedelsMethodSEQ::MorozovNSiedelsMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MorozovNSiedelsMethodSEQ::ValidationImpl() {
  bool matrix_correct = false;
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
  return matrix_correct;
}

bool MorozovNSiedelsMethodSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovNSiedelsMethodSEQ::RunImpl() {
  int n = std::get<0>(GetInput());
  std::vector<double> &a = std::get<1>(GetInput());
  std::vector<double> &b = std::get<2>(GetInput());
  double eps = std::get<3>(GetInput());

  std::vector<double> x(n, 0);
  std::vector<double> iter_eps(n, -1);
  do {
    for (int i = 0; i < n; i++) {
      double iter_x = b[i];
      for (int j = 0; j < i; j++) {
        iter_x = iter_x - a[(i * n) + j] * x[j];
      }
      for (int j = i + 1; j < n; j++) {
        iter_x = iter_x - a[(i * n) + j] * x[j];
      }
      iter_x = iter_x / a[(i * n) + i];

      // обновление погрешности
      iter_eps[i] = std::fabs(iter_x - x[i]);
      // обновление полученного корня
      x[i] = iter_x;
    }
  } while (InEpsBound(iter_eps, eps));

  // for (size_t i = 0; i < x.size(); i++) {
  //   std::cout << x[i] << " ";
  // }
  // std::cout << "\n";
  GetOutput() = x;
  return true;
}

bool MorozovNSiedelsMethodSEQ::PostProcessingImpl() {
  return true;
}

bool MorozovNSiedelsMethodSEQ::InEpsBound(std::vector<double> &iter_eps, double correct_eps) {
  double max_in_iter = *(std::max_element(begin(iter_eps), end(iter_eps)));
  return max_in_iter > correct_eps;
}
int MorozovNSiedelsMethodSEQ::CalcMatrixRank(int n, int m, std::vector<double> &a) {
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
