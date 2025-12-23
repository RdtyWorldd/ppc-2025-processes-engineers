#include "morozov_n_siedels_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "morozov_n_siedels_method/common/include/common.hpp"

namespace morozov_n_siedels_method {

MorozovNSiedelsMethodSEQ::MorozovNSiedelsMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MorozovNSiedelsMethodSEQ::ValidationImpl() {
  bool matrix_correct = false;
  std::size_t n = std::get<0>(GetInput());
  std::vector<double> a = std::get<1>(GetInput());
  std::vector<double> b = std::get<2>(GetInput());
  if ((a.size() == (n * n)) && (b.size() == n)) {
    int rank_a = CalcMatrixRank(n, n, a);

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
  return matrix_correct;
}

bool MorozovNSiedelsMethodSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovNSiedelsMethodSEQ::RunImpl() {
  std::size_t n = std::get<0>(GetInput());
  std::vector<double> &a = std::get<1>(GetInput());
  std::vector<double> &b = std::get<2>(GetInput());
  double eps = std::get<3>(GetInput());

  std::vector<double> x(n, 0);
  std::vector<double> iter_eps(n, -1);
  bool complete = false;
  while (!complete) {
    for (std::size_t i = 0; i < n; i++) {
      double iter_x = b[i];
      for (std::size_t j = 0; j < i; j++) {
        iter_x = iter_x - (a[(i * n) + j] * x[j]);
      }
      for (std::size_t j = i + 1; j < n; j++) {
        iter_x = iter_x - (a[(i * n) + j] * x[j]);
      }
      iter_x = iter_x / a[(i * n) + i];

      // обновление погрешности
      iter_eps[i] = std::fabs(iter_x - x[i]);
      // обновление полученного корня
      x[i] = iter_x;
    }
    complete = !EpsOutOfBound(iter_eps, eps);
  }

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

bool MorozovNSiedelsMethodSEQ::EpsOutOfBound(std::vector<double> &iter_eps, double correct_eps) {
  double max_in_iter = *std::ranges::max_element(iter_eps);
  return max_in_iter > correct_eps;
}
int MorozovNSiedelsMethodSEQ::CalcMatrixRank(std::size_t n, std::size_t m, std::vector<double> &a) {
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

bool MorozovNSiedelsMethodSEQ::GetPivotRow(std::size_t *pivot_row, std::vector<bool> &row_selected, std::size_t col,
                                           std::vector<std::vector<double>> &mat, double e) {
  for (std::size_t row = 0; row < mat.size(); row++) {
    if (!row_selected[row] && std::fabs(mat[row][col]) > e) {
      *pivot_row = row;
      return true;
    }
  }
  return false;
}

void MorozovNSiedelsMethodSEQ::SubRow(std::size_t pivot_row, std::size_t col, std::vector<std::vector<double>> &mat,
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
