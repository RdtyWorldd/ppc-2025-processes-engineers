#pragma once

#include <cstddef>
#include <vector>

#include "morozov_n_siedels_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozov_n_siedels_method {

class MorozovNSiedelsMethodSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MorozovNSiedelsMethodSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static bool EpsOutOfBound(std::vector<double> &iter_eps, double correct_eps);
  static int CalcMatrixRank(std::size_t n, std::size_t m, std::vector<double> &a);
  static bool GetPivotRow(std::size_t *pivot_row, std::vector<bool> &row_selected, std::size_t col,
                          std::vector<std::vector<double>> &mat, double e);
  static void SubRow(std::size_t pivot_row, std::size_t col, std::vector<std::vector<double>> &mat, double e);
};

}  // namespace morozov_n_siedels_method
