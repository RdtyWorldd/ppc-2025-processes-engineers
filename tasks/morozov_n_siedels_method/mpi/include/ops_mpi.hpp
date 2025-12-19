#pragma once

#include <vector>

#include "morozov_n_siedels_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozov_n_siedels_method {

class MorozovNSiedelsMethodMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MorozovNSiedelsMethodMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static bool EpsOutOfBound(std::vector<double> &iter_eps, double correct_eps);
  static int CalcMatrixRank(int n, int m, std::vector<double> &a);
};

}  // namespace morozov_n_siedels_method
