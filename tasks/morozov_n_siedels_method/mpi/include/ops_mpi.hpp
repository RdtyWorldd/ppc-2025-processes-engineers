#pragma once

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
  bool InEpsBound(std::vector<double>& iter_eps, double correct_eps);
};

}  // namespace morozov_n_siedels_method
