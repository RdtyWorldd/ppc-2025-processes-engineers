#pragma once

#include "krapivin_a_min_vector_elem/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krapivin_a_min_vector_elem {

class KrapivinAMinVectorElemSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KrapivinAMinVectorElemSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krapivin_a_min_vector_elem
