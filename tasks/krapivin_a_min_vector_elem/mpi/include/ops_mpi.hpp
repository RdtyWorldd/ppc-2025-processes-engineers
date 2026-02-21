#pragma once
#include "krapivin_a_min_vector_elem/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krapivin_a_min_vector_elem {

class KrapivinAMinVectorElemMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KrapivinAMinVectorElemMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void SplitData(std::vector<int> &input, std::vector<int> &send_counts, std::vector<int> &displacements, int rank,
                 int mpi_size);
  int FindMin(const std::vector<int> &v);
  void GatherResuts(std::vector<int> &input, int local_res, std::vector<int> &send_counts,
                    std::vector<int> &displacements, int rank, int mpi_size);
};

}  // namespace krapivin_a_min_vector_elem
