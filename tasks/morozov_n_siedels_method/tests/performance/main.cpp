#include <gtest/gtest.h>

#include "morozov_n_siedels_method/common/include/common.hpp"
#include "morozov_n_siedels_method/mpi/include/ops_mpi.hpp"
#include "morozov_n_siedels_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozov_n_siedels_method {

class MorozovNSiedelsMethodPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MorozovNSiedelsMethodPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MorozovNSiedelsMethodMPI, MorozovNSiedelsMethodSEQ>(PPC_SETTINGS_morozov_n_siedels_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MorozovNSiedelsMethodPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MorozovNSiedelsMethodPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace morozov_n_siedels_method
