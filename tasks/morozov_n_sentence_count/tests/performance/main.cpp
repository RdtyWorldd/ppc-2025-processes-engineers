#include <gtest/gtest.h>

#include "morozov_n_sentence_count/common/include/common.hpp"
#include "morozov_n_sentence_count/mpi/include/ops_mpi.hpp"
#include "morozov_n_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozov_n_sentence_count {

class MorozovNRunSentenceCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
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

TEST_P(MorozovNRunSentenceCountPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MorozovNSentenceCountMPI, MorozovNSentenceCountSEQ>(PPC_SETTINGS_morozov_n_sentence_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MorozovNRunSentenceCountPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MorozovNRunSentenceCountPerfTests, kGtestValues, kPerfTestName);

}  // namespace morozov_n_sentence_count
