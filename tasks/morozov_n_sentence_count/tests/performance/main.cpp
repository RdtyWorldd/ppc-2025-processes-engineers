#include <gtest/gtest.h>

#include "morozov_n_sentence_count/common/include/common.hpp"
#include "morozov_n_sentence_count/mpi/include/ops_mpi.hpp"
#include "morozov_n_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozov_n_sentence_count {

class MorozovNRunSentenceCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const std::string test_file_path = "test_4.txt";
  const std::size_t task_answer = 18000;
  InType input_data_{};

  void SetUp() override {
    std::string text = "";
    {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_morozov_n_sentence_count, test_file_path);
      std::ifstream file(abs_path);
      std::stringstream ss;
      ss << file.rdbuf();
      text = ss.str();
    }
    input_data_ = text;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::cout << output_data;
    std::cout << std::endl;
    return output_data == task_answer;
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
