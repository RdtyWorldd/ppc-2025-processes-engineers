#include <gtest/gtest.h>

#include <limits>
#include <random>

#include "krapivin_a_min_vector_elem/common/include/common.hpp"
#include "krapivin_a_min_vector_elem/mpi/include/ops_mpi.hpp"
#include "krapivin_a_min_vector_elem/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace krapivin_a_min_vector_elem {

class KrapivinAMinVectorElemPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int k_count_ = 100000;
  int correct_data_ = 0;
  InType input_data_;

  void SetUp() override {
    GenerateVector(k_count_, 777);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == correct_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
  void GenerateVector(int n, int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist;

    input_data_.resize(n);
    int min_element = std::numeric_limits<int>::max();

    for (auto &elem : input_data_) {
      elem = dist(gen);
      if (elem < min_element) {
        min_element = elem;
      }
    }
    correct_data_ = min_element;
  }
};

TEST_P(KrapivinAMinVectorElemPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KrapivinAMinVectorElemMPI, KrapivinAMinVectorElemSEQ>(
    PPC_SETTINGS_krapivin_a_min_vector_elem);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KrapivinAMinVectorElemPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KrapivinAMinVectorElemPerfTest, kGtestValues, kPerfTestName);

}  // namespace krapivin_a_min_vector_elem
