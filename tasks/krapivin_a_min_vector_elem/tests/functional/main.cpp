#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "krapivin_a_min_vector_elem/common/include/common.hpp"
#include "krapivin_a_min_vector_elem/mpi/include/ops_mpi.hpp"
#include "krapivin_a_min_vector_elem/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace krapivin_a_min_vector_elem {

class KrapivinAMinVectorElemRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int n = std::get<0>(params);
    GenerateVector(n, 777);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (correct_data_ == output_data);
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

 private:
  InType input_data_;
  int correct_data_ = 0;
};

namespace {

TEST_P(KrapivinAMinVectorElemRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(15, "3"), std::make_tuple(50, "5"),
                                            std::make_tuple(1000, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KrapivinAMinVectorElemMPI, InType>(kTestParam, PPC_SETTINGS_krapivin_a_min_vector_elem),
    ppc::util::AddFuncTask<KrapivinAMinVectorElemSEQ, InType>(kTestParam, PPC_SETTINGS_krapivin_a_min_vector_elem));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    KrapivinAMinVectorElemRunFuncTestsProcesses::PrintFuncTestName<KrapivinAMinVectorElemRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KrapivinAMinVectorElemRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace krapivin_a_min_vector_elem
