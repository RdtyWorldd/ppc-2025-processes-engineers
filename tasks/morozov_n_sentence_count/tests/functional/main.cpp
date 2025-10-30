#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include "morozov_n_sentence_count/common/include/common.hpp"
#include "morozov_n_sentence_count/mpi/include/ops_mpi.hpp"
#include "morozov_n_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace morozov_n_sentence_count {

class MorozovNRunSentenceCountTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + "file";
  }

 protected:
  void SetUp() override {
    std::string text = "";
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    
    //Read text from params
    {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_morozov_n_sentence_count, std::get<1>(params));
      std::ifstream file(abs_path);
      std::stringstream ss;
      ss << file.rdbuf();
      text = ss.str();
    }
    
    task_answer = std::get<2>(params);
    input_data_ = text;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::cout << output_data << std::endl;
    return output_data == task_answer;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = "";
  std::size_t task_answer = 0;
};

namespace {

TEST_P(MorozovNRunSentenceCountTests, SentenceCountFromText) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "test_1.txt", 1), std::make_tuple(2, "test_2.txt", 4), 
                                            std::make_tuple(3, "test_3.txt", 100)};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<MorozovNSentenceCountMPI, InType>(kTestParam, PPC_SETTINGS_morozov_n_sentence_count),
                   ppc::util::AddFuncTask<MorozovNSentenceCountSEQ, InType>(kTestParam, PPC_SETTINGS_morozov_n_sentence_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = MorozovNRunSentenceCountTests::PrintFuncTestName<MorozovNRunSentenceCountTests>;

INSTANTIATE_TEST_SUITE_P(SentenceCountTest, MorozovNRunSentenceCountTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace morozov_n_sentence_count
