#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "morozov_n_block_gauss_filter/common/include/common.hpp"
#include "morozov_n_block_gauss_filter/mpi/include/ops_mpi.hpp"
#include "morozov_n_block_gauss_filter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace morozov_n_block_gauss_filter {

class MorozovNBlockGaussFilterFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::vector<uint8_t> img(5 * 5 * 3, 0);
    int count = 0;
    for (int i = 0; i < 5; i++) {
      for (int j = 0; j < 5; j++) {
        for (int k = 0; k < 3; k++) {
          img[3 * ((i * 5) + j) + k] = count;
          count++;
        }
      }
    }
    input_data_ = std::make_tuple(img, 5, 5);
    CalcCorrectData();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // debug
    // if (output_data.size() <= 16) {
    //   std::string result;
    //   for (std::size_t i = 0; i < output_data.size(); i++) {
    //     result += std::to_string(correct_data_[i]) + " ";
    //   }
    //   result += '\n';
    //   std::cout << result;
    // }
    // for (std::size_t i = 0; i < output_data.size(); i++) {
    //   if (abs((output_data[i] - correct_data_[i])) > task_eps_) {
    //     return false;
    //   }
    // }
    if(output_data.size() != correct_data_.size()){
      return false;
    }
    for(size_t i = 0; i < output_data.size(); i++) {
      if(output_data[i] != correct_data_[i]) {
        return false;
      }
    }
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<uint8_t> correct_data_;
  // double global_eps_ = 1e-9;
  int seed_ = 777;

  void CalcCorrectData() {
    MorozovNBlockGaussFilterSEQ task(input_data_);
    task.Validation();
    task.PreProcessing();
    task.Run();
    task.PostProcessing();

    correct_data_ = task.GetOutput();
  }
  // void GenerateTestData(std::size_t n, int seed) {}
  // void GetTestFromFile(TestType &params) {}
};

namespace {

TEST_P(MorozovNBlockGaussFilterFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {std::make_tuple(4, "test_1", 0.01)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MorozovNBlockGaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_morozov_n_block_gauss_filter),
    ppc::util::AddFuncTask<MorozovNBlockGaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_morozov_n_block_gauss_filter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MorozovNBlockGaussFilterFuncTestsProcesses::PrintFuncTestName<MorozovNBlockGaussFilterFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, MorozovNBlockGaussFilterFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace morozov_n_block_gauss_filter
