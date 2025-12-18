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

#include "morozov_n_siedels_method/common/include/common.hpp"
#include "morozov_n_siedels_method/mpi/include/ops_mpi.hpp"
#include "morozov_n_siedels_method/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace morozov_n_siedels_method {

class MorozovNSiedelsMethodFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    std::vector<double> a({6.1, 2.2, 1.2, 2.2, 5.5, -1.5, 1.2, -1.5, 7.2});
    std::vector<double> b({16.55, 10.55, 16.80});
    input_data_ = std::make_tuple(3, a, b, 0.01);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::cout << "HUI\n";
    return output_data.size() > 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(MorozovNSiedelsMethodFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 1> kTestParam = {std::make_tuple(3, "3")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MorozovNSiedelsMethodMPI, InType>(kTestParam, PPC_SETTINGS_morozov_n_siedels_method),
    ppc::util::AddFuncTask<MorozovNSiedelsMethodSEQ, InType>(kTestParam, PPC_SETTINGS_morozov_n_siedels_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MorozovNSiedelsMethodFuncTestsProcesses::PrintFuncTestName<MorozovNSiedelsMethodFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, MorozovNSiedelsMethodFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace morozov_n_siedels_method
