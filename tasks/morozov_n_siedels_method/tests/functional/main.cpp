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
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    task_eps_ = std::get<2>(params);
    if (std::get<1>(params) == "gen") {
      std::size_t n = std::get<0>(params);
      GenerateTestData(n, seed_);
    } else {
      GetTestFromFile(params);
    }
    // std::vector<double> a({6.1, 2.2, 1.2, 2.2, 5.5, -1.5, 1.2, -1.5, 7.2});
    // std::vector<double> b({16.55, 10.55, 16.80});
    // task_eps_ = 0.01;
    // input_data_ = std::make_tuple(3, a, b, 0.01);
    // correct_data_ = {1.5 , 2.0, 2.5};
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
    for (std::size_t i = 0; i < output_data.size(); i++) {
      if (abs((output_data[i] - correct_data_[i])) > task_eps_) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<double> correct_data_;
  double task_eps_ = 0.0;
  // double global_eps_ = 1e-9;
  int seed_ = 777;

  void GenerateTestData(std::size_t n, int seed) {
    std::vector<double> x(n, 0.0);
    std::vector<double> a(n * n, 0.0);
    std::vector<double> b(n, 0.0);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist_coeff(0.0, 1.0);
    std::uniform_real_distribution<double> dist_solution(-10.0, 10.0);

    for (std::size_t i = 0; i < n; i++) {
      x[i] = dist_solution(gen);
    }

    // debug
    //  for(int i = 0; i < n; i++){
    //     std::cout << x[i] << " ";
    //  }
    //  std::cout << "\n\n";

    // Генерируем матрицу с диагональным преобладанием
    for (std::size_t i = 0; i < n; i++) {
      double row_sum = 0.0;
      for (std::size_t j = 0; j < n; j++) {
        if (i != j) {
          a[(i * n) + j] = dist_coeff(gen);
          row_sum += std::abs(a[(i * n) + j]);
        }
      }
      a[(i * n) + i] = row_sum + 1.0 + dist_coeff(gen);  // гарантируем преобладание

      // debug
      //  for (int j = 0; j < n; j++) {
      //      std::cout << a[i * n + j] << " ";
      //  }
      //  std::cout << "\n";
    }

    // Вычисляем правую часть
    for (std::size_t i = 0; i < n; i++) {
      b[i] = 0.0;
      for (std::size_t j = 0; j < n; j++) {
        b[i] += a[(i * n) + j] * x[j];
      }
    }
    // debug
    //  for(int i = 0;i < n; i++){
    //    std::cout << b[i] << " ";
    // }
    // std::cout << "\n\n";

    input_data_ = std::make_tuple(n, a, b, task_eps_);
    correct_data_ = x;
  }
  void GetTestFromFile(TestType &params) {
    std::string local = std::get<1>(params) + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_morozov_n_siedels_method, local);
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }
    std::size_t n = std::get<0>(params);
    std::vector<double> x(n, 0);
    std::vector<double> a(n * n, 0);
    std::vector<double> b(n, 0);
    for (std::size_t i = 0; i < n * n; i++) {
      file >> a[i];
    }
    for (std::size_t i = 0; i < n; i++) {
      file >> b[i];
    }
    for (std::size_t i = 0; i < n; i++) {
      file >> x[i];
    }

    input_data_ = std::make_tuple(n, a, b, task_eps_);
    correct_data_ = x;
  }
};

namespace {

TEST_P(MorozovNSiedelsMethodFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(4, "test_1", 0.01),
                                            std::make_tuple(4, "gen", 0.0001),
                                            std::make_tuple(16, "gen", 0.0001),
                                            /*std::make_tuple(100, "gen", 0.0001)*/};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MorozovNSiedelsMethodMPI, InType>(kTestParam, PPC_SETTINGS_morozov_n_siedels_method),
    ppc::util::AddFuncTask<MorozovNSiedelsMethodSEQ, InType>(kTestParam, PPC_SETTINGS_morozov_n_siedels_method));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MorozovNSiedelsMethodFuncTestsProcesses::PrintFuncTestName<MorozovNSiedelsMethodFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, MorozovNSiedelsMethodFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace morozov_n_siedels_method
