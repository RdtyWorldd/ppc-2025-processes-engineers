#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include "morozov_n_siedels_method/common/include/common.hpp"
#include "morozov_n_siedels_method/mpi/include/ops_mpi.hpp"
#include "morozov_n_siedels_method/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozov_n_siedels_method {

class MorozovNSiedelsMethodPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  std::vector<double> correct_data_;
  double task_eps_ = 0.000001;
  // double global_eps_ = 1e-9;
  int seed_ = 777;
  std::size_t n_ = 2000;

  void SetUp() override {
    GenerateTestData(n_, seed_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    for (std::size_t i = 0; i < output_data.size(); i++) {
      if (std::fabs((output_data[i] - correct_data_[i])) > task_eps_) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
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
          row_sum += std::fabs(a[(i * n) + j]);
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
};

TEST_P(MorozovNSiedelsMethodPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MorozovNSiedelsMethodMPI, MorozovNSiedelsMethodSEQ>(
    PPC_SETTINGS_morozov_n_siedels_method);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MorozovNSiedelsMethodPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MorozovNSiedelsMethodPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace morozov_n_siedels_method
