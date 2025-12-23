#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "morozov_n_block_gauss_filter/common/include/common.hpp"
#include "morozov_n_block_gauss_filter/mpi/include/ops_mpi.hpp"
#include "morozov_n_block_gauss_filter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozov_n_block_gauss_filter {

class MorozovNBlockGaussFilterPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  std::vector<uint8_t> correct_data_;
  // double global_eps_ = 1e-9;
  int seed_ = 777;
  int height_ = 2000;
  int width_ = 2000;

  void SetUp() override {
    // GenerateTestData(n_, seed_);
    input_data_ = GenerateImage(width_, height_, seed_);
    CalcCorrectData();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != correct_data_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); i++) {
      if (output_data[i] != correct_data_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  void CalcCorrectData() {
    MorozovNBlockGaussFilterSEQ task(input_data_);
    task.Validation();
    task.PreProcessing();
    task.Run();
    task.PostProcessing();

    correct_data_ = task.GetOutput();
  }

  // Generate a test image with specified dimensions and seed
  static std::tuple<std::vector<uint8_t>, int, int> GenerateImage(int width, int height, int seed) {
    if (width <= 0 || height <= 0) {
      throw std::invalid_argument("Image dimensions must be positive");
    }
    std::vector<uint8_t> img(static_cast<size_t>(width * height * 3), 0);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 255);

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        int pixel_idx = 3 * ((i * width) + j);
        img[pixel_idx + 0] = static_cast<uint8_t>(dis(gen));  // R
        img[pixel_idx + 1] = static_cast<uint8_t>(dis(gen));  // G
        img[pixel_idx + 2] = static_cast<uint8_t>(dis(gen));  // B
      }
    }
    return std::make_tuple(img, width, height);
  }
};

TEST_P(MorozovNBlockGaussFilterPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MorozovNBlockGaussFilterMPI, MorozovNBlockGaussFilterSEQ>(
        PPC_SETTINGS_morozov_n_block_gauss_filter);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MorozovNBlockGaussFilterPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MorozovNBlockGaussFilterPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace morozov_n_block_gauss_filter
