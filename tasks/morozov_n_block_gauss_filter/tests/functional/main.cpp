#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstdint>
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
    return std::get<0>(test_param) + "_" + std::to_string(std::get<1>(test_param)) + "_" +
           std::to_string(std::get<2>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string &test_type = std::get<0>(params);
    int width = std::get<1>(params);
    int height = std::get<2>(params);

    if (test_type == "gen") {
      // Generate image with specified dimensions
      input_data_ = GenerateImage(width, height, (width * 1000) + height);
    } else {
      // Read image from file (width and height will be set from file)
      input_data_ = ReadImageFromTaskData(test_type + ".jpg");
    }
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
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<uint8_t> correct_data_;

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

  // Read image from file using stb_image
  static std::tuple<std::vector<uint8_t>, int, int> ReadImageFromFile(const std::string &filename) {
    int width = -1;
    int height = -1;
    int channels = -1;

    auto *data = stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb);
    if (data == nullptr) {
      throw std::runtime_error("Failed to load image: " + filename + " - " + std::string(stbi_failure_reason()));
    }

    const int expected_channels = 3;
    std::vector<uint8_t> img(data, data + (static_cast<ptrdiff_t>(width * height * expected_channels)));
    stbi_image_free(data);

    return std::make_tuple(img, width, height);
  }

  // Read image from task data directory
  static std::tuple<std::vector<uint8_t>, int, int> ReadImageFromTaskData(const std::string &relative_path) {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_morozov_n_block_gauss_filter, relative_path);
    return ReadImageFromFile(abs_path);
  }
};

namespace {

TEST_P(MorozovNBlockGaussFilterFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple("img_1", 0, 0),    // Read from file img_1.jpg
    std::make_tuple("gen", 50, 50),    // Generated 50x50 image
    std::make_tuple("gen", 100, 100),  // Generated 100x100 image
    std::make_tuple("gen", 150, 150),  // Generated 150x150 image
    std::make_tuple("gen", 200, 200)   // Generated 200x200 image
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MorozovNBlockGaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_morozov_n_block_gauss_filter),
    ppc::util::AddFuncTask<MorozovNBlockGaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_morozov_n_block_gauss_filter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    MorozovNBlockGaussFilterFuncTestsProcesses::PrintFuncTestName<MorozovNBlockGaussFilterFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, MorozovNBlockGaussFilterFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace morozov_n_block_gauss_filter
