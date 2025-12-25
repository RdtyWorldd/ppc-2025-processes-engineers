#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace morozov_n_siedels_method {

using InType = std::tuple<std::size_t, std::vector<double>, std::vector<double>, double>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace morozov_n_siedels_method
