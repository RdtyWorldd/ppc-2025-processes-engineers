#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace morozov_n_sentence_count {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<int, std::string, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace morozov_n_sentence_count
