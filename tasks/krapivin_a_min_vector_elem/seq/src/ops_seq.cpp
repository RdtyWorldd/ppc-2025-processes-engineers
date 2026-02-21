#include "krapivin_a_min_vector_elem/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "krapivin_a_min_vector_elem/common/include/common.hpp"
#include "util/include/util.hpp"

namespace krapivin_a_min_vector_elem {

KrapivinAMinVectorElemSEQ::KrapivinAMinVectorElemSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrapivinAMinVectorElemSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool KrapivinAMinVectorElemSEQ::PreProcessingImpl() {
  return !GetInput().empty();
}

bool KrapivinAMinVectorElemSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int result = GetInput()[0];

  for (size_t i = 1; i < GetInput().size(); i++) {
    if (result > GetInput()[i]) {
      result = GetInput()[i];
    }
  }

  GetOutput() = result;
  return true;
}

bool KrapivinAMinVectorElemSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace krapivin_a_min_vector_elem
