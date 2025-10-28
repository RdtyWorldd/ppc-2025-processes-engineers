#include "morozov_n_sentence_count/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>
#include <string>

#include "morozov_n_sentence_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace morozov_n_sentence_count {

MorozovNSentenceCountSEQ::MorozovNSentenceCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MorozovNSentenceCountSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool MorozovNSentenceCountSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovNSentenceCountSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }
  
  std::string s = GetInput();
  int counter = 0;
  for (size_t i = 0; i < s.length(); i++) {
    if((s[i] == '.') && (s[i-1] != '.') && (s[i-1] != '?') && (s[i-1] != '!')) {
        counter++;
    }
    else if((s[i] == '!') && (s[i-1] != '.') && (s[i-1] != '?') && (s[i-1] != '!')) {
      counter++;
    }
    else if((s[i] == '?') && (s[i-1] != '.') && (s[i-1] != '?') && (s[i-1] != '!')) {
      counter++;
    }
  }

  if (counter != 0) {
    GetOutput() = counter;
  }
  return GetOutput() > 0;
}

bool MorozovNSentenceCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace morozov_n_sentence_count
