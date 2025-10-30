#include "morozov_n_sentence_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "morozov_n_sentence_count/common/include/common.hpp"
#include "util/include/util.hpp"

namespace morozov_n_sentence_count {

MorozovNSentenceCountMPI::MorozovNSentenceCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MorozovNSentenceCountMPI::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool MorozovNSentenceCountMPI::PreProcessingImpl() {
  if(GetInput()[0] == '.' || GetInput()[0] == '!' || GetInput()[0] == '?')
  {
    GetInput()[0] = ' ';
  }
  return true;
}

bool MorozovNSentenceCountMPI::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }
  std::string input = GetInput();

  int mpi_size = 0;
  int rank = 0;

  // int start_with = 0;
  // int end_with = 0;

  std::size_t index_start = 0;
  std::size_t index_end = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);

  std::size_t step = input.length() / mpi_size;
  std::size_t mod = input.length() - step * mpi_size;

  if(2 * mod >= step) {
    step = step + mod;
  }
  index_start = step * rank;
  index_end = step * (rank + 1);

  if(rank == mpi_size - 1) {
    index_end = input.length();
  }

  std::size_t counter = 0;
  for (std::size_t i = index_start; i < index_end; i++) {
    if((input[i] == '.') && (input[i-1] != '.') && (input[i-1] != '?') && (input[i-1] != '!')) {
        counter++;
    }
    else if((input[i] == '!') && (input[i-1] != '.') && (input[i-1] != '?') && (input[i-1] != '!')) {
      counter++;
    }
    else if((input[i] == '?') && (input[i-1] != '.') && (input[i-1] != '?') && (input[i-1] != '!')) {
      counter++;
    }
  }

  const std::size_t k_counter = counter;
  std::size_t counter_sum = 0;
  MPI_Reduce(&k_counter, &counter_sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&counter_sum, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  
  GetOutput() = counter_sum;
  return GetOutput() > 0;
}

bool MorozovNSentenceCountMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace morozov_n_sentence_count
