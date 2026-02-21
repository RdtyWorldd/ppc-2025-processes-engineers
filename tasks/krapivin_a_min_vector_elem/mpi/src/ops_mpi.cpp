#include "krapivin_a_min_vector_elem/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "krapivin_a_min_vector_elem/common/include/common.hpp"
#include "util/include/util.hpp"

namespace krapivin_a_min_vector_elem {

KrapivinAMinVectorElemMPI::KrapivinAMinVectorElemMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrapivinAMinVectorElemMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool KrapivinAMinVectorElemMPI::PreProcessingImpl() {
  return !GetInput().empty();
}

bool KrapivinAMinVectorElemMPI::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::vector<int> input;
  std::vector<int> send_counts(mpi_size, 0);
  std::vector<int> displacements(mpi_size, 0);
  std::vector<int> recv_buf(mpi_size, 0);

  SplitData(input, send_counts, displacements, rank, mpi_size);
  int local_result = FindMin(input);

  MPI_Gather(&local_result, 1, MPI_INT, recv_buf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    GetOutput() = FindMin(recv_buf);
  }
  return true;
}

bool KrapivinAMinVectorElemMPI::PostProcessingImpl() {
  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int result = GetOutput();
  MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = result;

  return true;
}

void KrapivinAMinVectorElemMPI::SplitData(std::vector<int> &input, std::vector<int> &send_counts,
                                          std::vector<int> &displacements, int rank, int mpi_size) {
  int n = 0;
  if (rank == 0) {
    input = GetInput();
    n = static_cast<int>(input.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int step = n / mpi_size;
  int remainder = n % mpi_size;

  for (auto &elem : send_counts) {
    elem = step;
  }

  for (int i = 0; i < remainder; i++) {
    send_counts[i]++;
  }

  displacements[0] = 0;
  int disp_sum = 0;
  for (int i = 1; i < mpi_size; i++) {
    disp_sum += send_counts[i - 1];
    displacements[i] = disp_sum;
  }

  if (rank != 0) {
    input.resize(send_counts[rank]);
  }

  MPI_Scatterv(input.data(), send_counts.data(), displacements.data(), MPI_INT, input.data(),
               static_cast<int>(input.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

int KrapivinAMinVectorElemMPI::FindMin(const std::vector<int> &v) {
  int result = v[0];

  for (size_t i = 1; i < v.size(); i++) {
    if (result > v[i]) {
      result = v[i];
    }
  }

  return result;
}
}  // namespace krapivin_a_min_vector_elem
