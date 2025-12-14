#include "morozov_n_siedels_method/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>
#include <algorithm>

#include "morozov_n_siedels_method/common/include/common.hpp"
#include "util/include/util.hpp"
#include "ops_mpi.hpp"

namespace morozov_n_siedels_method {

MorozovNSiedelsMethodMPI::MorozovNSiedelsMethodMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MorozovNSiedelsMethodMPI::ValidationImpl() {
  return true;
}

bool MorozovNSiedelsMethodMPI::PreProcessingImpl() {
  return true;
}

bool MorozovNSiedelsMethodMPI::RunImpl() {
  double eps = std::get<3>(GetInput());

  int rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<double>& a = std::vector();
  std::vector<double>& b = std::vector();

  int total_elements = 0;
  if(rank == 0) {
    int n = std::get<0>(GetInput());
    total_elements = n;
    a = std::get<1>(GetInput());
    b = std::get<2>(GetInput());
  }
  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int step = total_elements / mpi_size;
  int remainder = total_elements % mpi_size;

  std::vector<int> send_counts(mpi_size, step);
  std::vector<int> displacements(mpi_size, 0);
  int displacement = 0;
  for (int i = 0; i < remainder; ++i) {
    send_counts[i]++;
    displacements[i]++;
  }
  
  displacements[0] = 0;
  for (int i = 1; i < mpi_size; ++i) {
    displacements[i] += step * i;
  }
  displacement = displacements[rank];

  std::vector<double> local_b(send_counts[rank]);
  MPI_Scatterv(b.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_b.data(), local_b.size(), MPI_DOUBLE,
      0, MPI_COMM_WORLD);

  for(int i = 0; i < mpi_size; i++) {
    send_counts[i] *= total_elements;
    displacements[i] *= total_elements;
  }
  std::vector<double> local_a(send_counts[rank])
  MPI_Scatterv(a.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_a.data(), local_a.size(), MPI_DOUBLE,
      0, MPI_COMM_WORLD);
  
  std::vector<double> x(total_elements, 0); //вектор ответа
  std::vector<double> x_new(local_b.size(), 0); //вектор для рассылки
  std::vector<double> iter_eps(total_elements, -1); //вектор погрешности
  std::vector<double> iter_eps_new(local_b.size(), -1);
  do { 
    for(int i = 0; i < local_b.size(); i++) {
      int exp_pos = displacement + i;  
      double iter_x = local_b[i];
      for(int j = 0; j < exp_pos; j++){
        iter_x = iter_x - local_a[(exp_pos * total_elements) + j] * x[j];
      }
      for(int j = exp_pos + 1; j < n; j++) {
        iter_x = iter_x - local_a[(exp_pos * total_elements) + j] * x[j];
      }
      iter_x = iter_x / local_a[(exp_pos * total_elements) + exp_pos];
      
      //обновление погрешности
      iter_eps[exp_pos] = std::fabs(iter_x - x[exp_pos]);
      iter_eps_new[exp_pos] = iter_eps[exp_pos];
      //обновление полученного корня
      x[exp_pos] = iter_x;
      x_new[i] = iter_x; 
    }
    MPI_Allgather(x_new.data(), x_new.size(), MPI_DOUBLE, x.data(), x.size(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(iter_eps_new.data(), iter_eps_new.size(), MPI_DOUBLE, iter_eps.data(), iter_eps.size(), MPI_DOUBLE, MPI_COMM_WORLD)
  } while (InEpsBound(iter_eps, eps));
  
  GetOutput() = x;
  return true;
}

bool MorozovNSiedelsMethodMPI::PostProcessingImpl() {
  return true;
}

bool MorozovNSiedelsMethodMPI::InEpsBound(std::vector<double> &iter_eps, double correct_eps) {
  double max_in_iter = *(std::max_element(begin(iter_eps), end(iter_eps)));
  return max_in_iter > correct_eps;
}
}  // namespace morozov_n_siedels_method
