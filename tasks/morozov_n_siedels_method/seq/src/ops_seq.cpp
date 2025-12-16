#include "morozov_n_siedels_method/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>
#include <algorithm>
#include <tuple>

#include "morozov_n_siedels_method/common/include/common.hpp"
#include "util/include/util.hpp"

namespace morozov_n_siedels_method {

MorozovNSiedelsMethodSEQ::MorozovNSiedelsMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MorozovNSiedelsMethodSEQ::ValidationImpl() {
  return true;
}

bool MorozovNSiedelsMethodSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovNSiedelsMethodSEQ::RunImpl() {
  int n = std::get<0>(GetInput());
  std::vector<double>& a = std::get<1>(GetInput());
  std::vector<double>& b = std::get<2>(GetInput());
  double eps = std::get<3>(GetInput());

  std::vector<double> x(n, 0);
  std::vector<double> iter_eps(n, -1);
  do { 
    for(int i = 0; i < n; i++) {
      double iter_x = b[i];
      for(int j = 0; j < i; j++){
        iter_x = iter_x - a[(i * n) + j] * x[j];
      }
      for(int j = i + 1; j < n; j++) {
        iter_x = iter_x - a[(i * n) + j] * x[j];
      }
      iter_x = iter_x / a[(i * n) + i];
      
      //обновление погрешности
      iter_eps[i] = std::fabs(iter_x - x[i]);
      //обновление полученного корня
      x[i] = iter_x;
    }
  } while (InEpsBound(iter_eps, eps));
  
  for(size_t i = 0; i < x.size(); i++) {
    std::cout << x[i] << " ";
  }
  std::cout <<"\n";
  GetOutput() = x;
  return true;
}

bool MorozovNSiedelsMethodSEQ::PostProcessingImpl() {
  return true;
}

bool MorozovNSiedelsMethodSEQ::InEpsBound(std::vector<double>& iter_eps, double correct_eps) {
  double max_in_iter = *(std::max_element(begin(iter_eps), end(iter_eps)));
  return max_in_iter > correct_eps;
}
}  // namespace morozov_n_siedels_method
