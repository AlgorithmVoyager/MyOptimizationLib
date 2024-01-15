#include "core/ConvexOptimization/QuasiNetwonMethod/usage/path_generation.h"

#include <memory>

#include "logging/log.h"

int main() {
  int constexpr obs_size = 3;

  auto path_opt =
      MyOptimization::ConvexOptimization::PathOptimization<double, obs_size>();

  path_opt.GenerateCoefficient();
  MyOptimization::ConvexOptimization::Path<double> path;
  path_opt.GeneratePath(path);

  //   auto res = path_opt.GetCoeffVector();

  //   CoeffVector init_coeff = Eigen::Matrix<T, 8 * N, 1>::Ones(8 * N);

  //   const double iteration_error = 1e-4;
  //   const double parameter = 0.1;
  //   const double step = 10000.0;

  //   std::shared_ptr<LimitMemoryCautiousBFGSWithLewis<double, 8 * N, 6>>
  //   bfgs_ptr =
  //       std::make_shared<LimitMemoryCautiousBFGSWithLewis<double, 8 * N, 6>>(
  //           init_coeff, iteration_error, step);
  //   bfgs_ptr->Step(getOptimizationFunction);

  //   coeff_vec_.noalias() = bfgs_ptr->GetSearchResults();
  //   MLOG_ERROR("coeff_vec_ " << coeff_vec_);

  //   MLOG_ERROR("res " << res);

  return 0;
}