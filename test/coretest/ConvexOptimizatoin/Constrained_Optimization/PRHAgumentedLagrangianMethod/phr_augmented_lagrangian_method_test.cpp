#include "core/ConvexOptimization/Constrained_Optimization/PRHAgumentedLagrangianMethod/phr_augmented_lagrangian_method.h"

#include <gtest/gtest.h>

#include <cmath>
#include <eigen3/Eigen/Core>
#include <functional>
#include <memory>
#include <vector>

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

double f(Eigen::Matrix<double, 2, 1> x) {
  return std::pow(x[0] - 2, 2) + std::pow(x[1] - 1, 2);
}

double e_c(Eigen::Matrix<double, 2, 1> x) { return x[0] + 2 * x[1] - 2; }

TEST(FUNCTIONTEST, Step) {
  const double iteration_error = 1e-3;
  const double step = 10;
  Eigen::VectorXd init_pos(2);
  init_pos << 0.1, 0.4;

  Eigen::VectorXd lambda(1);
  lambda << 1;

  std::shared_ptr<PHRAugmentedLagrangianMethod<double, 2, 1>>
      phr_alm_method_ptr =
          std::make_shared<PHRAugmentedLagrangianMethod<double, 2, 1>>(
              init_pos, lambda, iteration_error, step);
  std::vector<std::function<decltype(e_c)>> ev;
  ev.emplace_back(e_c);

  phr_alm_method_ptr
      ->Step<std::function<decltype(f)>, std::function<decltype(e_c)>>(f, ev);
  auto res = phr_alm_method_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization