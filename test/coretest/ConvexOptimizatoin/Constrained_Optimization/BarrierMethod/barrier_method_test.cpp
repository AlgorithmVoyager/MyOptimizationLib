#include "core/ConvexOptimization/Constrained_Optimization/BarrierMethod/barrier_method.h"

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
  return std::pow(x[0], 3) / 12 + x[1];
}

double ue_c(Eigen::Matrix<double, 2, 1> x) { return -x[0] + 1; }
double ue_c1(Eigen::Matrix<double, 2, 1> x) { return -x[1]; }

TEST(FUNCTIONTEST, Step) {
  const double iteration_error = 1e-3;
  const double step = 10;
  Eigen::VectorXd init_pos(2);
  init_pos << 0.1, 0.4;

  std::shared_ptr<BarrierMethod<double, 2>> barrier_method_ptr =
      std::make_shared<BarrierMethod<double, 2>>(init_pos, iteration_error,
                                                 step);
  std::vector<std::function<decltype(ue_c)>> uev;
  uev.emplace_back(ue_c);
  // uev.emplace_back(ue_c1);

  barrier_method_ptr
      ->Step<std::function<decltype(f)>, std::function<decltype(ue_c)>, 1>(f,
                                                                           uev);
  auto res = barrier_method_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization