#include "core/ConvexOptimization/Constrained_Optimization/PenaltyMethod/penalty_method.h"

#include <gtest/gtest.h>

#include <cmath>
#include <eigen3/Eigen/Core>
#include <functional>
#include <memory>
#include <vector>

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

// double f(double x1, double x2) {
//   return std::pow(x1 - 2, 2) + std::pow(x2 - 1, 2);
// }

// double e_c(double x1, double x2) { return x1 + 2 * x2 - 2; }

// double ue_c(double x1, double x2) { return x1; }

double f(Eigen::Matrix<double, 2, 1> x) {
  return std::pow(x[0] - 2, 2) + std::pow(x[1] - 1, 2);
}

double e_c(Eigen::Matrix<double, 2, 1> x) { return x[0] + 2 * x[1] - 2; }

double ue_c(Eigen::Matrix<double, 2, 1> x) { return x[0]; }
double ue_c1(Eigen::Matrix<double, 2, 1> x) { return x[1] - 0.1; }

TEST(FUNCTIONTEST, Step) {
  const double iteration_error = 1e-3;
  const double step = 10;
  Eigen::VectorXd init_pos(2);
  init_pos << 0.1, 0.4;

  std::shared_ptr<PenaltyMethod<double, 2>> penalty_method_ptr =
      std::make_shared<PenaltyMethod<double, 2>>(init_pos, iteration_error,
                                                 step);
  std::vector<std::function<decltype(e_c)>> ev, uev;
  ev.emplace_back(e_c);
  uev.emplace_back(ue_c);
  uev.emplace_back(ue_c1);

  penalty_method_ptr
      ->Step<std::function<decltype(f)>, std::function<decltype(e_c)>,
             std::function<decltype(ue_c)>, 1, 2>(f, ev, uev);
  auto res = penalty_method_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization