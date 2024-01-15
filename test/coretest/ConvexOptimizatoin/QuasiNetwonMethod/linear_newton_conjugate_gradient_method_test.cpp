#include "core/ConvexOptimization/QuasiNetwonMethod/linear_newton_conjugate_gradient_method.h"

#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <memory>

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

double square(double x, double y) {
  return 1.5 * x * x + 0.5 * y * y - x * y - 2 * x;
}
double square_three_dimensional(double x, double y, double z) {
  return (x - 2) * (x - 3) + 2 * x + 2 * x * y + y * y + (z + 1) * (z + 1);
}
double f(double x, double y) {
  return (1 - x) * (1 - x) + std::fabs(y - x * x);
}

TEST(FUNCTIONTEST, step) {
  const double iteration_error = 1e-4;
  Eigen::VectorXd init_pos(2);
  init_pos << 109090, -400;

  Eigen::VectorXd b(2);
  b << 2, 0;

  Eigen::Matrix2d A;
  A << 3, -1, -1, 1;

  std::shared_ptr<LinearNewtonConjugateGradientMethod<double, 2, 6>> bfgs_ptr =
      std::make_shared<LinearNewtonConjugateGradientMethod<double, 2, 6>>(
          init_pos, b, A, iteration_error);

  bfgs_ptr->Step<decltype(square)>(square);

  auto res = bfgs_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization