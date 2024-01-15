#include "core/ConvexOptimization/QuasiNetwonMethod/truncated_newton_cg.h"

#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <memory>

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

double square(double x, double y) { return x * x + y * y - 2 * x + 4 * y + 10; }
double square_three_dimensional(double x, double y, double z) {
  return (x - 2) * (x - 3) + 2 * x + 2 * x * y + y * y + (z + 1) * (z + 1);
}
double f(double x, double y) {
  return (1 - x) * (1 - x) + std::fabs(y - x * x);
}

TEST(FUNCTIONTEST, step) {
  const double iteration_error = 1e-4;
  const double step = 10;
  Eigen::VectorXd init_pos(2);
  init_pos << 109090, -400;

  Eigen::VectorXd b(2);
  b << 2, -4;

  Eigen::Matrix2d A;
  A << 2, 0, 0, 2;

  std::shared_ptr<
      TruncatedNewtonConjugateGradientMethodWithArmijo<double, 2, 6>>
      bfgs_ptr = std::make_shared<
          TruncatedNewtonConjugateGradientMethodWithArmijo<double, 2, 6>>(
          init_pos, b, A, iteration_error, step);

  bfgs_ptr->Step<decltype(square)>(square);

  auto res = bfgs_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization