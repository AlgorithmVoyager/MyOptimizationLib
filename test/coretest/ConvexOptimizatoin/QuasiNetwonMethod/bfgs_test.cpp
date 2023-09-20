#include "core/ConvexOptimization/QuasiNetwonMethod/bfgs.h"

#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <memory>

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

double square(double x, double y) { return x * x + y * y; }
double square_three_dimensional(double x, double y, double z) {
  return (x - 1) * (x - 1) + y * y * y * y + (z + 1) * (z + 1);
}

TEST(FUNCTIONTEST, StepThreeDimensional) {
  const double iteration_error = 1e-4;
  const double parameter = 0.1;
  const double step = 1.0;
  Eigen::VectorXd init_pos(3);
  init_pos << 10, 20, 30;

  std::shared_ptr<BFGS<double, 3>> bfgs_ptr = std::make_shared<BFGS<double, 3>>(
      init_pos, iteration_error, parameter, step);

  bfgs_ptr->Step<decltype(square_three_dimensional)>(square_three_dimensional);

  auto res = bfgs_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);

  // Eigen::VectorXd expected_res(3);
  // expected_res << 0.0, 1.0, -1.0;

  // for (auto i = 0; i < 3; ++i) {
  //   EXPECT_LT(std::fabs(res[i] - expected_res[i]), 1e-1);
  // }
}
}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization