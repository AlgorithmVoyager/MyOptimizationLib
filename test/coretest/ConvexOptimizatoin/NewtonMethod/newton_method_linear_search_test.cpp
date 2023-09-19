#include "core/ConvexOptimization/NewtonMethod/newton_method_linear_search.h"

#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <memory>

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

double square(double x, double y) { return (x - 1) * (x - 1) + y * y; }
double square_three_dimensional(double x, double y, double z) {
  return x * x + (y - 1) * (y - 1) * (y - 1) * (y - 1) + (z + 1) * (z + 1);
}

TEST(FUNCTIONTEST, STEP) {
  const double iteration_error = 1e-3;
  const float parameter = 0.1;
  const double x_epsilon = 1e-3;
  const double y_epsilon = 1e-3;
  const double step = 10.0;
  Eigen::VectorXd init_pos(2);
  init_pos << 0.1, 0.4;

  std::shared_ptr<NetwonMethodLinearSearch<double, 2>> newton_method_ptr =
      std::make_shared<NetwonMethodLinearSearch<double, 2>>(
          iteration_error, parameter, step, init_pos);

  newton_method_ptr->Step<decltype(square)>(x_epsilon, y_epsilon, square);

  auto res = newton_method_ptr->GetSearchResults();
  MLOG_ERROR("newton method linear search res " << res);

  Eigen::VectorXd expected_res(2);
  expected_res << 1.0, 0.0;

  for (auto i = 0; i < 2; ++i) {
    EXPECT_LT(std::fabs(res[i] - expected_res[i]), 1e-1);
  }
}

TEST(FUNCTIONTEST, StepThreeDimensional) {
  const double iteration_error = 1e-3;
  const float parameter = 0.2;
  const double x_epsilon = 1e-3;
  const double y_epsilon = 1e-3;
  const double step = 10.0;
  Eigen::VectorXd init_pos(3);
  init_pos << 1, 2, 3;

  std::shared_ptr<NetwonMethodLinearSearch<double, 3>> newton_method_ptr =
      std::make_shared<NetwonMethodLinearSearch<double, 3>>(
          iteration_error, parameter, step, init_pos);

  newton_method_ptr->Step<decltype(square_three_dimensional)>(
      x_epsilon, y_epsilon, square_three_dimensional);

  auto res = newton_method_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);

  Eigen::VectorXd expected_res(3);
  expected_res << 0.0, 1.0, -1.0;

  for (auto i = 0; i < 3; ++i) {
    EXPECT_LT(std::fabs(res[i] - expected_res[i]), 1e-1);
  }
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization