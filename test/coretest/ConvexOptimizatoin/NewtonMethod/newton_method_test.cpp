#include "core/ConvexOptimization/NewtonMethod/newton_method.h"

#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <memory>

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

double square(double x, double y) { return x * x + y * y; }
double square_three_dimensional(double x, double y, double z) {
  return (x - 1) * (x - 1) + y * y * y + (z + 1) * (z + 1) * (z + 1);
}

// TEST(FUNCTIONTEST, UpdateSearchDirection) {
//   const double iteration_error = 1e-3;
//   const double epsilon = 1e-3;
//   Eigen::VectorXd init_pos(2);
//   init_pos << 2.0, 3.0;

//   std::shared_ptr<NetwonMethod<double, 2>> newton_method_ptr =
//       std::make_shared<NetwonMethod<double, 2>>(iteration_error, init_pos);

//   newton_method_ptr->UpdateSearchDirection<decltype(square)>(square,
//   epsilon);

//   auto search_direction = newton_method_ptr->GetSearchDirection();

//   Eigen::VectorXd expected_searched_pos(2);
//   expected_searched_pos << -4.0, -6.0;
//   for (auto i = 0; i < 2; i++) {
//     EXPECT_FLOAT_EQ(search_direction(i), expected_searched_pos(i));
//   }
// }

// TEST(FUNCTIONTEST, UpdateNewtonStep) {
//   const double iteration_error = 1e-3;
//   const double x_epsilon = 1e-3;
//   const double y_epsilon = 1e-3;
//   Eigen::VectorXd init_pos(2);
//   init_pos << 2.0, 3.0;

//   std::shared_ptr<NetwonMethod<double, 2>> newton_method_ptr =
//       std::make_shared<NetwonMethod<double, 2>>(iteration_error, init_pos);

//   newton_method_ptr->UpdateNewtonStep<decltype(square)>(square, x_epsilon,
//                                                         y_epsilon);

//   auto newton_step = newton_method_ptr->GetNewtonStep();

//   Eigen::Matrix2d expected_matrix_res(2, 2);
//   expected_matrix_res << 0.5, 0.0, 0.0, 0.5;

//   MLOG_ERROR("newton_step " << newton_step);

//   for (auto i = 0; i < 2; i++) {
//     for (auto j = 0; j < 2; j++) {
//       EXPECT_LT(std::fabs(newton_step(i, j) - expected_matrix_res(i, j)),
//       1e-1);
//     }
//   }
// }

TEST(FUNCTIONTEST, StepTwoDimensional) {
  const double iteration_error = 1e-3;
  const double x_epsilon = 1e-3;
  const double y_epsilon = 1e-3;
  Eigen::VectorXd init_pos(2);
  init_pos << 0.1, 0.4;

  std::shared_ptr<NetwonMethod<double, 2>> newton_method_ptr =
      std::make_shared<NetwonMethod<double, 2>>(iteration_error, init_pos);

  newton_method_ptr->Step<decltype(square)>(x_epsilon, y_epsilon, square);

  auto res = newton_method_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);

  Eigen::VectorXd expected_res(2);
  expected_res << 0.0, 0.0;

  for (auto i = 0; i < 2; ++i) {
    EXPECT_LT(std::fabs(res[i] - expected_res[i]), 1e-1);
  }
}

TEST(FUNCTIONTEST, StepThreeDimensional) {
  const double iteration_error = 1e-3;
  const double x_epsilon = 1e-3;
  const double y_epsilon = 1e-3;
  Eigen::VectorXd init_pos(3);
  init_pos << 1, 4, 43;

  std::shared_ptr<NetwonMethod<double, 3>> newton_method_ptr =
      std::make_shared<NetwonMethod<double, 3>>(iteration_error, init_pos);

  newton_method_ptr->Step<decltype(square_three_dimensional)>(
      x_epsilon, y_epsilon, square_three_dimensional);

  auto res = newton_method_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);

  Eigen::VectorXd expected_res(3);
  expected_res << 1.0, 0.0, -1.0;

  for (auto i = 0; i < 3; ++i) {
    EXPECT_LT(std::fabs(res[i] - expected_res[i]), 1e-1);
  }
}
}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization