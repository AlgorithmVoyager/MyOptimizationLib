#include "core/ConvexOptimization/SteepestGradientDescent/steepest_gradient_descent.h"
#include <gtest/gtest.h>
#include <memory>

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

float square(float x) { return x * x; }
float g_square(float x) { return 2 * x; }

float two_value_square(float x, float y) { return x * x + y * y; }
float two_differen_square(float x, float y) {
  return x * x + 2 * x + 1 + y * y;
}

// TEST(FUNCTIONTEST, ConstructorParameterTest) {

//   const float step{10.0};
//   const float parameter{0.9};
//   const float iteration_error{1e-3};
//   std::array<float, 1> init_pos{10};

//   std::function<decltype(square)> func = square;
//   SteepestGradientDescent<float, 1> steepest_gradient_descent_entity(
//       step, parameter, iteration_error, init_pos);
// }

// TEST(FUNCTIONTEST, ChooseGradientDescentSignalValue) {

//   const float step{10.0};
//   const float parameter{0.9};
//   const float iteration_error{1e-3};
//   std::array<float, 1> init_pos{10};

//   std::shared_ptr<SteepestGradientDescent<float, 1>> sgd_ptr_ =
//       std::make_shared<SteepestGradientDescent<float, 1>>(
//           step, parameter, iteration_error, init_pos);
//   sgd_ptr_->ChooseSerachDirection<float, decltype(square)>(-1, square);

//   auto res = sgd_ptr_->GetSearchDirection();

//   const float expect_search_direction = -20;
//   EXPECT_LT(std::fabs(res[0] - expect_search_direction), 1e-1);
// }

// TEST(FUNCTIONTEST, UpdateIteratorSignalValue) {

//   const float step{10.0};
//   const float parameter{0.9};
//   const float iteration_error{1e-3};
//   std::array<float, 1> init_pos{10};

//   std::shared_ptr<SteepestGradientDescent<float, 1>> sgd_ptr_ =
//       std::make_shared<SteepestGradientDescent<float, 1>>(
//           step, parameter, iteration_error, init_pos);
//   sgd_ptr_->ChooseSerachDirection<float, decltype(square)>(-1, square);

//   sgd_ptr_->UpdateIterate();
//   auto res = sgd_ptr_->GetSearchResult();

//   const float expect_search_direction = -190;
//   EXPECT_LT(std::fabs(res[0] - expect_search_direction), 2 * 1e-1);
// }

// TEST(FUNCTIONTEST, StatisfyGradientDescentFinishConditionSignalValue) {

//   const float step{10.0};
//   const float parameter{0.9};
//   const float iteration_error{1e-3};
//   std::array<float, 1> init_pos{10};

//   std::shared_ptr<SteepestGradientDescent<float, 1>> sgd_ptr_ =
//       std::make_shared<SteepestGradientDescent<float, 1>>(
//           step, parameter, iteration_error, init_pos);
//   bool flag_false =
//       sgd_ptr_->StatisfyGradientDescentFinishCondition<decltype(square)>(
//           init_pos, square);
//   EXPECT_FALSE(flag_false);

//   // update search direction
//   sgd_ptr_->ChooseSerachDirection<float, decltype(square)>(-1, square);

//   sgd_ptr_->UpdateIterate();
//   auto last_iterate = sgd_ptr_->GetSearchDirection();

//   bool flag_true =
//       sgd_ptr_->StatisfyGradientDescentFinishCondition<decltype(square)>(
//           last_iterate, square);

//   EXPECT_TRUE(flag_true);
// }

TEST(FUNCTIONTEST, StepwithSignalValue) {
  const float step{10.0};
  const float parameter{0.9};
  const float iteration_error{1e-3};
  std::array<float, 1> init_pos{10};

  std::shared_ptr<SteepestGradientDescent<float, 1>> sgd_ptr_ =
      std::make_shared<SteepestGradientDescent<float, 1>>(
          step, parameter, iteration_error, init_pos);
  sgd_ptr_->Step<decltype(square)>(square);
  auto res_pos = sgd_ptr_->GetSearchResult();
  for (auto res : res_pos) {
    MLOG_INFO("final res pos " << res);
  }
  const std::array<float, 1> expected_search_res{0.0};
  EXPECT_LT(std::fabs(res_pos[0] - expected_search_res[0]), 1e-1);
}

TEST(FUNCTIONTEST, StepwithTwoValue) {
  const float step{10.0};
  const float parameter{0.9};
  const float iteration_error{1e-3};
  std::array<float, 2> init_pos{10, 10};

  std::shared_ptr<SteepestGradientDescent<float, 2>> sgd_ptr_ =
      std::make_shared<SteepestGradientDescent<float, 2>>(
          step, parameter, iteration_error, init_pos);
  sgd_ptr_->Step<decltype(two_value_square)>(two_value_square);
  auto res_pos = sgd_ptr_->GetSearchResult();
  for (auto res : res_pos) {
    MLOG_INFO("final res pos " << res);
  }
  const std::array<float, 2> expected_search_res{0.0, 0.0};
  for (auto i = 0U; i < res_pos.size(); i++) {
    EXPECT_LT(std::fabs(res_pos[i] - expected_search_res[i]), 1e-1);
  }
}

TEST(FUNCTIONTEST, StepwithTwoDifferenValue) {
  const float step{10.0};
  const float parameter{0.9};
  const float iteration_error{1e-3};
  std::array<float, 2> init_pos{1, 0};

  std::shared_ptr<SteepestGradientDescent<float, 2>> sgd_ptr_ =
      std::make_shared<SteepestGradientDescent<float, 2>>(
          step, parameter, iteration_error, init_pos);
  sgd_ptr_->Step<decltype(two_differen_square)>(two_differen_square);
  auto res_pos = sgd_ptr_->GetSearchResult();
  for (auto res : res_pos) {
    MLOG_INFO("final res pos " << res);
  }
  const std::array<float, 2> expected_search_res{-1.0, 0};
  for (auto i = 0U; i < res_pos.size(); i++) {
    EXPECT_LT(std::fabs(res_pos[i] - expected_search_res[i]), 1e-1);
  }
}

} // namespace
} // namespace ConvexOptimization
} // namespace MyOptimization