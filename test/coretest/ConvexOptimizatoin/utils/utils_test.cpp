#include "core/ConvexOptimization/utils/utils.h"

#include <gtest/gtest.h>

#include <array>

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

TEST(FUNCTIONTEST, GetNextStepInputValue) {
  std::array<float, 3> init_input_value{1.0, 2.0, 3.0};
  std::array<float, 3> gradients{1.0, 2.0, 3.0};
  float step = 2.0;
  std::array<float, 3> res =
      utils::GetNextStepInputValue<float, 3>(init_input_value, gradients, step);

  EXPECT_FLOAT_EQ(res[0], 3.0);
  EXPECT_FLOAT_EQ(res[1], 6.0);
  EXPECT_FLOAT_EQ(res[2], 9.0);
}

TEST(FUNCTIONTEST, GetDotProductOfTwoArray) {
  std::array<float, 3> init_input_value{1.0, 2.0, 3.0};
  std::array<float, 3> gradients{1.1, 2.2, 3.3};
  auto res =
      utils::GetDotProductOfTwoArray<float, 3>(init_input_value, gradients);
  const float expected_res = 15.4f;
  EXPECT_FLOAT_EQ(res, expected_res);

  std::array<float, 3> first{1.1, 0.01, 10.01};
  std::array<float, 3> second{1.1, 2.2, -30.78};
  auto res_new = utils::GetDotProductOfTwoArray<float, 3>(first, second);
  const float expected_res_new = -306.8758f;
  EXPECT_FLOAT_EQ(res_new, expected_res_new);
}

TEST(FUNCTIONTEST, GetArraySelfMultiLambda) {
  std::array<float, 3> init_input_value{1.1, 2.2, 3.3};
  const float lambda = 3.3;
  utils::GetArraySelfMultiLambda<float, float, 3>(lambda, init_input_value);
  EXPECT_FLOAT_EQ(init_input_value[0], 3.63);
  EXPECT_FLOAT_EQ(init_input_value[1], 7.26);
  EXPECT_FLOAT_EQ(init_input_value[2], 10.89);
}

TEST(FUNCTIONTEST, GetArrayMultiLambda) {
  std::array<float, 3> init_input_value{1.1, 2.2, 3.3};
  const float lambda = 3.3;
  auto res =
      utils::GetArrayMultiLambda<float, float, 3>(lambda, init_input_value);
  EXPECT_FLOAT_EQ(res[0], 3.63);
  EXPECT_FLOAT_EQ(res[1], 7.26);
  EXPECT_FLOAT_EQ(res[2], 10.89);
}

TEST(FUNCTIONTEST, SelfAddArray) {
  std::array<float, 3> init_input_value{1.1, 2.2, 3.3};
  const std::array<float, 3> added_input_value{1.1, 2.2, 3.3};

  utils::SelfAddArray<float, 3>(added_input_value, init_input_value);
  EXPECT_FLOAT_EQ(init_input_value[0], 2.2);
  EXPECT_FLOAT_EQ(init_input_value[1], 4.4);
  EXPECT_FLOAT_EQ(init_input_value[2], 6.6);
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization