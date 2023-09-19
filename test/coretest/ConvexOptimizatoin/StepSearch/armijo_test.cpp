#include <gtest/gtest.h>

#include <array>

#include "core/ConvexOptimization/StepSearch/armijo_condition.h"

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

float getValue(float x, float y) { return x * x + y * y; }

TEST(FUNCTIONTEST, ArmijoParameterCheck) { ArmijoParameterCheck(0.9); }

TEST(FUNCTIONTEST, LinearSearchWithArmijoCondition) {
  const float armijo_parameter = 0.9f;
  float step = 10.0f;

  std::array<float, 2> current_step_input_value{0.1, 0.1};
  std::array<float, 2> negative_gradients{-0.2, -0.2};

  LinearSearchWithArmijoCondition<decltype(getValue), float, 2>(
      armijo_parameter, step, current_step_input_value, negative_gradients,
      getValue);
  EXPECT_FLOAT_EQ(step, 0.078125);
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization