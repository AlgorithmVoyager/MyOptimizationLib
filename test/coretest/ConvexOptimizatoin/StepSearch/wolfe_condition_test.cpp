#include "core/ConvexOptimization/StepSearch/wolfe_condition.h"

#include <gtest/gtest.h>

#include <eigen3/Eigen/Dense>
#include <vector>

#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

float getValue(float x, float y) { return x * x + y * y; }

TEST(FUNCTIONTEST, WolfeParameterCheck) {
  WolfeConditionParameterCheck(1e-4, 0.9);
}

TEST(FUNCTIONTEST, LinearSearchForEigenVectorWithWeakWolfeCondition) {
  float step = 10.0f;

  Eigen::Vector2f init_pose, search_direction;
  init_pose << 1, 1;
  search_direction << -4, -4;

  LinearSearchForEigenVectorWithWeakWolfeCondition<decltype(getValue), float,
                                                   2>(
      init_pose, search_direction, step, getValue);

  MLOG_ERROR("step " << step);
  EXPECT_FLOAT_EQ(step, 0.3125);
}

TEST(FUNCTIONTEST, LinearSearchForEigenVectorWithStrongWolfeCondition) {
  float step = 10.0f;

  Eigen::Vector2f init_pose, search_direction;
  init_pose << 1, 1;
  search_direction << -4, -4;

  LinearSearchForEigenVectorWithStrongWolfeCondition<decltype(getValue), float,
                                                     2>(
      init_pose, search_direction, step, getValue);

  MLOG_ERROR("step " << step);
  EXPECT_FLOAT_EQ(step, 0.3125);
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization