#include <gtest/gtest.h>

#include <eigen3/Eigen/Dense>
#include <vector>

#include "core/ConvexOptimization/StepSearch/lewis_overton_condition.h"
#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {
namespace {

float getValue(float x, float y) { return x * x + y * y; }

TEST(FUNCTIONTEST, LevisAndOvertonConditionParameterCheck) {
  LevisAndOvertonConditionParameterCheck(1e-4, 0.9);
}

TEST(FUNCTIONTEST, LinearSearchForEigenVectorWithLevisAndOvertonCondition) {
  float step = 1000.0f;

  Eigen::Vector2f init_pose, search_direction;
  init_pose << 1, 1;
  search_direction << -2, -2;

  LinearSearchForEigenVectorWithLevisAndOvertonCondition<decltype(getValue),
                                                         float, 2>(
      init_pose, search_direction, step, getValue);

  MLOG_ERROR("step " << step);
  EXPECT_LT(std::fabs(step - 0.97656202), 1e-3);
}

}  // namespace
}  // namespace ConvexOptimization
}  // namespace MyOptimization