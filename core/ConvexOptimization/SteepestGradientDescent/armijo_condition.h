#ifndef MYOPTIMIZATION_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_ARMIJOCONDITION_H_
#define MYOPTIMIZATION_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_ARMIJOCONDITION_H_
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <stdexcept>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/SteepestGradientDescent/utils/utils.h"

namespace MyOptimization {
namespace ConvexOptimization {
void ArmijoParameterCheck(const float parameter);

template <typename Func, typename T, size_t N>
void LinearSearchWithArmijoCondition(
    const Func &func, const std::array<T, N> &current_step_input_value,
    const std::array<T, N> &negative_gradients, const float armijo_parameter,
    float &step);

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // MYOPTIMIZATION_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_ARMIJOCONDITION_H_
