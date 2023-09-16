#ifndef CORE_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_ARMIJO_CONDITION_H
#define CORE_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_ARMIJO_CONDITION_H

#include <algorithm>
#include <array>
#include <stdexcept>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/SteepestGradientDescent/utils/utils.h"
#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {
void ArmijoParameterCheck(const float parameter) {
  if (parameter <= 0.0 || parameter >= 1.0) {
    try {
      throw std::runtime_error("the param must be between (0,1)");
    } catch (const std::exception &e) {
      MLOG_ERROR("[Runtime Error]:" << e.what());
    }
  }
  return;
}

template <typename Func, typename T, size_t N>
void LinearSearchWithArmijoCondition(const float armijo_parameter, float &step,
                                     std::array<T, N> current_step_input_value,
                                     std::array<T, N> negative_gradients,
                                     Func func) {
  // static_assert(std::is_function_v<Func>, "Func must be a function");

  const float stop_step = 1e-8;

  const T function_value_in_init_step =
      MyOptimization::BaseMath::GetFuncValueForArray(func,
                                                     current_step_input_value);
  const T product_of_negative_gradients_with_gradients =
      -1.0 * MyOptimization::ConvexOptimization::utils::GetDotProductOfTwoArray(
                 negative_gradients, negative_gradients);

  // c * t * d^T * Df(x^k), d = - Df(x^k)
  while (MyOptimization::BaseMath::GetFuncValueForArray(
             func,
             MyOptimization::ConvexOptimization::utils::GetNextStepInputValue(
                 current_step_input_value, negative_gradients, step)) >
         (function_value_in_init_step +
          (armijo_parameter * step *
           product_of_negative_gradients_with_gradients))) {

    /// @brief: debug info
    MLOG_INFO(
        "f(x+td)  "
        << MyOptimization::BaseMath::GetFuncValueForArray(
               func,
               MyOptimization::ConvexOptimization::utils::GetNextStepInputValue(
                   current_step_input_value, negative_gradients, step))
        << ",\n f(x) " << function_value_in_init_step << "\n c_p "
        << (armijo_parameter * step *
            product_of_negative_gradients_with_gradients));

    step /= 2;
    if (step < stop_step) {
      MLOG_ERROR("step has minize als "
                 << stop_step << ", but linear search still not finished !!!");
      break;
    }
  }
  MLOG_INFO(
      "f(x+td)  "
      << MyOptimization::BaseMath::GetFuncValueForArray(
             func,
             MyOptimization::ConvexOptimization::utils::GetNextStepInputValue(
                 current_step_input_value, negative_gradients, step))
      << ",\n f(x) " << function_value_in_init_step << "\n c_p "
      << (armijo_parameter * step *
          product_of_negative_gradients_with_gradients));

  return;
}

} // namespace ConvexOptimization
} // namespace MyOptimization

#endif // CORE_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_ARMIJO_CONDITION_H
