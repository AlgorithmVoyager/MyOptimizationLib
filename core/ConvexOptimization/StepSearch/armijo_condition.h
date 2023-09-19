#ifndef CORE_CONVEXOPTIMIZATION_STEPSEARCH_ARMIJO_CONDITION_H
#define CORE_CONVEXOPTIMIZATION_STEPSEARCH_ARMIJO_CONDITION_H

#include <algorithm>
#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <stdexcept>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/utils/utils.h"
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

/// @brief : the function is used in practical newthon method, its mainly used
/// situation f(x) = g(x) +h(x), g(x) is convex & differental, & h(x) is convex
/// and can be not contonously.
/// its iteration form like this:
///          v_+ = argmin_z D g(x)^T * v  + 1/2*v^T * H^(k-1)*v + h(x^(k-1)+v)
/// in out situation, we dont handle h(x), so upper iteration step can be
/// wrriten as:
///          v_+ = argmin_z D g(x)^T * v  + 1/2*t * H^(k-1)*v
/// Dg is gradients, H is hessian matrix
/// @tparam Func
/// @tparam T
/// @tparam N
/// @param v
/// @param
template <typename Func, typename T, size_t N>
void UseSecondOrderLossFunctionGetClosetVector(
    const Eigen::Matrix<T, N, 1> &gradients,
    const Eigen::Matrix<T, N, N> &hessian, Eigen::Matrix<T, N, 1> &v,
    const double learing_rate = 0.6, const double tolerance = 1e-3) {
  Eigen::Matrix<T, N, 1> last_iterate_v;

  auto has_get_minimum_v = [&]() {
    return v.isApprox(last_iterate_v, tolerance);
  };
  bool continue_approximate = true;

  const int maximum_iteration = 100000;
  int iter_step = 0;
  // MLOG_ERROR("=========================================");
  // while (continue_approximate) {
  //   last_iterate_v.noalias() = v;
  //   Eigen::Matrix<T, N, 1> step;
  //   MLOG_ERROR("======== last last_iterate_v " << last_iterate_v);
  //   MLOG_ERROR("======== hessian * v " << hessian * v);
  //   MLOG_ERROR("========gradients " << gradients);
  //   step.noalias() = gradients + hessian * v;
  //   MLOG_ERROR("======== step " << step);
  //   v.noalias() = last_iterate_v - learing_rate * step;
  //   MLOG_ERROR("======== v " << v);
  //   continue_approximate = !has_get_minimum_v();
  //   MLOG_ERROR("======== continue_approximate " << continue_approximate);

  //   /// avoid entry into dead loop
  //   iter_step++;
  //   if (iter_step > maximum_iteration) {
  //     return;
  //   }
  // }
  // !!! so fucking interesting
  v = -1.0 * hessian.inverse() * gradients;
  return;
}

/// @brief: this function is mainly update step for practical NetwonMethod,
/// according vector get by UseSecondOrderLossFunctionGetClosetVector, next step
/// and current step value by function f will be calculated. then we can compare
/// next step value with sum of the current step value and linear part
/// @tparam Func
/// @tparam T
/// @tparam N
/// @param armijo_parameter
/// @param gradients
/// @param hessian
/// @param v
/// @param init_x
/// @param func
/// @param step
/// @param scaler_coefficient
/// @param tolerance
template <typename Func, typename T, size_t N>
void SecondOrderSearchWithArmijoCondition(
    const float armijo_parameter, const Eigen::Matrix<T, N, 1> &gradients,
    const Eigen::Matrix<T, N, N> &hessian, const Eigen::Matrix<T, N, 1> &v,
    const Eigen::Matrix<T, N, 1> &init_x, Func func, float &step,
    const double scaler_coefficient = 0.5, const double tolerance = 1e-3) {
  // calculat functoin value in current step: f(x)
  const T function_value_in_current_step =
      MyOptimization::BaseMath::GetFuncValueForEigenVector<Func, T, N>(func,
                                                                       init_x);
  // MLOG_ERROR("===inti step " << step);
  auto linear_part_sum = [&]() -> T {
    // here not consider h(x), so no need calculat h part
    return armijo_parameter * step * gradients.dot(v);
  };

  auto next_step_result = [&]() -> T {
    Eigen::Matrix<T, N, 1> next_step;
    next_step.noalias() = init_x + step * v;
    return MyOptimization::BaseMath::GetFuncValueForEigenVector<Func, T, N>(
        func, next_step);
  };

  // c * t * d^T * Df(x^k), d = - Df(x^k)
  // MLOG_ERROR("next_step_result() "
  //            << next_step_result() << ",\n function_value_in_current_step "
  //            << function_value_in_current_step << ", linear part sum \n"
  //            << linear_part_sum());
  while (next_step_result() >
         function_value_in_current_step + linear_part_sum()) {
    step *= scaler_coefficient;
  }

  return;
}

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_STEPSEARCH_ARMIJO_CONDITION_H
