#ifndef CORE_CONVEXOPTIMIZATION_STEPSEARCH_LEWIS_OVERTON_CONDITION_H
#define CORE_CONVEXOPTIMIZATION_STEPSEARCH_LEWIS_OVERTON_CONDITION_H

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <numeric>
#include <stdexcept>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/utils/utils.h"
#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {

template <typename T>
void LevisAndOvertonConditionParameterCheck(T c1, T c2) {
  if ((static_cast<T>(0.0) < c1) && (c1 < c2) && (c2 < static_cast<T>(1.0))) {
    return;
  }

  try {
    throw std::runtime_error(
        "Levis & overton condition parameter c1 and c2 should be in range "
        "(0,1), and c1 "
        "should be smaller than c2!!!");
  } catch (const std::exception& e) {
    MLOG_ERROR(e.what());
  }
  return;
}

/// @brief : for levis and overton linear search condition, it contains two
/// parts:
/*
 *  S(a): armijo part (sufficient decrease condition):
 *     f(x_k) - f(x_k + alpha * d) >= - c1 * alpha * d^T * gradients(f(x_k))
 *  C(a): curvature condition:
 *     d^T * f(x_k + alpha * d) >= c2 *  d^T * gradients(f(x_k))
 *   : @param : alpha -> step
 *   : @param : c1,c2 -> wolfe condition parameter
 *   : @param : d -> search_direction
 *   : @param : gradients(f(x_k)) -> gradients at x_k of function f
 * */
/// @tparam T
/// @param gradients
/// @param search_direction
/// @param c1
/// @param c2
/// @param step
template <typename Callable, typename T, size_t N>
void LinearSearchForEigenVectorWithLevisAndOvertonCondition(
    const Eigen::Matrix<T, N, 1>& init_pos,
    const Eigen::Matrix<T, N, 1>& search_direction, T& step, Callable func,
    T decrease_coeff = 0.5, T c1 = 1e-4, T c2 = 0.9, T epsilon = 1e-3) {
  const T stop_step = 1e-6;

  const T func_value_at_current_step =
      MyOptimization::BaseMath::GetFuncValueForEigenVector<Callable, T, N>(
          func, init_pos);

  Eigen::Matrix<T, N, 1> gradients_at_current_step;
  gradients_at_current_step.noalias() = MyOptimization::BaseMath::
      GetNumericGrandientForEigenVectorByForwardDifference<Callable, T, N>(
          func, init_pos, epsilon);
  const T dot_product_of_search_direction_with_current_step_gradients =
      search_direction.dot(gradients_at_current_step);

  Eigen::Matrix<T, N, 1> next_step, gradients_at_next_step;
  T func_value_at_next_step;

  next_step.noalias() = init_pos + step * search_direction;
  gradients_at_next_step.noalias() = MyOptimization::BaseMath::
      GetNumericGrandientForEigenVectorByForwardDifference<Callable, T, N>(
          func, next_step, epsilon);
  func_value_at_next_step =
      MyOptimization::BaseMath::GetFuncValueForEigenVector<Callable, T, N>(
          func, next_step);

  // * define weak wolfe condition
  auto armijo_condition = [&]() {
    return (func_value_at_current_step - func_value_at_next_step) >=
           -1.0 * c1 * step *
               dot_product_of_search_direction_with_current_step_gradients;
  };

  auto curvature_condition = [&]() {
    return (search_direction.dot(gradients_at_next_step)) >=
           (c2 * dot_product_of_search_direction_with_current_step_gradients);
  };

  int search_step = 0;
  const T max_upper_bound = std::numeric_limits<T>::max();
  T upper_bound = max_upper_bound;
  T lower_bound = 0.0;

  bool continue_search = true;
  while (continue_search) {
    // ? [debug info]
    // MLOG_ERROR(
    //     "func_value_at_current_step = "
    //     << func_value_at_current_step << ", func_value_at_next_step "
    //     << func_value_at_next_step
    //     << "(func_value_at_current_step - func_value_at_next_step) = "
    //     << (func_value_at_current_step - func_value_at_next_step)
    //     << " -1.0 * step * c1  "
    //        "dot_product_of_search_direction_with_current_step_gradients = "
    //     << -1.0 * c1 * step *
    //            dot_product_of_search_direction_with_current_step_gradients);
    // MLOG_ERROR("================");
    // MLOG_ERROR(
    //     "search_direction.dot(gradients_at_next_step) = "
    //     << search_direction.dot(gradients_at_next_step)
    //     << ", (c2 * "
    //        "dot_product_of_search_direction_with_current_step_gradients) = "
    //     << (c2 *
    //     dot_product_of_search_direction_with_current_step_gradients));

    // * update step , lower bound & upper bound
    if (!armijo_condition()) {
      upper_bound = step;
    } else if (!curvature_condition()) {
      lower_bound = step;
    } else {
      return;
    }

    if (upper_bound < max_upper_bound) {
      step = lower_bound + (upper_bound - lower_bound) / 2;
    } else {
      step = 2 * lower_bound;
    }
    // MLOG_ERROR("upper_bound " << upper_bound << ", lower_bound " <<
    // lower_bound
    //                           << ", step " << step);

    // * update next search gradients and function value
    step *= decrease_coeff;
    next_step.noalias() = init_pos + step * search_direction;
    gradients_at_next_step.noalias() = MyOptimization::BaseMath::
        GetNumericGrandientForEigenVectorByForwardDifference<Callable, T, N>(
            func, next_step, epsilon);
    func_value_at_next_step =
        MyOptimization::BaseMath::GetFuncValueForEigenVector<Callable, T, N>(
            func, next_step);

    // * check if need continue search
    continue_search = (!armijo_condition()) || (!curvature_condition());

    // * if search step has exceed max search step or step has minial als stop
    // * step, stop search
    search_step++;
    if (search_step > 100000) {
      MLOG_ERROR(
          "levis & overton condition has searched 100000 times, but still not "
          "get "
          "the best step");
      return;
    }

    if (step < stop_step) {
      MLOG_ERROR("step has minize als "
                 << stop_step << ", but linear search still not finished !!!");
      break;
    }
  }
  return;
}

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_STEPSEARCH_LEWIS_OVERTON_CONDITION_H
