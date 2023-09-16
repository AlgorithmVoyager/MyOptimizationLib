#ifndef CORE_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_STEEPEST_GRADIENT_DESCENT_H
#define CORE_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_STEEPEST_GRADIENT_DESCENT_H

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/SteepestGradientDescent/armijo_condition.h"
#include "core/ConvexOptimization/SteepestGradientDescent/utils/utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <optional>

namespace MyOptimization {
namespace ConvexOptimization {

template <typename T, size_t N> class SteepestGradientDescent {
public:
  SteepestGradientDescent(){};
  ~SteepestGradientDescent(){};

  /// @brief function for input as array and use numeric method to calculate
  /// gradients
  /// @param step
  /// @param parameter
  /// @param
  SteepestGradientDescent(const float step, const float parameter,
                          const float iteration_error,
                          std::array<T, N> init_pos)
      : step_(step), const_parameter_for_amijro_condition_(parameter),
        iteration_error_(iteration_error), current_searched_results_(init_pos),
        init_step_(step) {
    ArmijoParameterCheck(parameter);
  };
  // SteepestGradientDescent(const float step, const float parameter,
  //                         const float iteration_error, Function
  //                         value_function, std::array<T, N> init_pos);

  /// @brief function for input as container and use numeric method to calculate
  /// gradients
  /// @param step
  /// @param parameter
  /// @param
  // template <typename Container>
  // SteepestGradientDescent(const float step, const float parameter,
  //                         const float iteration_error, Function
  //                         value_function, Container init_pos);

  /// @brief function for input as series variales and use numeric method to
  /// calculate gradients
  /// @tparam ...Args
  /// @param step
  /// @param parameter
  /// @param function
  /// @param ...args
  // template <typename... Args>
  // SteepestGradientDescent(const float step, const float parameter,
  //                         const float iteration_error, Function
  //                         value_function, Args... args);

  /// @brief : main function loop
  template <typename Function> void Step(Function value_function) {
    std::array<T, N> last_iterate = current_searched_results_;

    bool continue_gradient_descent = true;
    while (continue_gradient_descent) {
      step_ = init_step_;
      ChooseSerachDirection<T, Function>(static_cast<T>(-1.0), value_function);
      LinearSearchWithArmijoCondition(const_parameter_for_amijro_condition_,
                                      step_, current_searched_results_,
                                      search_direction_, value_function);
      last_iterate = current_searched_results_;
      UpdateIterate();
      continue_gradient_descent =
          StatisfyGradientDescentFinishCondition(last_iterate, value_function);
    }
    return;
  }

  template <typename Function, typename... GradientFunction>
  void Step(Function value_function, GradientFunction... gradients_functions) {
    std::array<T, N> last_iterate = current_searched_results_;

    bool continue_gradient_descent = true;
    while (continue_gradient_descent) {
      // step_ = init_step_;
      ChooseSerachDirection(static_cast<T>(-1.0), value_function,
                            gradients_functions...);
      LinearSearchWithArmijoCondition(const_parameter_for_amijro_condition_,
                                      step_, current_searched_results_,
                                      search_direction_, value_function);
      last_iterate = current_searched_results_;
      UpdateIterate();
      continue_gradient_descent =
          StatisfyGradientDescentFinishCondition(last_iterate, value_function);
    }
    return;
  }

  const std::array<T, N> GetSearchDirection() const {
    return search_direction_;
  }

  const std::array<T, N> GetSearchResult() const {
    return current_searched_results_;
  }

private:
  /// @brief : choose griandent descent direction
  template <typename Lambda, typename Function>
  void ChooseSerachDirection(Lambda lambda, Function value_function) {

    const T epsilon_step = 1e-3;

    search_direction_ = MyOptimization::BaseMath::
        GetNumericGrandientForArrayByForwardDifference(
            value_function, current_searched_results_, epsilon_step);
    MyOptimization::ConvexOptimization::utils::GetArraySelfMultiLambda<Lambda>(
        lambda, search_direction_);

    return;
  }

  template <typename Lambda, typename Function, typename... GradientFunction>
  void ChooseSerachDirection(Lambda lambda, Function value_function,
                             GradientFunction... gradients_functions) {

    search_direction_ = {MyOptimization::BaseMath::GetFuncValueForArray(
        gradients_functions, current_searched_results_)...};

    MyOptimization::ConvexOptimization::utils::GetArraySelfMultiLambda<Lambda>(
        lambda, search_direction_);
    return;
  }

  /// @brief : check if current loop, if it has satisify armijo linear search
  /// condition
  /// @return
  template <typename Function>
  bool
  StatisfyGradientDescentFinishCondition(const std::array<T, N> &last_iterator,
                                         Function value_function) {
    auto last_iterate_func_value =
        MyOptimization::BaseMath::GetFuncValueForArray(value_function,
                                                       last_iterator);
    auto current_iterate_func_value =
        MyOptimization::BaseMath::GetFuncValueForArray(
            value_function, current_searched_results_);

    // std::array<T, N> iteration_diff;
    // std::transform(
    //     current_searched_results_.begin(), current_searched_results_.end(),
    //     last_iterator.begin(), iteration_diff.begin(),
    //     [](const auto &lhs, const auto &rhs) { return std::fabs(lhs - rhs);
    //     });

    // bool iteration_value_near_real_value = std::all_of(
    //     iteration_diff.begin(), iteration_diff.end(),
    //     [this](auto &iter_diff) { return iter_diff <= iteration_error_; });

    MLOG_ERROR("last_iterator "
               << last_iterator[0] << ", current_searched_results_ "
               << current_searched_results_[0] << "last_iterate_func_value "
               << last_iterate_func_value << ",current_iterate_func_value "
               << current_iterate_func_value);
    return (std::fabs(current_iterate_func_value - last_iterate_func_value) >=
            iteration_error_);
    // &&  iteration_value_near_real_value;
  }

  /// @brief : update iterator value
  void UpdateIterate() {
    // x_k+1 = x_k + step * descent_gradient
    std::array<T, N> iterator_increment =
        MyOptimization::ConvexOptimization::utils::GetArrayMultiLambda<T, float,
                                                                       N>(
            step_, search_direction_);
    MyOptimization::ConvexOptimization::utils::SelfAddArray<T, N>(
        iterator_increment, current_searched_results_);
    return;
  }

private:
  float step_;
  float const_parameter_for_amijro_condition_ = 0.5;
  float iteration_error_ = 10.0;
  const float init_step_;

  std::array<T, N> search_direction_;
  std::array<T, N> current_searched_results_;
};

} // namespace ConvexOptimization

} // namespace MyOptimization

#endif // CORE_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_STEEPEST_GRADIENT_DESCENT_H
