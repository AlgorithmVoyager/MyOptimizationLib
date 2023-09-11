#include "core/ConvexOptimization/SteepestGradientDescent/steepest_gradient_descent.h"

#include <algorithm>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/SteepestGradientDescent/armijo_condition.h"
#include "core/ConvexOptimization/SteepestGradientDescent/utils/utils.h"

using namespace MyOptimization::BaseMath;

namespace MyOptimization {
namespace ConvexOptimization {

template <typename Function, typename GradientFunction, typename T, size_t N>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const float iteration_error,
                            const Function &value_function,
                            const std::array<T, N> &init_pos)
    : step_(step),
      const_parameter_for_amijro_condition_(parameter),
      iteration_error_(iteration_error),
      current_searched_results_(init_pos) {
  MyOptimization::ConvexOptimization::ArmijoParameterCheck(parameter);
  MyOptimization::ConvexOptimization::utils::ValueFunctionCheck(value_function);
  value_function_ = value_function;
  gradient_function_ = std::nullopt;
}

template <typename Function, typename GradientFunction, typename T, size_t N>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const float iteration_error,
                            const Function &value_function,
                            const GradientFunction &gradient_fucntion,
                            const std::array<T, N> &init_pos)
    : step_(step),
      const_parameter_for_amijro_condition_(parameter),
      current_searched_results_(init_pos) {
  MyOptimization::ConvexOptimization::ArmijoParameterCheck(parameter);
  MyOptimization::ConvexOptimization::utils::ValueAndGradientFunctionCheck(
      value_function, gradient_fucntion);
  value_function_ = value_function;
  gradient_function_ = gradient_fucntion;
}

template <typename Function, typename GradientFunction, typename T, size_t N>
template <typename Container>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const float iteration_error,
                            const Function &value_function,
                            const Container &init_pos)
    : step_(step),
      const_parameter_for_amijro_condition_(parameter),
      iteration_error_(iteration_error) {
  MyOptimization::ConvexOptimization::ArmijoParameterCheck(parameter);
  MyOptimization::ConvexOptimization::utils::ContainerSizeCheck(init_pos);
  std::copy(init_pos.begin(), init_pos.end(),
            current_searched_results_.begin());

  MyOptimization::ConvexOptimization::utils::ValueFunctionCheck(value_function);
  value_function_ = value_function;
  gradient_function_ = std::nullopt;
}

template <typename Function, typename GradientFunction, typename T, size_t N>
template <typename Container>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const float iteration_error,
                            const Function &value_function,
                            const GradientFunction &gradient_fucntion,
                            const Container &init_pos)
    : step_(step),
      const_parameter_for_amijro_condition_(parameter),
      iteration_error_(iteration_error) {
  MyOptimization::ConvexOptimization::ArmijoParameterCheck(parameter);
  MyOptimization::ConvexOptimization::utils::ContainerSizeCheck(init_pos);
  std::copy(init_pos.begin(), init_pos.end(),
            current_searched_results_.begin());

  MyOptimization::ConvexOptimization::utils::ValueAndGradientFunctionCheck(
      value_function, gradient_fucntion);
  value_function_ = value_function;
  gradient_function_ = gradient_fucntion;
}

template <typename Function, typename GradientFunction, typename T, size_t N>
template <typename... Args>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const float iteration_error,
                            const Function &value_function, const Args &...args)
    : step_(step), iteration_error_(iteration_error) {
  MyOptimization::ConvexOptimization::ArmijoParameterCheck(parameter);
  MyOptimization::ConvexOptimization::utils::ValueFunctionCheck(value_function);
  value_function_ = value_function;
  gradient_function_ = std::nullopt;

  utils::ArgsCheck(value_function, args...);
  current_searched_results_ = {args...};
}

template <typename Function, typename GradientFunction, typename T, size_t N>
template <typename... Args>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const float iteration_error,
                            const Function &value_function,
                            const GradientFunction &gradient_fucntion,
                            const Args &...args)
    : step_(step), iteration_error_(iteration_error) {
  MyOptimization::ConvexOptimization::ArmijoParameterCheck(parameter);
  MyOptimization::ConvexOptimization::utils::ValueFunctionCheck(value_function);
  value_function_ = value_function;
  gradient_function_ = gradient_fucntion;

  MyOptimization::ConvexOptimization::utils::ArgsCheck(value_function, args...);
  current_searched_results_ = {args...};
}

template <typename Function, typename GradientFunction, typename T, size_t N>
void SteepestGradientDescent<Function, GradientFunction, T, N>::Step() {
  std::array<T, N> last_iterate = current_searched_results_;
  current_searched_results_.fill(std::numeric_limits<T>::max());
  while (true) {
    ChooseSerachDirection();
    LinearSearchWithArmijoCondition(
        value_function_, current_searched_results_, search_direction_,
        const_parameter_for_amijro_condition_, step_);
    last_iterate = current_searched_results_;
    UpdateIterate();
  }
}

template <typename Function, typename GradientFunction, typename T, size_t N>
void SteepestGradientDescent<Function, GradientFunction, T,
                             N>::ChooseSerachDirection() {
  if (gradient_function_.has_value()) {
    search_direction_ = GetFuncValueForArray(gradient_function_.value(),
                                             current_searched_results_);
    MyOptimization::ConvexOptimization::utils::GetArrayMultiLambda(
        search_direction_, -1.0);
  } else {
    const float epsilon_step = 1e-3;

    search_direction_ = MyOptimization::BaseMath::
        GetNumericGrandientForArrayByForwardDifference(
            value_function_, current_searched_results_, epsilon_step);
    MyOptimization::ConvexOptimization::utils::GetArrayMultiLambda(
        search_direction_, -1.0);
  }

  return;
}

template <typename Function, typename GradientFunction, typename T, size_t N>
void SteepestGradientDescent<Function, GradientFunction, T,
                             N>::UpdateIterate() {
  // x_k+1 = x_k + step * descent_gradient
  std::array<T, N> iterator_increment =
      MyOptimization::ConvexOptimization::utils::GetArrayMultiLambda(
          search_direction_, step_);
  MyOptimization::ConvexOptimization::utils::AddArray(
      iterator_increment, current_searched_results_);
  return;
}

}  // namespace ConvexOptimization
}  // namespace MyOptimization
