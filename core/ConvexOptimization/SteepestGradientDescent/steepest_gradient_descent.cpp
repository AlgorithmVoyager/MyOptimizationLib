#include "core/ConvexOptimization/SteepestGradientDescent/steepest_gradient_descent.h"

#include <glog/logging.h>

#include <algorithm>
#include <stdexcept>

namespace MyOptimization {
namespace ConvexOptimization {

namespace {
template <typename Function>
void ValueFunctionCheck(const Function& value_function) {
  if (!std::is_function_v<value_function>) {
    try {
      throw std::runtime_error(
          "Input Value Function is not a functional type!!! "
          "Construct SteepestGradientDescent Failed");
    } catch (const std::exception& e) {
      LOG(ERROR) << "[Runtime Error]: " << e.what();
    }
  }
  return;
}

template <typename Function, typename GradientFunction>
void ValueAndGradientFunctionCheck(const Function& value_function,
                                   const GradientFunction& gradient_fucntion) {
  if (!std::is_function_v<value_function> &&
      std::is_function_v<gradient_fucntion>) {
    try {
      throw std::runtime_error(
          "Input Value Or Gradient Function is not a functional type!!! "
          "Construct SteepestGradientDescent Failed");
    } catch (const std::exception& e) {
      LOG(ERROR) << "[Runtime Error]: " << e.what();
    }
  }
  return;
}

template <typename Container, size_t N>
void ContainerSizeCheck(const Container& container) {
  if (container.size() != N) {
    try {
      throw std::runtime_error(
          "Input Container Size must same with N, Construct "
          "SteepestGradientDescent Failed!!!");
    } catch (const std::exception& e) {
      LOG(ERROR) << "[Runtime Error]: " << e.what();
    }
  }
  return;
}

template <typename Function, typename... Args, size_t N>
void ArgsCheck(const Function& value_function, const Args&... args) {
  if (N != sizeof...(args) || std::is_invocable<Function, Args...>::value) {
    try {
      throw std::runtime_error(
          "Input Function And Args not match, Can't run the function,"
          "Construct SteepestGradientDescent Failed!!!");
    } catch (const std::exception& e) {
      LOG(ERROR) << "[Runtime Error]: " << e.what();
    }
  }
  return;
}

}  // namespace

template <typename Function, typename GradientFunction, typename T, size_t N>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const Function& value_function,
                            const std::array<T, N>& init_pos)
    : step_(step),
      const_parameter_for_amijro_condition_(parameter),
      final_search_results_(init_pos) {
  ValueFunctionCheck(value_function);
  value_function_ = value_function;
  gradient_function_ = std::nullopt;
}

template <typename Function, typename GradientFunction, typename T, size_t N>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const Function& value_function,
                            const GradientFunction& gradient_fucntion,
                            const std::array<T, N>& init_pos)
    : step_(step),
      const_parameter_for_amijro_condition_(parameter),
      final_search_results_(init_pos) {
  ValueAndGradientFunctionCheck(value_function, gradient_fucntion);
  value_function_ = value_function;
  gradient_function_ = gradient_fucntion;
}

template <typename Function, typename GradientFunction, typename T, size_t N>
template <typename Container>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const Function& value_function,
                            const Container& init_pos)
    : step_(step), const_parameter_for_amijro_condition_(parameter) {
  ContainerSizeCheck(init_pos);
  std::copy(init_pos.begin(), init_pos.end(), final_search_results_.begin());

  ValueFunctionCheck(value_function);
  value_function_ = value_function;
  gradient_function_ = std::nullopt;
}

template <typename Function, typename GradientFunction, typename T, size_t N>
template <typename Container>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const Function& value_function,
                            const GradientFunction& gradient_fucntion,
                            const Container& init_pos)
    : step_(step), const_parameter_for_amijro_condition_(parameter) {
  ContainerSizeCheck(init_pos);
  std::copy(init_pos.begin(), init_pos.end(), final_search_results_.begin());

  ValueAndGradientFunctionCheck(value_function, gradient_fucntion);
  value_function_ = value_function;
  gradient_function_ = gradient_fucntion;
}

template <typename Function, typename GradientFunction, typename T, size_t N>
template <typename... Args>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const Function& value_function, const Args&... args)
    : step_(step) {
  ValueFunctionCheck(value_function);
  value_function_ = value_function;
  gradient_function_ = std::nullopt;

  ArgsCheck(value_function, args...);
  final_search_results_ = {args...};
}

template <typename Function, typename GradientFunction, typename T, size_t N>
template <typename... Args>
SteepestGradientDescent<Function, GradientFunction, T, N>::
    SteepestGradientDescent(const float step, const float parameter,
                            const Function& value_function,
                            const GradientFunction& gradient_fucntion,
                            const Args&... args)
    : step_(step) {
  ValueFunctionCheck(value_function);
  value_function_ = value_function;
  gradient_function_ = gradient_fucntion;

  ArgsCheck(value_function, args...);
  final_search_results_ = {args...};
}
}  // namespace ConvexOptimization
}  // namespace MyOptimization
