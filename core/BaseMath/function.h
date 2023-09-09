#ifndef MYOPTIMIZATION_BASEMATH_FUNCTION_H_
#define MYOPTIMIZATION_BASEMATH_FUNCTION_H_

#include <functional>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/BaseMath/container.h"

namespace MyOptimization {
namespace BaseMath {

// /// @param: func: user self defined function
// /// @param: value: the input value of function
// /// @brief: the fuinction will return final value calculate by func
template <typename Callable, typename T>
T GetValue(Callable func, T&& t) {
  static_assert(std::is_function<Callable>::value,
                "input Func must be a function!!!");

  return func(std::forward<T>(t));
}

// /// @param: func: user self defined function
// /// @param: value: the inputs value of function
// /// @brief: the fuinction will return final value calculate by func
template <typename Callable, typename... T>
auto GetFuncValueForMultiInput(Callable func, T&&... t) {
  static_assert(std::is_invocable<Callable, T...>::value,
                "Input Func must be callable with input arguments");
  return func(std::forward<T>(t)...);
}

// /*use value, not ref, to avoid left or right value error*/
// /// @param: func: user self defined function
// /// @param: container: container e.g std::vector and so on,
// /// @brief: the fuinction will return final value calculate by func
template <typename Callable, typename Container, size_t N>
auto GetFuncValueForContainer(Callable func, Container container) {
  return std::apply(func, CreateTupleFromContainer<Container, N>(container));
}

// /// @param: func: user self defined function
// /// @param: container: std::array
// /// @brief: the fuinction will return final value calculate by func
template <typename Callable, typename T, size_t N>
auto GetFuncValueForArray(Callable func, std::array<T, N> container) {
  return std::apply(func, CreateTupleFromArray(container));
}

template <typename Callable, typename Container, typename T, size_t N>
Container GetNumericnGrandientForContainerByForwardDifference(
    Callable func, Container container, double epsilon) {
  Container gradients;

  if (std::begin(container) == std::end(container)) {
    return gradients;
  }

  // static_assert(std::is_same_v<decltype(*(std::begin(container))), T>,
  //               "Container's each element type should be same with T's
  //               type");
  T original_value =
      GetFuncValueForContainer<Callable, Container, N>(func, container);

  for (size_t i = 0U; i < N; ++i) {
    Container inputs = container;
    inputs[i] += epsilon;

    T perturbed_value =
        GetFuncValueForContainer<Callable, Container, N>(func, inputs);
    std::cout << "perturbed_value " << perturbed_value << "original_value "
              << original_value << "\n";

    gradients.emplace_back((perturbed_value - original_value) / epsilon);
  }

  return gradients;
}

template <typename Callable, typename T, size_t N>
std::array<T, N> GetNumericGrandientForArrayByForwardDifference(
    Callable func, std::array<T, N> array, double epsilon) {
  std::array<T, N> gradients;

  if (array.empty()) {
    return gradients;
  }

  T original_value = GetFuncValueForArray<Callable, T, N>(func, array);

  for (size_t i = 0U; i < N; ++i) {
    std::array<T, N> inputs = array;
    inputs[i] += epsilon;

    T perturbed_value = GetFuncValueForArray<Callable, T, N>(func, inputs);
    gradients[i] = (perturbed_value - original_value) / epsilon;
  }

  return gradients;
}

template <typename Callable, typename Container, typename T, size_t N>
Container GetNumericnGrandientForContainerByCenteralDifference(
    Callable func, Container container, double epsilon) {
  Container gradients;

  if (std::begin(container) == std::end(container)) {
    return gradients;
  }

  // static_assert(std::is_same_v<decltype(*(std::begin(container))), T>,
  //               "Container's each element type should be same with T's
  //               type");

  for (size_t i = 0U; i < N; ++i) {
    Container last_half_step_inputs = container;
    Container next_half_step_inputs = container;
    last_half_step_inputs[i] -= epsilon / 2;
    next_half_step_inputs[i] += epsilon / 2;

    T next_half_step_value = GetFuncValueForContainer<Callable, Container, N>(
        func, next_half_step_inputs);
    T last_half_step_value = GetFuncValueForContainer<Callable, Container, N>(
        func, last_half_step_inputs);
    std::cout << "next_half_step_value " << next_half_step_value
              << "last_half_step_value " << last_half_step_value << "\n";

    gradients.emplace_back((next_half_step_value - last_half_step_value) /
                           epsilon);
  }

  return gradients;
}

template <typename Callable, typename T, size_t N>
std::array<T, N> GetNumericGrandientForArrayByCenteralDifference(
    Callable func, std::array<T, N> array, double epsilon) {
  std::array<T, N> gradients;

  if (array.empty()) {
    return gradients;
  }

  for (size_t i = 0U; i < N; ++i) {
    std::array<T, N> last_half_step_inputs = array;
    std::array<T, N> next_half_step_inputs = array;
    last_half_step_inputs[i] -= epsilon / 2;
    next_half_step_inputs[i] += epsilon / 2;

    T next_half_step_value =
        GetFuncValueForArray<Callable, T, N>(func, next_half_step_inputs);
    T last_half_step_value =
        GetFuncValueForArray<Callable, T, N>(func, last_half_step_inputs);
    gradients[i] = (next_half_step_value - last_half_step_value) / epsilon;
  }

  return gradients;
}

}  // namespace BaseMath

}  // namespace MyOptimization

#endif  // MYOPTIMIZATION_BASEMATH_FUNCTION_H_