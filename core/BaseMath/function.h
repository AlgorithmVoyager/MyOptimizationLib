#ifndef MYOPTIMIZATION_BASEMATH_FUNCTION_H_
#define MYOPTIMIZATION_BASEMATH_FUNCTION_H_

#include <functional>
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
  return std::apply(
      func, CreateTupleFromArray(std::forward<std::array<T, N>>(container)));
}

}  // namespace BaseMath

}  // namespace MyOptimization

#endif  // MYOPTIMIZATION_BASEMATH_FUNCTION_H_