#include "core/BaseMath/function.h"

namespace MyOptimization {
namespace BaseMath {

// template <typename Callable, typename T>
// T GetValue(Callable func, T&& t) {
//   static_assert(std::is_function<Callable>::value,
//                 "input Func must be a function!!!");

//   return func(std::forward<T>(t));
// }

// template <typename Callable, typename... T>
// auto GetFuncValueForMultiInput(Callable func, T&&... t) {
//   static_assert(std::is_invocable<Callable, T...>::value,
//                 "Input Func must be callable with input arguments");
//   return func(std::forward<T>(t)...);
// }

// template <typename Callable, typename Container, size_t N>
// auto GetFuncValueForContainer(Callable func, Container container) {
//   return std::apply(func, CreateTupleFromContainer<Container, N>(container));
// }

// template <typename Callable, typename T, size_t N>
// auto GetFuncValueForArray(Callable func, std::array<T, N> container) {
//   return std::apply(
//       func, CreateTupleFromArray(std::forward<std::array<T, N>>(container)));
// }

}  // namespace BaseMath

}  // namespace MyOptimization
