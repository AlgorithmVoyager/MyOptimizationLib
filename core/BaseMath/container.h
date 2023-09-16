#ifndef CORE_BASEMATH_CONTAINER_H
#define CORE_BASEMATH_CONTAINER_H

#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace MyOptimization {
namespace BaseMath {

/// @brief : convert a container into tuple impl
/// @tparam Container
/// @tparam ...Index
/// @param container
/// @param
/// @return
template <typename Container, std::size_t... Index>
auto CreateTupleFromContainerImpl(Container container,
                                  std::index_sequence<Index...>) {
  auto begin = std::begin(container);
  auto end = std::end(container);
  if (begin == end) {
    throw std::runtime_error(
        "Input Variable Container must be not empty,but intput variable "
        "container is empty !!!");
  }
  return std::make_tuple(container[Index]...);
}

/// @brief  convert a container into tuple
/// @tparam Container
/// @tparam N
/// @param container
/// @return
template <typename Container, std::size_t N>
auto CreateTupleFromContainer(Container container) {
  return CreateTupleFromContainerImpl(container, std::make_index_sequence<N>{});
}

// template <typename Container, std::size_t N>
// auto CreateTupleFromContainer(Container &container) {
//   return CreateTupleFromContainerImpl(container,
//   std::make_index_sequence<N>{});
// }

template <typename Array, std::size_t... Index>
auto CreateTupleFromArrayImpl(Array container, std::index_sequence<Index...>) {
  if (container.empty()) {
    throw std::runtime_error(
        "Input Variable Container must be not empty,but intput variable "
        "container is empty !!!");
  }
  return std::make_tuple(container[Index]...);
}

template <typename T, std::size_t N,
          typename Indices = std::make_index_sequence<N>>
auto CreateTupleFromArray(const std::array<T, N> array) {
  return CreateTupleFromArrayImpl(array, Indices{});
}

} // namespace BaseMath
} // namespace MyOptimization

#endif // CORE_BASEMATH_CONTAINER_H
