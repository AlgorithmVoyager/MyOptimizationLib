#ifndef MYOPTIMIZATION_BASEMATH_CONTAINER_H_
#define MYOPTIMIZATION_BASEMATH_CONTAINER_H_
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace MyOptimization {
namespace BaseMath {

// /// @brief: convert container to tuple
// /// @tparam Container
// /// @tparam ...Index
// /// @param container
// /// @param
// /// @return
// template <typename Container, std::size_t... Index>
// auto CreateTupleFromContainerImpl(Container&& container,
//                                   std::index_sequence<Index...>);

// /// @brief: convert container to tuple
// /// @tparam Container
// /// @tparam N
// /// @param container
// /// @return
// template <typename Container, std::size_t N>
// auto CreateTupleFromContainer(Container&& container);

// /// @brief : convert container to tuple for lvalue
// /// @tparam Container
// /// @tparam N
// /// @param container
// /// @return
// template <typename Container, std::size_t N>
// auto CreateTupleFromContainer(Container& container);

// /// @brief
// /// @tparam Array
// /// @tparam ...Index
// /// @param container
// /// @param
// /// @return
// template <typename Array, std::size_t... Index>
// auto CreateTupleFromArrayImpl(Array&& container,
// std::index_sequence<Index...>);

// /// @brief
// /// @tparam T
// /// @tparam Indices
// /// @tparam N
// /// @param array
// /// @return
// template <typename T, std::size_t N,
//           typename Indices = std::make_index_sequence<N>>
// auto CreateTupleFromArray(const std::array<T, N>&& array);

// /// @brief
// /// @tparam T
// /// @tparam Indices
// /// @tparam N
// /// @param array
// /// @return
// template <typename T, std::size_t N,
//           typename Indices = std::make_index_sequence<N>>
// auto CreateTupleFromArray(const std::array<T, N>& array);

template <typename Container, std::size_t... Index>
auto CreateTupleFromContainerImpl(Container&& container,
                                  std::index_sequence<Index...>) {
  auto begin = std::begin(std::forward<Container>(container));
  auto end = std::end(std::forward<Container>(container));
  if (begin == end) {
    throw std::runtime_error(
        "Input Variable Container must be not empty,but intput variable "
        "container is empty !!!");
  }
  return std::make_tuple(std::forward<Container>(container)[Index]...);
}

template <typename Container, std::size_t N>
auto CreateTupleFromContainer(Container&& container) {
  return CreateTupleFromContainerImpl(std::forward<Container>(container),
                                      std::make_index_sequence<N>{});
}

template <typename Container, std::size_t N>
auto CreateTupleFromContainer(Container& container) {
  return CreateTupleFromContainerImpl(container, std::make_index_sequence<N>{});
}

template <typename Array, std::size_t... Index>
auto CreateTupleFromArrayImpl(Array&& container,
                              std::index_sequence<Index...>) {
  if (container.empty()) {
    throw std::runtime_error(
        "Input Variable Container must be not empty,but intput variable "
        "container is empty !!!");
  }
  return std::make_tuple(std::forward<Array>(container)[Index]...);
}

template <typename T, std::size_t N,
          typename Indices = std::make_index_sequence<N>>
auto CreateTupleFromArray(const std::array<T, N>&& array) {
  return CreateTupleFromArrayImpl(std::forward<decltype(array)>(array),
                                  Indices{});
}

template <typename T, std::size_t N,
          typename Indices = std::make_index_sequence<N>>
auto CreateTupleFromArray(const std::array<T, N>& array) {
  return CreateTupleFromArrayImpl(array, Indices{});
}

}  // namespace BaseMath
}  // namespace MyOptimization

#endif  // MYOPTIMIZATION_BASEMATH_CONTAINER_H_