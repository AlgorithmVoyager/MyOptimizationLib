#ifndef CORE_BASEMATH_FUNCTION_H
#define CORE_BASEMATH_FUNCTION_H

#include <eigen3/Eigen/Core>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/BaseMath/container.h"
#include "logging/log.h"

namespace MyOptimization {
namespace BaseMath {

/// @param: func: user self defined function
/// @param: value: the input value of function
/// @brief: the fuinction will return final value calculate by func
template <typename Callable, typename T>
T GetValue(Callable func, T t) {
  // static_assert(std::is_function<Callable>::value,
  //               "input Func must be a function!!!");
  return func(t);
}

/// @param: func: user self defined function
/// @param: value: the inputs value of function
/// @brief: the fuinction will return final value calculate by func
template <typename Callable, typename... T>
auto GetFuncValueForMultiInput(Callable func, T... t) {
  static_assert(std::is_invocable<Callable, T...>::value,
                "Input Func must be callable with input arguments");
  return func(t...);
}

// /*use value, not ref, to avoid left or right value error*/
/// @param: func: user self defined function
/// @param: container: container e.g std::vector and so on,
/// @brief: the fuinction will return final value for input container
template <typename Callable, typename Container, size_t N>
auto GetFuncValueForContainer(Callable func, Container container) {
  return std::apply(func, CreateTupleFromContainer<Container, N>(container));
}

/// @param: func: user self defined function
/// @param: container: std::array
/// @brief: the fuinction will convert a array into tuple
template <typename Callable, typename T, size_t N>
T GetFuncValueForArray(Callable func, std::array<T, N> container) {
  return std::apply(func, CreateTupleFromArray(container));
}

/// @brief
/// @tparam Callable
/// @tparam Container
/// @tparam N
/// @param func
/// @param container
/// @return
template <typename Callable, typename T, size_t N>
T GetFuncValueForEigenVector(Callable func,
                             Eigen::Matrix<T, N, 1> eigen_vector) {
  if (eigen_vector.size() == 0) {
    return 0.0;
  }

  // * for function form like f(eigen_vector)
  if constexpr (std::is_invocable_v<Callable, Eigen::Matrix<T, N, 1>>) {
    return std::invoke(func, eigen_vector);
  } else {
    // * for function form like f(x1,x2,...)
    return std::apply(func, CreateTupleFromEigenVector<T, N>(eigen_vector));
  }
}

/// @brief : the function will return a numeric grdients in Container form
/// @tparam Callable: value function
/// @tparam Container: container which store init_pos
/// @tparam T : type dealing with
/// @tparam N : N-dimesional problem
/// @param func
/// @param container
/// @param epsilon: step
/// @return
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
    gradients.emplace_back((perturbed_value - original_value) / epsilon);
  }

  return gradients;
}

/// @brief the function will return a numeric grdients in std::array form
/// @tparam Callable
/// @tparam T
/// @tparam N
/// @param func
/// @param array
/// @param epsilon
/// @return
template <typename Callable, typename T, size_t N>
std::array<T, N> GetNumericGrandientForArrayByForwardDifference(
    Callable func, std::array<T, N> array, double epsilon) {
  std::array<T, N> gradients;

  if (array.empty()) {
    return gradients;
  }

  T original_value =
      static_cast<T>(GetFuncValueForArray<Callable, T, N>(func, array));

  for (size_t i = 0U; i < N; ++i) {
    std::array<T, N> inputs = array;
    inputs[i] += epsilon;

    T perturbed_value =
        static_cast<T>(GetFuncValueForArray<Callable, T, N>(func, inputs));
    gradients[i] = (perturbed_value - original_value) / epsilon;

    MLOG_WARNING("perturbed_value " << perturbed_value << ", original_value "
                                    << perturbed_value << ", epsilon "
                                    << epsilon << ", gradients[i] "
                                    << gradients[i])
  }

  return gradients;
}

/// @brief  the function will return a numeric grdients in Container form using
/// central difference
/// @tparam Callable
/// @tparam Container
/// @tparam T
/// @tparam N
/// @param func
/// @param container
/// @param epsilon
/// @return
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

    gradients.emplace_back((next_half_step_value - last_half_step_value) /
                           epsilon);
  }

  return gradients;
}

/// @brief :  the function will return a numeric grdients in std::array form
/// using central difference
/// @tparam Callable
/// @tparam T
/// @tparam N
/// @param func
/// @param array
/// @param epsilon
/// @return
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

template <typename Callable, typename T, size_t N>
auto GetNumericGrandientForEigenVectorByForwardDifference(
    Callable func, Eigen::Matrix<T, N, 1> input_vector, double epsilon)
    -> Eigen::Matrix<T, N, 1> {
  if (input_vector.size() == 0) return {};

  const size_t vector_size = input_vector.size();

  using EigenVector = Eigen::Matrix<T, N, 1>;

  EigenVector gradients;

  for (size_t i = 0U; i < vector_size; ++i) {
    EigenVector last_half_step_inputs{input_vector};
    EigenVector next_half_step_inputs{input_vector};
    last_half_step_inputs(i) -= epsilon / 2;
    next_half_step_inputs(i) += epsilon / 2;

    // MLOG_ERROR("last_half_step_inputs " << last_half_step_inputs);

    T next_half_step_value =
        GetFuncValueForEigenVector<Callable, T, N>(func, next_half_step_inputs);
    T last_half_step_value =
        GetFuncValueForEigenVector<Callable, T, N>(func, last_half_step_inputs);

    gradients(i) =
        static_cast<T>((next_half_step_value - last_half_step_value) / epsilon);
    // MLOG_ERROR("next_half_step_value " << next_half_step_value
    //                                    << ", last_half_step_value "
    //                                    << last_half_step_value <<
    //                                    gradients(i));
  }

  return gradients;
}

template <typename Callable, typename T, size_t N>
auto GetNumericHessianForEigenVectorByForwardDifference(
    const double x_epsilon, const double y_epsilon,
    const Eigen::Matrix<T, N, 1> eigen_vector, Callable func) {
  using EigenVector = Eigen::Matrix<T, N, 1>;
  using EigenMatrix = Eigen::Matrix<T, N, N>;

  EigenMatrix hessian;
  EigenVector input_vector(eigen_vector);

  const size_t vector_size = eigen_vector.size();

  if (vector_size == 0) return hessian;

  T current_step_value =
      GetFuncValueForEigenVector<Callable, T, N>(func, input_vector);

  /**
   *  for x,x
   *  ((f(x+h)-f(x)/h) - (f(x)-f(x-h)/h))/h
   *
   *  for x,y
   *  (f(x+h，y+k)-f(x+h,y-k)-f(x-h,y+k)+f(x-h,y-k))/4hk
   *
   */
  for (size_t i = 0U; i < vector_size; ++i) {
    input_vector(i) += x_epsilon;
    T next_step_value =
        GetFuncValueForEigenVector<Callable, T, N>(func, input_vector);
    input_vector(i) -= 2 * x_epsilon;
    T last_step_value =
        GetFuncValueForEigenVector<Callable, T, N>(func, input_vector);
    hessian(i, i) =
        (next_step_value + last_step_value - 2 * current_step_value) /
        (x_epsilon * x_epsilon);
    // MLOG_ERROR("next_step_value " << next_step_value << ",last_step_value "
    //                               << last_step_value << ",  hessian(i, i) "
    //                               << hessian(i, i));

    // reset input value
    input_vector(i) = eigen_vector(i);

    for (size_t j = i + 1; j < vector_size; ++j) {
      // f(x+h,y+k)
      input_vector(i) += x_epsilon;
      input_vector(j) += y_epsilon;
      T up_right_value =
          GetFuncValueForEigenVector<Callable, T, N>(func, input_vector);
      // f(x+h,y-k)
      input_vector(j) -= 2 * y_epsilon;
      T down_right_value =
          GetFuncValueForEigenVector<Callable, T, N>(func, input_vector);
      // f(x-h,y-k)
      input_vector(i) -= 2 * x_epsilon;
      T down_left_value =
          GetFuncValueForEigenVector<Callable, T, N>(func, input_vector);
      // f(x-h,y+k)
      input_vector(j) += 2 * x_epsilon;
      T up_left_value =
          GetFuncValueForEigenVector<Callable, T, N>(func, input_vector);

      // calculate current d^2f/dxdy
      hessian(i, j) = (up_right_value + down_left_value - down_right_value -
                       up_left_value) /
                      (x_epsilon * y_epsilon);
      hessian(j, i) = hessian(i, j);
      // MLOG_ERROR("up_right_value "
      //            << up_right_value << ",down_left_value " << down_left_value
      //            << "down_left_value " << down_left_value << ",up_left_value
      //            "
      //            << up_left_value << ",  hessian(i, j) " << hessian(i, j));

      // reset input value
      input_vector(i) = eigen_vector(i);
      input_vector(j) = eigen_vector(j);
    }
  }

  // MLOG_ERROR("hessian " << hessian);
  return hessian;
}

}  // namespace BaseMath

}  // namespace MyOptimization

#endif  // CORE_BASEMATH_FUNCTION_H
