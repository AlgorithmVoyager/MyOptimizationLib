#ifndef CORE_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_UTILS_UTILS_H
#define CORE_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_UTILS_UTILS_H

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {
namespace utils {

// template <typename Function> void ValueFunctionCheck(Function value_function)
// {
//   if (!std::is_function_v<std::remove_cv<Function>>) {
//     try {
//       throw std::runtime_error(
//           "Input Value Function is not a functional type!!! "
//           "Construct SteepestGradientDescent Failed");
//     } catch (const std::exception &e) {
//       MLOG_ERROR("[Runtime Error]: " << e.what());
//     }
//   }
//   return;
// }

// template <typename Function, typename GradientFunction>
// void ValueAndGradientFunctionCheck(const Function &value_function,
//                                    const GradientFunction &gradient_fucntion)
//                                    {
//   if (!std::is_function_v<value_function> &&
//       std::is_function_v<gradient_fucntion>) {
//     try {
//       throw std::runtime_error(
//           "Input Value Or Gradient Function is not a functional type!!! "
//           "Construct SteepestGradientDescent Failed");
//     } catch (const std::exception &e) {
//       MLOG_ERROR("[Runtime Error]: " << e.what());
//     }
//   }
//   return;
// }

// template <typename Container, size_t N>
// void ContainerSizeCheck(const Container &container) {
//   if (container.size() != N) {
//     try {
//       throw std::runtime_error(
//           "Input Container Size must same with N, Construct "
//           "SteepestGradientDescent Failed!!!");
//     } catch (const std::exception &e) {
//       MLOG_ERROR("[Runtime Error]: " << e.what());
//     }
//     return;
//   }
//   return;
// }

// template <typename Function, typename... Args>
// void ArgsCheck(Function value_function, Args... args) {
//   if (std::is_invocable<Function, Args...>::value) {
//     try {
//       throw std::runtime_error(
//           "Input Function And Args not match, Can't run the function,"
//           "Construct SteepestGradientDescent Failed!!!");
//     } catch (const std::exception &e) {
//       MLOG_ERROR("[Runtime Error]: " << e.what());
//     }
//   }
//   return;
// }

template <typename T, size_t N>
std::array<T, N> GetNextStepInputValue(const std::array<T, N> &init_input_value,
                                       const std::array<T, N> &gradients,
                                       const float step) {
  if (init_input_value.size() != gradients.size()) {
    try {
      throw std::runtime_error(
          "Input value and gradients value size not same!!!");
    } catch (const std::exception &e) {
      MLOG_ERROR("[Runtime Error]: " << e.what());
    }
  }
  std::array<T, N> next_step_input_value;
  std::transform(
      init_input_value.begin(), init_input_value.end(), gradients.begin(),
      next_step_input_value.begin(),
      [&step](const T &lhs, const T &rhs) { return lhs + step * rhs; });
  return next_step_input_value;
}

template <typename T, size_t N>
T GetDotProductOfTwoArray(const std::array<T, N> &lhs,
                          const std::array<T, N> &rhs) {
  if (lhs.size() != rhs.size()) {
    try {
      throw std::runtime_error("The inputed Two Array's size not same!!!");
    } catch (const std::exception &e) {
      MLOG_ERROR("[Runtime Error]: " << e.what());
    }
  }
  T dot_product = std::inner_product(lhs.begin(), lhs.end(), rhs.begin(),
                                     static_cast<T>(0));
  return dot_product;
}

template <typename T, typename Lambda, size_t N>
void GetArraySelfMultiLambda(const Lambda &lambda,
                             std::array<T, N> &init_input_value) {
  std::transform(init_input_value.begin(), init_input_value.end(),
                 init_input_value.begin(), [&](T x) { return lambda * x; });
  return;
}

template <typename T, typename Lambda, size_t N>
std::array<T, N> GetArrayMultiLambda(const Lambda &lambda,
                                     const std::array<T, N> &init_input_value) {
  std::array<T, N> multi_res;
  std::transform(init_input_value.begin(), init_input_value.end(),
                 multi_res.begin(), [&](T x) { return lambda * x; });
  return multi_res;
}

template <typename T, size_t N>
void SelfAddArray(const std::array<T, N> &increment_input_value,
                  std::array<T, N> &input_value) {
  if (increment_input_value.size() != input_value.size()) {
    try {
      throw std::runtime_error("The inputed Two Array's size not same!!!");
    } catch (const std::exception &e) {
      MLOG_ERROR("[Runtime Error]: " << e.what());
    }
  }
  std::transform(increment_input_value.begin(), increment_input_value.end(),
                 input_value.begin(), input_value.begin(),
                 [](const T &lhs, T &rhs) { return rhs += lhs; });
  return;
}

} // namespace utils
} // namespace ConvexOptimization

} // namespace MyOptimization

#endif // CORE_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_UTILS_UTILS_H
