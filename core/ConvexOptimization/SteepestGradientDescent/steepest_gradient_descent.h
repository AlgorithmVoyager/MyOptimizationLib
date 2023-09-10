#ifndef MYOPTIMIZATION_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_STEEPESTGRADIENTDESCENT_H_
#define MYOPTIMIZATION_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_STEEPESTGRADIENTDESCENT_H_

#include <array>
#include <optional>

namespace MyOptimization {
namespace ConvexOptimization {

template <typename Function, typename GradientFunction, typename T, size_t N>
class SteepestGradientDescent {
 public:
  SteepestGradientDescent() = delete;
  ~SteepestGradientDescent(){};

  /// @brief function for input as array and use numeric method to calculate
  /// gradients
  /// @param step
  /// @param parameter
  /// @param
  SteepestGradientDescent(const float step, const float parameter,
                          const Function& value_function,
                          const std::array<T, N>& init_pos);

  /// @brief function for input as array and use function method to calculate
  /// gradients
  /// @tparam GradientFunction
  /// @param step
  /// @param parameter
  /// @param function
  /// @param gradient_fucntion
  /// @param init_pos
  SteepestGradientDescent(const float step, const float parameter,
                          const Function& value_function,
                          const GradientFunction& gradient_fucntion,
                          const std::array<T, N>& init_pos);

  /// @brief function for input as container and use numeric method to calculate
  /// gradients
  /// @param step
  /// @param parameter
  /// @param
  template <typename Container>
  SteepestGradientDescent(const float step, const float parameter,
                          const Function& value_function,
                          const Container& init_pos);

  /// @brief function for input as container and use function method to
  /// calculate gradients
  /// @tparam Container
  /// @param step
  /// @param parameter
  /// @param function
  /// @param gradient_fucntion
  /// @param init_pos
  template <typename Container>
  SteepestGradientDescent(const float step, const float parameter,
                          const Function& value_function,
                          const GradientFunction& gradient_fucntion,
                          const Container& init_pos);

  /// @brief function for input as series variales and use numeric method to
  /// calculate gradients
  /// @tparam ...Args
  /// @param step
  /// @param parameter
  /// @param function
  /// @param ...args
  template <typename... Args>
  SteepestGradientDescent(const float step, const float parameter,
                          const Function& value_function, const Args&... args);

  /// @brief  function for input as series variables and use function method to
  /// calculate gradients
  /// @tparam GradientFunction
  /// @tparam ...Args
  /// @param step
  /// @param parameter
  /// @param function
  /// @param ...args
  template <typename... Args>
  SteepestGradientDescent(const float step, const float parameter,
                          const Function& value_function,
                          const GradientFunction& gradient_fucntion,
                          const Args&... args);

  /// @brief : main function loop
  void Step();

 private:
  /// @brief : choose griandent descent direction
  void ChooseSerachDirection();

  /// @brief : check if current loop, if it has satisify armijo linear search
  /// condition
  /// @return
  bool StatisfyArmijoCondition();

  /// @brief : update iterator value
  void UpdateIterate();

 private:
  std::array<T, N> search_direction_;
  std::array<T, N> final_search_results_;
  float step_;
  const float const_parameter_for_amijro_condition_;
  Function value_function_;
  std::optional<GradientFunction> gradient_function_;
};

}  // namespace ConvexOptimization

}  // namespace MyOptimization

#endif  // MYOPTIMIZATION_CONVEXOPTIMIZATION_STEEPESTGRADIENTDESCENT_STEEPESTGRADIENTDESCENT_