#ifndef CORE_CONVEXOPTIMIZATION_NEWTONMETHOD_NEWTON_METHOD_H
#define CORE_CONVEXOPTIMIZATION_NEWTONMETHOD_NEWTON_METHOD_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "core/BaseMath/function.h"
#include "logging/log.h"
namespace MyOptimization {
namespace ConvexOptimization {

/**
 *  x_k+1 = x_k -(d^2f)^(-1)Df
 * */
template <typename T, size_t N>
class NewtonMethod {
 public:
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;

  NewtonMethod() = delete;
  ~NewtonMethod(){};

  NewtonMethod(const float iteration_error, EigenVector init_pos)
      : iteration_error_(iteration_error), search_results_(init_pos) {}

  template <typename Function>
  void Step(const T x_epsilon, const T y_epsilon, Function func) {
    EigenVector last_iterate = search_results_;

    bool continue_gradient_descent = true;
    while (continue_gradient_descent) {
      UpdateSearchDirection(func, x_epsilon);
      UpdateNewtonStep(func, x_epsilon, y_epsilon);
      last_iterate = search_results_;
      UpdateIterate();
      continue_gradient_descent =
          StatisfyNewtonMethodFinishCondition(last_iterate, func);
    }
    return;
  }

  const EigenVector GetSearchResults() const { return search_results_; }
  const EigenVector GetSearchDirection() const { return search_direction_; }
  const EigenMatrix GetNewtonStep() const { return newton_step_; }

 private:
  template <typename Function>
  bool StatisfyNewtonMethodFinishCondition(const EigenVector &last_iterator,
                                           Function value_function) {
    T last_iterate_func_value =
        MyOptimization::BaseMath::GetFuncValueForEigenVector<Function, T, N>(
            value_function, last_iterator);
    T current_iterate_func_value =
        MyOptimization::BaseMath::GetFuncValueForEigenVector<Function, T, N>(
            value_function, search_results_);

    // bool iteration_value_near_real_value = std::all_of(
    //     search_direction_.begin(), search_direction_.end(),
    //     [this](auto &val) { return std::fabs(val) <= iteration_error_; });

    // MLOG_ERROR("last_iterator "
    //            << last_iterator << ", current_searched_results_ "
    //            << search_results_ << "last_iterate_func_value "
    //            << last_iterate_func_value << ",current_iterate_func_value "
    //            << current_iterate_func_value);
    return (std::fabs(current_iterate_func_value - last_iterate_func_value) >=
            iteration_error_);
    //   &&  iteration_value_near_real_value;
  }

  void UpdateIterate() { search_results_ += newton_step_ * search_direction_; }

  template <typename Callable>
  void UpdateSearchDirection(Callable func, double epsilon) {
    if (search_results_.size() == 0) {
      return;
    }

    auto res = MyOptimization::BaseMath::
        GetNumericGrandientForEigenVectorByForwardDifference<Callable, T, N>(
            func, search_results_, epsilon);
    search_direction_ = static_cast<T>(-1.0) * res;
    return;
  }

  template <typename Callable>
  void UpdateNewtonStep(Callable func, double x_epsilon, double y_epsilon) {
    EigenMatrix hessian = MyOptimization::BaseMath::
        GetNumericHessianForEigenVectorByForwardDifference<Callable, T, N>(
            x_epsilon, y_epsilon, search_results_, func);
    newton_step_ = hessian.inverse();

    return;
  };

 private:
  float iteration_error_;
  EigenMatrix newton_step_;
  EigenVector search_direction_;
  EigenVector search_results_;
};

}  // namespace ConvexOptimization
}  // namespace MyOptimization
#endif  // CORE_CONVEXOPTIMIZATION_NEWTONMETHOD_NEWTON_METHOD_H
