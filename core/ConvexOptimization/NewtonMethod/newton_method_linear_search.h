#ifndef CORE_CONVEXOPTIMIZATION_NEWTONMETHOD_NEWTON_METHOD_LINEAR_SEARCH_H
#define CORE_CONVEXOPTIMIZATION_NEWTONMETHOD_NEWTON_METHOD_LINEAR_SEARCH_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/StepSearch/armijo_condition.h"
#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {

template <typename T, size_t N>
class NetwonMethodLinearSearch {
 public:
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;

  NetwonMethodLinearSearch() = delete;
  ~NetwonMethodLinearSearch(){};
  /// @brief : constructor function for newton method with linear search
  /// ! for non Convex problem, newton method may get better results, because
  /// ! for linear search will delete h(x) part. later a better version will be
  /// ! added
  /// @param iteration_error
  /// @param parameter // ! for the linear search, when param within
  /// ! (0,0.5),search result is better
  /// @param step
  /// @param init_pos
  NetwonMethodLinearSearch(const float iteration_error, const float parameter,
                           const float step, EigenVector init_pos)
      : iteration_error_(iteration_error),
        armijo_parameter_(parameter),
        step_(step),
        init_step_(step),
        search_results_(init_pos) {
    // v_.setRandom();
    v_ = init_pos;
    ArmijoParameterCheck(parameter);
  }

  template <typename Function>
  void Step(const T x_epsilon, const T y_epsilon, Function func) {
    EigenVector last_iterate = search_results_;

    bool continue_gradient_descent = true;
    while (continue_gradient_descent) {
      // ! for first step, we must update gradients and hessian first, or the
      // ! following step resize can't continue
      // get gradients and hassian
      UpdateSearchDirection(func, x_epsilon);
      UpdateHessian(func, x_epsilon, y_epsilon);
      last_iterate = search_results_;

      // MLOG_ERROR("-1 * gradients " << -1 * search_direction_ << "\n, hessian
      // "
      //                              << hessian_);

      // * linear search for step
      // get v
      UseSecondOrderLossFunctionGetClosetVector<Function, T, N>(
          static_cast<T>(-1.0) * search_direction_, hessian_, v_);
      // get optim step
      step_ = init_step_;
      SecondOrderSearchWithArmijoCondition<Function, T, N>(
          armijo_parameter_, static_cast<T>(-1.0) * search_direction_, hessian_,
          v_, search_results_, func, step_);
      // MLOG_ERROR("v_ " << v_ << "\n, step " << step_);

      // * update iterate and check if continue search
      UpdateIterate();
      // MLOG_ERROR("last_iterate " << last_iterate << "\n, search_results_ "
      //  << search_results_);
      continue_gradient_descent =
          StatisfyNewtonMethodFinishCondition(last_iterate, func);
    }
    return;
  }

  const EigenVector GetSearchResults() const { return search_results_; }
  const EigenVector GetSearchDirection() const { return search_direction_; }
  const EigenMatrix GetHessian() const { return hessian_; }

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

    return (std::fabs(current_iterate_func_value - last_iterate_func_value) >=
            iteration_error_);
  }

  void UpdateIterate() { search_results_.noalias() += step_ * v_; }

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
  void UpdateHessian(Callable func, double x_epsilon, double y_epsilon) {
    hessian_.noalias() = MyOptimization::BaseMath::
        GetNumericHessianForEigenVectorByForwardDifference<Callable, T, N>(
            x_epsilon, y_epsilon, search_results_, func);

    return;
  };

 private:
  float iteration_error_;
  float armijo_parameter_;
  float step_;
  float init_step_;
  EigenMatrix hessian_;
  EigenVector search_direction_;
  EigenVector search_results_;
  EigenVector v_;
};

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_NEWTONMETHOD_NEWTON_METHOD_LINEAR_SEARCH_H
