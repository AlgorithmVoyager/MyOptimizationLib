#ifndef CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_PRHAGUMENTEDLAGRANGIANMETHOD_PHR_AUGMENTED_LAGRANGIAN_METHOD_H
#define CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_PRHAGUMENTEDLAGRANGIANMETHOD_PHR_AUGMENTED_LAGRANGIAN_METHOD_H

#include <cmath>
#include <eigen3/Eigen/Core>
#include <memory>

#include "core/ConvexOptimization/QuasiNetwonMethod/lbfgs_with_lewis_and_overton.h"
#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {

/*
 * for PenaltyMethod, its form is:
 *             min f(x)
 *             s.t.h(x) = 0
 * so the constriants can be non linear, so here not use matrix while using
 * function
 *
 * PHR Augmented Lagrangian is
 * L(x,lambda) = f(x) + lambda * h(x) - 1/(2* rou)|| lambda - lambda_bar ||^2
 * |
 * v
 * L(x,lambda) = f(x) + lambda_bar * h(x) + 1/(2* rou)||h(x)||^2
 */

template <typename T, size_t N, size_t ec_size>
class PHRAugmentedLagrangianMethod {
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;
  using ConstriantVector = Eigen::Matrix<T, ec_size, 1>;

 public:
  PHRAugmentedLagrangianMethod() = delete;
  ~PHRAugmentedLagrangianMethod() {}

  PHRAugmentedLagrangianMethod(const EigenVector& init_pos,
                               const ConstriantVector& lambda,
                               T iteration_error, T step,
                               T step_coefficient = 0.5, T c1 = 1e-4,
                               T c2 = 0.9, T epsilon = 1e-3,
                               T lambda_coefficient = 10,
                               bool weak_wolfe = true)
      : rou_(lambda_coefficient),
        penalty_epsilon_(epsilon),
        lm_bfgs_ptr_(
            std::make_shared<MyOptimization::ConvexOptimization::
                                 LimitMemoryCautiousBFGSWithLewis<T, N, 6>>(
                init_pos, iteration_error, step)) {
    alm_method_res_.noalias() = init_pos;
    lambda_.noalias() = lambda;
  }

  template <typename F, typename EC>
  void Step(F value_func, std::vector<EC> equal_constraints) {
    // * 1. construct L(x)
    auto augmentation_part([&](EigenVector input) {
      return 0.5 * rou_ *
             std::accumulate(
                 std::begin(equal_constraints), std::end(equal_constraints),
                 T(0.0),
                 [&](T current_sum, const auto current_equal_constraint) {
                   return current_sum +=
                          std::pow(current_equal_constraint(input), 2);
                 });
    });

    int step_count(0);

    bool continue_search = true;
    while (continue_search) {
      // * 1. update L(x)
      auto lagrangian_part([&](EigenVector input) {
        size_t i(0);
        return std::accumulate(
            std::begin(equal_constraints), std::end(equal_constraints), T(0.0),
            [&](T current_sum, const auto current_equal_constraint) {
              return current_sum +=
                     current_equal_constraint(input) * lambda_[i++];
            });
      });

      auto whole_cost_func([&](EigenVector input) {
        return value_func(input) + lagrangian_part(input) +
               augmentation_part(input);
      });

      // * 2. search optimial res
      lm_bfgs_ptr_->Step(whole_cost_func);
      alm_method_res_ = lm_bfgs_ptr_->GetSearchResults();

      // * check if search finished
      continue_search = [&, this]() {
        return lagrangian_part(alm_method_res_) +
                   augmentation_part(alm_method_res_) <
               penalty_epsilon_;
      }();

      // * 3. update current lambda
      for (size_t i(0); i < ec_size; i++) {
        lambda_[i] += rou_ * equal_constraints[i](alm_method_res_);
      }

      // * 4. stop condition
      step_count++;
      if (step_count > 10000) {
        MLOG_WARNING(
            "step count has exceed 100000 times, and still not get the optimal "
            "res");
        return;
      }

      continue_search = false;
    }
    return;
  }

  const EigenVector GetSearchResults() const { return alm_method_res_; }

 private:
  T rou_;
  T penalty_epsilon_;

  ConstriantVector lambda_;
  EigenVector alm_method_res_;

  std::shared_ptr<MyOptimization::ConvexOptimization::
                      LimitMemoryCautiousBFGSWithLewis<T, N, 6>>
      lm_bfgs_ptr_ = nullptr;
};

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_PRHAGUMENTEDLAGRANGIANMETHOD_PHR_AUGMENTED_LAGRANGIAN_METHOD_H
