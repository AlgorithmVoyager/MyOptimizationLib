#ifndef CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_PENALTYMETHOD_PENALTY_METHOD_H
#define CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_PENALTYMETHOD_PENALTY_METHOD_H

#include <eigen3/Eigen/Core>
#include <memory>

#include "core/ConvexOptimization/QuasiNetwonMethod/lbfgs_with_lewis_and_overton.h"
#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {

/*
 * for PenaltyMethod, its form is:
 *             min f(x)
 *             s.t. ai(x)=0    i= 0...N
 *                  cj(x)>=0   j=0... N
 * so the constriants can be non linear, so here not use matrix while using
 * function
 */
template <typename T, size_t N>
class PenaltyMethod {
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;

 public:
  PenaltyMethod() = delete;
  ~PenaltyMethod() {}

  PenaltyMethod(const EigenVector& init_pos, T iteration_error, T step,
                T step_coefficient = 0.5, T c1 = 1e-4, T c2 = 0.9,
                T epsilon = 1e-3, T sigma = 1, T sigma_coefficient = 10,
                bool weak_wolfe = true)
      : sigma_(sigma),
        sigma_coefficient_(sigma_coefficient),
        penalty_epsilon_(epsilon),
        lm_bfgs_ptr_(
            std::make_shared<MyOptimization::ConvexOptimization::
                                 LimitMemoryCautiousBFGSWithLewis<T, N, 6>>(
                init_pos, iteration_error, step)) {
    penalty_method_res_.noalias() = init_pos;
  }

  /*
   * F: value function of optimization problem
   * EC: equal constraints function , for simplfy the problem, if will prefer
   *      set its input as EigenVector
   * UEC: unequal constriants function
   * ec_size: equal constraints size
   * uec_size: unequal constriants size
   */
  template <typename F, typename EC, typename UEC, size_t ec_size,
            size_t uec_size>
  void Step(F value_func, std::vector<EC> equal_constraints,
            std::vector<UEC> unequal_constriants) {
    // * check constraints size and Callable

    // * 1. tranverse constraints optiomization as unconstraints problem,
    auto equal_constraints_part([&](EigenVector input) {
      return std::accumulate(std::begin(equal_constraints),
                             std::end(equal_constraints), T(0.0),
                             [&](T current_sum, const auto func) {
                               return current_sum += func(input) * func(input);
                             });
    });

    auto unequal_constriants_part([&](EigenVector input) {
      return std::accumulate(
          std::begin(unequal_constriants), std::end(unequal_constriants),
          T(0.0), [&](T current_sum, const auto func) {
            return current_sum += std::min(func(input), T(0.0)) *
                                  std::min(func(input), T(0.0));
          });
    });

    bool continue_search = true;

    // * 2. use lbfgs method solve optimial x*
    while (continue_search) {
      auto constrainsts_sum([&, this](EigenVector input) {
        return sigma_ * (equal_constraints_part(input) +
                         unequal_constriants_part(input));
      });

      auto whole_cost_func([&](EigenVector input) {
        return value_func(input) + constrainsts_sum(input);
      });

      // MLOG_ERROR("====" << constrainsts_sum(penalty_method_res_));
      // MLOG_ERROR("====" << whole_cost_func(penalty_method_res_));

      // * search optimal res
      lm_bfgs_ptr_->Step(whole_cost_func);
      penalty_method_res_ = lm_bfgs_ptr_->GetSearchResults();

      // * 3 check if penalty is so small to stop iteration
      continue_search = [&, this]() {
        return constrainsts_sum(penalty_method_res_) >= penalty_epsilon_;
      }();

      // * reset init x and sigma
      if (continue_search) {
        lm_bfgs_ptr_->ResetInitPos(penalty_method_res_);
        sigma_ *= sigma_coefficient_;
      }
    }
    return;
  }

  const EigenVector GetSearchResults() const { return penalty_method_res_; }

 private:
  T sigma_;
  T sigma_coefficient_;
  T penalty_epsilon_;

  EigenVector penalty_method_res_;

  std::shared_ptr<MyOptimization::ConvexOptimization::
                      LimitMemoryCautiousBFGSWithLewis<T, N, 6>>
      lm_bfgs_ptr_ = nullptr;
};

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_PENALTYMETHOD_PENALTY_METHOD_H
