#ifndef CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_BARRIERMETHOD_BARRIER_METHOD_H
#define CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_BARRIERMETHOD_BARRIER_METHOD_H

#include <cmath>
#include <eigen3/Eigen/Core>
#include <memory>

#include "core/ConvexOptimization/QuasiNetwonMethod/lbfgs_with_lewis_and_overton.h"
#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {

/*
 * barrier_method is a method solve optimization problem with unequal
 * constraints, its form like this:
 *              min f(x)
 *              s.t c_i(x) >= 0
 *
 * barrier_function is an ill-condition optiomization method
 *
 * transform problem into:
 *             min f(x) + r*B(x)
 * B(x) may be one of the following form
 *              1. B(x) =Sum:  inv(c_i(x)) = 1/(c_i(x))
 *              2. B(x) =Sum: - ln(c_i(x))
 */

template <typename T, size_t N>
class BarrierMethod {
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;

 public:
  BarrierMethod() = delete;
  ~BarrierMethod() {}

  BarrierMethod(const EigenVector& init_pos, T iteration_error, T step,
                T step_coefficient = 0.5, T c1 = 1e-4, T c2 = 0.9,
                T epsilon = 1e-3, T gamma = 10, T gamma_coefficient = 0.1,
                bool weak_wolfe = true)
      : gamma_(gamma),
        gamma_coefficient_(gamma_coefficient),
        barrier_epsilon_(epsilon),
        lm_bfgs_ptr_(
            std::make_shared<MyOptimization::ConvexOptimization::
                                 LimitMemoryCautiousBFGSWithLewis<T, N, 6>>(
                init_pos, iteration_error, step)) {
    barrier_method_res_.noalias() = init_pos;
  }

  /*
   * F: value function of optimization problem
   * UEC: unequal constriants function
   * uec_size: unequal constriants size
   * transfrom_index: show which transform here use
   */
  template <typename F, typename UEC, size_t uec_size>
  void Step(F value_func, std::vector<UEC> unequal_constriants,
            const size_t transfrom_index = 1) {
    // * 1. transform unequal_constriants into B(x)
    UEC barrier_function;
    if (transfrom_index == 0) {
      barrier_function = ([&](EigenVector input) {
        return std::accumulate(
            std::begin(unequal_constriants), std::end(unequal_constriants),
            T(0.0), [&](T current_sum, const auto ue_c) {
              return current_sum +=
                     T(1.0) / (std::abs(ue_c(input)) < 1e-6
                                   ? ue_c(input) + barrier_epsilon_
                                   : ue_c(input));
            });
      });
    } else if (transfrom_index == 1) {
      barrier_function = ([&](EigenVector input) {
        return std::accumulate(
            std::begin(unequal_constriants), std::end(unequal_constriants),
            T(0.0), [&](T current_sum, const auto ue_c) {
              return current_sum -=
                     std::log(std::abs(ue_c(input)) < 1e-6
                                  ? ue_c(input) + barrier_epsilon_
                                  : ue_c(input));
            });
      });
    } else {
      barrier_function = ([&](EigenVector input) {
        return std::accumulate(
            std::begin(unequal_constriants), std::end(unequal_constriants),
            T(0.0), [&](T current_sum, const auto ue_c) {
              return current_sum +=
                     std::exp(T(1.0) / (std::abs(ue_c(input)) < 1e-6
                                            ? ue_c(input) + barrier_epsilon_
                                            : ue_c(input)));
            });
      });
    }

    // * 2. search optimal res
    bool continue_search = true;
    while (continue_search) {
      // * create whloe cost function
      auto whole_cost_func([&](EigenVector input) {
        return value_func(input) + gamma_ * barrier_function(input);
      });

      // * search optimal res
      lm_bfgs_ptr_->Step(whole_cost_func);
      barrier_method_res_ = lm_bfgs_ptr_->GetSearchResults();

      // check if continue_search
      continue_search = [&, this]() {
        return gamma_ * barrier_function(barrier_method_res_) >
               barrier_epsilon_;
      }();

      // reset gamma and re-search
      if (continue_search) {
        lm_bfgs_ptr_->ResetInitPos(barrier_method_res_);
        gamma_ *= gamma_coefficient_;
      }
    }
  }

  const EigenVector GetSearchResults() const { return barrier_method_res_; }

 private:
  T gamma_;
  T gamma_coefficient_;
  T barrier_epsilon_;

  EigenVector barrier_method_res_;

  std::shared_ptr<MyOptimization::ConvexOptimization::
                      LimitMemoryCautiousBFGSWithLewis<T, N, 6>>
      lm_bfgs_ptr_ = nullptr;
};

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_BARRIERMETHOD_BARRIER_METHOD_H
