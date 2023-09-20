#ifndef CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_BFGS_H
#define CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_BFGS_H

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/StepSearch/armijo_condition.h"
#include "logging/log.h"
namespace MyOptimization {
namespace ConvexOptimization {

template <typename T, size_t N>
class BFGS {
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;

 public:
  BFGS() = delete;
  ~BFGS() {}

  BFGS(const EigenVector& init_pos, T iteration_error,
       T linear_search_parameter, T step, T step_coefficient = 0.5,
       T epsilon = 1e-3)
      : epsilon_(epsilon),
        iteration_error_(iteration_error),
        linear_search_parameter_(linear_search_parameter),
        step_(step),
        init_step_(step),
        step_descresing_coefficient_(step_coefficient),
        search_results_(init_pos) {
    // * check armijo linear search condition
    ArmijoParameterCheck(linear_search_parameter_);
  }

  const EigenVector GetSearchResults() const { return search_results_; }

  template <typename Callable>
  void Step(Callable func) {
    // * init
    Init(func);

    int iter_step = 0;
    // * main loop
    while (!SatisifySearchFinishCondition()) {
      // * update d by b matrix and gradient
      EigenVector d_vector;
      d_vector.noalias() = static_cast<T>(-1.0) * b_matrix_ * gradients_;
      MLOG_INFO("current b_matrix_ \n" << b_matrix_);
      MLOG_INFO("current d vector  \n" << d_vector);

      // * reset step & linear search step
      step_ = init_step_;
      LinearSearchForEigenVectorWithArmijoCondition<Callable, T, N>(
          linear_search_parameter_, step_descresing_coefficient_, step_,
          search_results_, d_vector, func);
      MLOG_INFO("current linear search gotten step " << step_);

      // * update iterate
      EigenVector last_x;
      last_x.noalias() = search_results_;
      UpdateIterate(d_vector);

      // * update gradients in k+1 step
      EigenVector last_gradients;
      last_gradients.noalias() = gradients_;
      gradients_.noalias() = MyOptimization::BaseMath::
          GetNumericGrandientForEigenVectorByForwardDifference<Callable, T, N>(
              func, search_results_, epsilon_);

      // * update b matrix in k+1 Step
      EigenVector delta_x;
      delta_x.noalias() = search_results_ - last_x;
      EigenVector delta_g;
      delta_g.noalias() = gradients_ - last_gradients;
      UpdateBMatrix(delta_x, delta_g);

      iter_step++;
      if (iter_step > 100000) {
        MLOG_INFO(
            "search step has passed 100000 times, but still not get optimal "
            "result!")
        return;
      }
    }
  }

 private:
  template <typename Callable>
  void Init(Callable func) {
    // * init b matrix and gradients
    gradients_.noalias() = MyOptimization::BaseMath::
        GetNumericGrandientForEigenVectorByForwardDifference<Callable, T, N>(
            func, search_results_, epsilon_);
    b_matrix_.noalias() =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(N, N);
  }

  void UpdateIterate(const EigenVector& d_vector) {
    search_results_.noalias() += step_ * d_vector;
  }

  void UpdateBMatrix(const EigenVector& delta_x, const EigenVector& delta_g) {
    T devide_part = delta_g.transpose().dot(delta_x);

    EigenMatrix L_B, U_B, add_part;

    L_B.noalias() =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(N, N) -
        (delta_x * delta_g.transpose()) / devide_part;

    U_B.noalias() =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(N, N) -
        (delta_g * delta_x.transpose()) / devide_part;

    add_part.noalias() = (delta_x * delta_x.transpose()) / devide_part;

    b_matrix_.noalias() = L_B * b_matrix_ * U_B + add_part;
    return;
  }

  bool SatisifySearchFinishCondition() {
    T gradients_second_order_norm = 0.0;
    std::for_each(gradients_.data(), gradients_.data() + gradients_.size(),
                  [&gradients_second_order_norm](const auto& gradient) {
                    gradients_second_order_norm += gradient * gradient;
                  });
    return (gradients_second_order_norm <= iteration_error_);
  }

 private:
  T epsilon_;
  T iteration_error_;
  T linear_search_parameter_;
  T step_;
  T init_step_;
  T step_descresing_coefficient_;

  EigenMatrix b_matrix_;
  EigenVector gradients_;
  EigenVector search_results_;
};
}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_BFGS_H
