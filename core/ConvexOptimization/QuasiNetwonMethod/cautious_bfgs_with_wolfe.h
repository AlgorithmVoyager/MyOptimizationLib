#ifndef CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_CAUTIOUS_BFGS_WITH_WOLFE_H
#define CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_CAUTIOUS_BFGS_WITH_WOLFE_H

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/StepSearch/wolfe_condition.h"
#include "logging/log.h"
namespace MyOptimization {
namespace ConvexOptimization {

template <typename T, size_t N>
class CautiousBFGSWithWolfe {
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;

 public:
  CautiousBFGSWithWolfe() = delete;
  ~CautiousBFGSWithWolfe() {}

  CautiousBFGSWithWolfe(const EigenVector& init_pos, T iteration_error, T step,
                        T step_coefficient = 0.5, T c1 = 1e-4, T c2 = 0.9,
                        T epsilon = 1e-3, bool weak_wolfe = true)
      : epsilon_(epsilon),
        iteration_error_(iteration_error),
        step_(step),
        init_step_(step),
        step_descresing_coefficient_(step_coefficient),
        c1_(c1),
        c2_(c2),
        weak_wolfe_(weak_wolfe),
        search_results_(init_pos) {
    // * check wolfe linear search condition's parameter
    WolfeConditionParameterCheck(c1_, c2_);
  }

  const EigenVector GetSearchResults() const { return search_results_; }

  template <typename Callable>
  void Step(Callable func) {
    // * init
    Init(func);

    int iter_step = 0;
    // * main loop
    MLOG_INFO("SatisifySearchFinishCondition "
              << SatisifySearchFinishCondition())

    // * update norm of gradients before search
    UpdateGradientSecondOrderNorm();
    while (!SatisifySearchFinishCondition()) {
      // * update d by b matrix and gradient
      EigenVector d_vector;
      d_vector.noalias() = static_cast<T>(-1.0) * b_matrix_ * gradients_;
      MLOG_INFO("current b_matrix_ \n" << b_matrix_);
      MLOG_INFO("current d vector  \n" << d_vector);

      // * reset step & linear search step
      step_ = init_step_;
      if (weak_wolfe_) {
        LinearSearchForEigenVectorWithWeakWolfeCondition<Callable, T, N>(
            search_results_, d_vector, step_, func,
            step_descresing_coefficient_, c1_, c2_);
      } else {
        LinearSearchForEigenVectorWithStrongWolfeCondition<Callable, T, N>(
            search_results_, d_vector, step_, func,
            step_descresing_coefficient_, c1_, c2_);
      }
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
      CautiouslyUpdateBMatrix(delta_x, delta_g);

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

  void CautiouslyUpdateBMatrix(const EigenVector& delta_x,
                               const EigenVector& delta_g) {
    T devide_part = delta_g.transpose().dot(delta_x);
    const T epsilon_cautious_update = 1e-6;
    T dot_of_deleta_x = delta_x.transpose().dot(delta_x);

    // * calculat gradients_second_order_norm
    UpdateGradientSecondOrderNorm();
    if (devide_part <= epsilon_cautious_update * gradients_second_order_norm_ *
                           dot_of_deleta_x) {
      return;
    }

    // * when last left part > right part, update b_matrix_
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

  void UpdateGradientSecondOrderNorm() {
    gradients_second_order_norm_ = 0.0;
    std::for_each(gradients_.data(), gradients_.data() + gradients_.size(),
                  [this](const auto& gradient) {
                    gradients_second_order_norm_ += gradient * gradient;
                  });
    return;
  }

  bool SatisifySearchFinishCondition() {
    return gradients_second_order_norm_ <= iteration_error_;
  }

 private:
  T epsilon_;
  T iteration_error_;
  T step_;
  T init_step_;
  T step_descresing_coefficient_;
  const T c1_;
  const T c2_;
  const bool weak_wolfe_;

  T gradients_second_order_norm_ = 0.0;

  EigenMatrix b_matrix_;
  EigenVector gradients_;
  EigenVector search_results_;
};
}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_CAUTIOUS_BFGS_WITH_WOLFE_H
