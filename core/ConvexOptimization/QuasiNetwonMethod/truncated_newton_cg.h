#ifndef CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_TRUNCATED_NEWTON_CG_H
#define CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_TRUNCATED_NEWTON_CG_H

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vector>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/StepSearch/armijo_condition.h"
#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {

/// @brief : // ! The function aims to the form like following:
// * f(x) = 0.5 * x_T * A * x - b_T * x.
// * and its min f(x) is equals to A * x = b;
// * so when use the following class, we need firstly get its A matrix and b
// * vector by ourselves
/// @tparam T
/// @tparam N
/// @tparam M
template <typename T, size_t N, size_t M>
class TruncatedNewtonConjugateGradientMethodWithArmijo {
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;

 public:
  TruncatedNewtonConjugateGradientMethodWithArmijo() = delete;
  ~TruncatedNewtonConjugateGradientMethodWithArmijo() {}

  TruncatedNewtonConjugateGradientMethodWithArmijo(
      const EigenVector &init_pos, const EigenVector &b, const EigenMatrix &A,
      T iteration_error, T step, T epsilon = 1e-3,
      const T armijo_parameter = 0.2, const T step_descresing_coefficient = 0.5)
      : epsilon_(epsilon),
        iteration_error_(iteration_error),
        step_(step),
        init_step_(step),
        search_results_(init_pos),
        parameter_(armijo_parameter),
        step_descresing_coefficient_(step_descresing_coefficient) {
    b_vector_.noalias() = b;
    A_matrix_.noalias() = A;
    ArmijoParameterCheck(parameter_);
  }

  const EigenVector GetSearchResults() const { return search_results_; }

  template <typename Callable>
  void Step(Callable func) {
    int step = 0;

    // * main while loop
    while (!SatisifySearchFinishCondition<Callable>(func)) {
      // * init
      Init(func);

      // * update epsilon_k
      T epsilon_k =
          std::fmin(static_cast<T>(1.0), gradients_second_order_norm_) / 10.0;

      int j_step = 0;
      while (SatisifyUpdateSearchDirectionCondition(epsilon_k)) {
        // * check if ill condition
        if (u_vector_.transpose() * A_matrix_ * u_vector_ <=
            static_cast<T>(0.0)) {
          if (j_step == 0) {
            search_direction_ = -1 * gradients_;
          }
          break;
        }

        // * update alpha
        UpdateAlhpa();

        // * update search direction
        UpdateSearchDirection();

        // * update residual vector
        UpdateResVector();

        // * update cg beta & update new conjugate direction vector
        UpdateConjugateVector();

        if (j_step > 100000) {
          MLOG_INFO(
              "search step has passed 100000 times, but still not get optimal "
              "result!"
              << "gradients_second_order_norm_ = "
              << gradients_second_order_norm_)
          return;
        }
      }
      // * update step, armijo linear search
      UpdateSetp<Callable>(func, search_direction_);

      // * update Iterator
      UpdateIterate(search_direction_);

      // * update step size
      step++;
      if (step > 100000) {
        MLOG_INFO(
            "search step has passed 100000 times, but still not get optimal "
            "result!"
            << "gradients_second_order_norm_ = "
            << gradients_second_order_norm_)
        return;
      }
    }
  }

 private:
  template <typename Callable>
  void Init(Callable func) {
    // * calculate res vector
    v_vector_.noalias() = -1 * gradients_;
    // *set first conjugate vector as res vector
    u_vector_.noalias() = v_vector_;

    search_direction_ = EigenVector::Zero(N);

    return;
  }

  void UpdateAlhpa() {
    v_k_second_norm_ = v_vector_.transpose().dot(v_vector_);
    T u_K_A_second_norm = u_vector_.transpose() * A_matrix_ * u_vector_;
    alpha_ = v_k_second_norm_ / u_K_A_second_norm;
    return;
  }

  void UpdateIterate(const EigenVector &d_vector) {
    search_results_.noalias() += step_ * d_vector;
  }

  void UpdateResVector() {
    v_vector_.noalias() = v_vector_ - alpha_ * A_matrix_ * u_vector_;
    return;
  }

  void UpdateConjugateVector() {
    T beta = (v_vector_.transpose().dot(v_vector_)) / v_k_second_norm_;
    u_vector_.noalias() = v_vector_ + beta * u_vector_;
    return;
  }

  void UpdateGradientSecondOrderNorm() {
    gradients_second_order_norm_ = 0.0;
    std::for_each(gradients_.data(), gradients_.data() + gradients_.size(),
                  [this](const auto &gradient) {
                    gradients_second_order_norm_ += gradient * gradient;
                  });
    return;
  }

  template <typename Callable>
  bool SatisifySearchFinishCondition(Callable func) {
    // * update gradients
    gradients_ = MyOptimization::BaseMath::
        GetNumericGrandientForEigenVectorByForwardDifference<Callable, T, N>(
            func, search_results_, epsilon_);
    // * update norm of gradients before search
    UpdateGradientSecondOrderNorm();
    MLOG_ERROR("gradients_ " << gradients_ << ", gradients_second_order_norm_ "
                             << gradients_second_order_norm_);
    return gradients_second_order_norm_ <= iteration_error_;
  }

  void UpdateSearchDirection() {
    search_direction_.noalias() = search_direction_ + alpha_ * u_vector_;
    return;
  }

  template <typename Callable>
  void UpdateSetp(Callable func, EigenVector d_vector) {
    step_ = init_step_;
    LinearSearchForEigenVectorWithArmijoCondition<Callable, T, N>(
        parameter_, step_descresing_coefficient_, step_, search_results_,
        d_vector, func);
    return;
  }

  bool SatisifyUpdateSearchDirectionCondition(const T epsilon_k) {
    T v_vector_second_norm = 0.0;
    std::for_each(v_vector_.data(), v_vector_.data() + v_vector_.size(),
                  [&v_vector_second_norm](const auto &v) {
                    v_vector_second_norm += v * v;
                  });
    T v_vector_second_norm_sqr = std::sqrt(v_vector_second_norm);

    MLOG_ERROR("v_vector_ "
               << v_vector_ << "gradients_ " << gradients_
               << "v_vector_second_norm_sqr " << v_vector_second_norm_sqr
               << ",  epsilon_k * gradients_second_order_norm_ "
               << epsilon_k * std::sqrt(gradients_second_order_norm_));

    return v_vector_second_norm_sqr >
           epsilon_k * std::sqrt(gradients_second_order_norm_);
  }

 private:
  T epsilon_;
  T iteration_error_;
  T step_;
  T init_step_;
  T gradients_second_order_norm_;
  T v_k_second_norm_;
  T parameter_;
  T alpha_;
  T step_descresing_coefficient_;

  EigenMatrix A_matrix_;          // A  matrix
  EigenVector b_vector_;          // gradients vector
  EigenVector v_vector_;          // res vector
  EigenVector u_vector_;          // conjugate vector
  EigenVector gradients_;         // gradients vector
  EigenVector search_direction_;  // conjugate vector

  EigenVector search_results_;
};

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_TRUNCATED_NEWTON_CG_H
