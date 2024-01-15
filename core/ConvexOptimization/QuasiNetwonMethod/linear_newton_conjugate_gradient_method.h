#ifndef CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_LINEAR_NEWTON_CONJUGATE_GRADIENT_METHOD_H
#define CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_LINEAR_NEWTON_CONJUGATE_GRADIENT_METHOD_H

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vector>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/StepSearch/wolfe_condition.h"
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
class LinearNewtonConjugateGradientMethod {
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;

 public:
  LinearNewtonConjugateGradientMethod() = delete;
  ~LinearNewtonConjugateGradientMethod() {}

  LinearNewtonConjugateGradientMethod(const EigenVector &init_pos,
                                      const EigenVector &b,
                                      const EigenMatrix &A, T iteration_error,
                                      T epsilon = 1e-3)
      : epsilon_(epsilon),
        iteration_error_(iteration_error),
        search_results_(init_pos) {
    b_vector_.noalias() = b;
    A_matrix_.noalias() = A;
  }

  const EigenVector GetSearchResults() const { return search_results_; }

  template <typename Callable>
  void Step(Callable func) {
    // * init
    Init(func);

    int step = 0;
    // * main while loop
    while (!SatisifySearchFinishCondition()) {
      // * update step
      UpdateStep();

      // * update Iterator
      UpdateIterate(u_vector_);

      // * update res
      UpdateResVector();

      // * update cg beta & update new conjugate direction vector
      UpdateConjugateVector();

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
    v_vector_.noalias() = b_vector_ - A_matrix_ * search_results_;
    // *set first conjugate vector as res vector
    u_vector_.noalias() = v_vector_;

    return;
  }

  void UpdateStep() {
    v_k_second_norm_ = v_vector_.transpose().dot(v_vector_);
    T u_K_A_second_norm = u_vector_.transpose() * A_matrix_ * u_vector_;
    step_ = v_k_second_norm_ / u_K_A_second_norm;
    return;
  }

  void UpdateIterate(const EigenVector &d_vector) {
    search_results_.noalias() += step_ * d_vector;
  }

  void UpdateResVector() {
    v_vector_.noalias() = v_vector_ - step_ * A_matrix_ * u_vector_;
    return;
  }

  void UpdateConjugateVector() {
    T beta = (v_vector_.transpose().dot(v_vector_)) / v_k_second_norm_;
    u_vector_.noalias() = v_vector_ + beta * u_vector_;
    return;
  }

  void UpdateGradientSecondOrderNorm() {
    gradients_second_order_norm_ = 0.0;
    std::for_each(
        v_vector_.data(), v_vector_.data() + v_vector_.size(),
        [this](const auto &v) { gradients_second_order_norm_ += v * v; });
    return;
  }

  bool SatisifySearchFinishCondition() {
    // * update norm of gradients before search
    UpdateGradientSecondOrderNorm();
    return gradients_second_order_norm_ <= iteration_error_;
  }

 private:
  T epsilon_;
  T iteration_error_;
  T step_;
  T gradients_second_order_norm_;
  T v_k_second_norm_;

  EigenMatrix A_matrix_;  // A matrix
  EigenVector b_vector_;  // b vector
  EigenVector v_vector_;  // res vector
  EigenVector u_vector_;  // conjugate vector

  EigenVector search_results_;
};

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_LINEAR_NEWTON_CONJUGATE_GRADIENT_METHOD_H
