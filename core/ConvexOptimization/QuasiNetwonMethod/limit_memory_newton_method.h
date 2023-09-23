#ifndef CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_LIMIT_MEMORY_NEWTON_METHOD_H
#define CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_LIMIT_MEMORY_NEWTON_METHOD_H

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vector>

#include "core/BaseMath/function.h"
#include "core/ConvexOptimization/StepSearch/wolfe_condition.h"
#include "logging/log.h"
namespace MyOptimization {
namespace ConvexOptimization {
template <typename T, size_t N, size_t M>
class LimitMemoryCautiousBFGSWithWolfe {
  using EigenMatrix = Eigen::Matrix<T, N, N>;
  using EigenVector = Eigen::Matrix<T, N, 1>;

 public:
  LimitMemoryCautiousBFGSWithWolfe() = delete;
  ~LimitMemoryCautiousBFGSWithWolfe() {}

  LimitMemoryCautiousBFGSWithWolfe(const EigenVector& init_pos,
                                   T iteration_error, T step,
                                   T step_coefficient = 0.5, T c1 = 1e-4,
                                   T c2 = 0.9, T epsilon = 1e-3,
                                   bool weak_wolfe = true)
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

    // * main loop
    int iter_step = 0;

    while (!SatisifySearchFinishCondition()) {
      // * update search direction as negative gradient direction
      search_direction_ = -1 * GetSearchDirectionByTwoLoopAlgorithm();

      // * reset step & linear search step
      step_ = init_step_;
      if (weak_wolfe_) {
        LinearSearchForEigenVectorWithWeakWolfeCondition<Callable, T, N>(
            search_results_, search_direction_, step_, func,
            step_descresing_coefficient_, c1_, c2_);
      } else {
        LinearSearchForEigenVectorWithStrongWolfeCondition<Callable, T, N>(
            search_results_, search_direction_, step_, func,
            step_descresing_coefficient_, c1_, c2_);
      }
      MLOG_INFO("current linear search gotten step " << step_);

      // * update iterate
      EigenVector last_x;
      last_x.noalias() = search_results_;
      UpdateIterate(search_direction_);
      MLOG_ERROR("=====search_results_======" << search_results_);

      // * update gradients in k+1 step
      EigenVector last_gradients;
      last_gradients.noalias() = gradients_;
      gradients_.noalias() = MyOptimization::BaseMath::
          GetNumericGrandientForEigenVectorByForwardDifference<Callable, T, N>(
              func, search_results_, epsilon_);
      MLOG_ERROR("=====gradients_======" << gradients_);

      // * update b matrix in k+1 Step
      EigenVector delta_x;
      delta_x.noalias() = search_results_ - last_x;
      EigenVector delta_g;
      delta_g.noalias() = gradients_ - last_gradients;

      // * update std::vector<T> delta_rou_vector_;
      delta_x_vector_.emplace_back(delta_x);
      delta_g_vector_.emplace_back(delta_g);
      delta_rou_vector_.emplace_back(1 / delta_g.transpose().dot(delta_x));
      MLOG_ERROR("=====delta_g======" << delta_g << ", delta_x = " << delta_x
                                      << ", delta rou ="
                                      << delta_rou_vector_.back());
      MLOG_ERROR("=====delta_x_vector_ size ======"
                 << delta_x_vector_.size()
                 << ", delta_g_vector_ size  = " << delta_g_vector_.size()
                 << ", delta_rou_vector_ " << delta_rou_vector_.size());

      // * deleta k-m step result if size > M
      DeleteLastHistoryInfo();

      iter_step++;
      if (iter_step > 100000) {
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
    // * init b matrix and gradients
    gradients_.noalias() = MyOptimization::BaseMath::
        GetNumericGrandientForEigenVectorByForwardDifference<Callable, T, N>(
            func, search_results_, epsilon_);
  }

  void UpdateIterate(const EigenVector& d_vector) {
    search_results_.noalias() += step_ * d_vector;
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
    // * update norm of gradients before search
    UpdateGradientSecondOrderNorm();
    return gradients_second_order_norm_ <= iteration_error_;
  }

  EigenVector GetSearchDirectionByTwoLoopAlgorithm() {
    // * input is gradient
    EigenVector d;
    d.noalias() = gradients_;

    if (delta_x_vector_.empty() || delta_g_vector_.empty() ||
        delta_rou_vector_.empty()) {
      return d;
    }

    // * check if same size
    if (delta_x_vector_.size() != delta_g_vector_.size() ||
        delta_rou_vector_.size() != delta_g_vector_.size()) {
      MLOG_ERROR("history info vector size not same!");
      return d;
    }

    // * first for loop: update alpha stack and search direction i=
    // [k-1,...,k-m]
    alpha_vector_.resize(delta_x_vector_.size());
    for (int i = delta_x_vector_.size() - 1; i >= 0; --i) {
      alpha_vector_[i] =
          delta_rou_vector_[i] * delta_x_vector_[i].transpose().dot(d);
      // MLOG_ERROR("===== alpha_vector_[i]======" << alpha_vector_[i]);
      d -= alpha_vector_[i] * delta_g_vector_[i];
      // MLOG_ERROR("===== d======" << d);
    }

    // * update search direction
    T gamma = delta_rou_vector_.back() *
              delta_g_vector_.back().transpose().dot(delta_g_vector_.back());
    // MLOG_ERROR("===== gamma======" << gamma);

    d /= gamma;
    // MLOG_ERROR("===== d======" << d);

    for (auto i = 0; i < delta_g_vector_.size(); ++i) {
      auto beta = delta_rou_vector_[i] * delta_g_vector_[i].transpose().dot(d);
      d += delta_x_vector_[i] * (alpha_vector_[i] - beta);
    }

    return d;
  }

  void DeleteLastHistoryInfo() {
    if (delta_x_vector_.size() >= M) {
      delta_x_vector_.erase(delta_x_vector_.begin());
    }
    if (delta_g_vector_.size() >= M) {
      delta_g_vector_.erase(delta_g_vector_.begin());
    }
    if (delta_rou_vector_.size() >= M) {
      delta_rou_vector_.erase(delta_rou_vector_.begin());
    }
    return;
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

  EigenVector gradients_;
  EigenVector search_results_;
  EigenVector search_direction_;

  std::vector<T> alpha_vector_;
  std::vector<T> delta_rou_vector_;
  std::vector<EigenVector> delta_x_vector_;
  std::vector<EigenVector> delta_g_vector_;
};

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_LIMIT_MEMORY_NEWTON_METHOD_H
