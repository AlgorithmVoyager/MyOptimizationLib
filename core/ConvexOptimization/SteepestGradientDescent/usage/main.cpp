#include "core/ConvexOptimization/SteepestGradientDescent/steepest_gradient_descent.h"
#include "logging/log.h"
#include <memory>

namespace {
double TwoDimensionalRosenBlockFunc(double x1, double x2) {
  return 100 * (x1 * x1 - x2) * (x1 * x1 - x2) + (x1 - 1) * (x1 - 1);
}

double TwoDiemensionalRosenBlockGradientFirstDimensionalFunc(double x1,
                                                             double x2) {
  return 400 * (x1 * x1 - x2) * x1 + 2 * (x1 - 1);
}

double TwoDiemensionalRosenBlockGradientSecondDimensionalFunc(double x1,
                                                              double x2) {
  return -200 * (x1 * x1 - x2);
}

} // namespace

int main() {
  const double step{1.0};
  const double parameter{0.25};
  const double iteration_error{1e-5};
  std::array<double, 2> init_pos{100, 100};
  // std::tuple<decltype(TwoDiemensionalRosenBlockGradientFirstDimensionalFunc),
  //            decltype(TwoDiemensionalRosenBlockGradientSecondDimensionalFunc)>
  //     gradients_functions{
  //         TwoDiemensionalRosenBlockGradientFirstDimensionalFunc,
  //         TwoDiemensionalRosenBlockGradientSecondDimensionalFunc};

  std::shared_ptr<
      MyOptimization::ConvexOptimization::SteepestGradientDescent<double, 2>>
      sgd_ptr_ = std::make_shared<MyOptimization::ConvexOptimization::
                                      SteepestGradientDescent<double, 2>>(
          step, parameter, iteration_error, init_pos);
  sgd_ptr_
      ->Step<decltype(TwoDimensionalRosenBlockFunc),
             decltype(TwoDiemensionalRosenBlockGradientFirstDimensionalFunc),
             decltype(TwoDiemensionalRosenBlockGradientSecondDimensionalFunc)>(
          TwoDimensionalRosenBlockFunc,
          TwoDiemensionalRosenBlockGradientFirstDimensionalFunc,
          TwoDiemensionalRosenBlockGradientSecondDimensionalFunc);
  auto res_pos = sgd_ptr_->GetSearchResult();

  for (auto res : res_pos) {
    MLOG_INFO("res :" << res);
  }
  return 0;
}
