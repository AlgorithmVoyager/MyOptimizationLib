#include <memory>

#include "core/ConvexOptimization/QuasiNetwonMethod/bfgs.h"
#include "logging/log.h"

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

}  // namespace

int main() {
  //   const double iteration_error = 1e-3;
  //   const double x_epsilon = 1e-3;
  //   const double y_epsilon = 1e-3;
  //   Eigen::VectorXd init_pos(2);
  //   init_pos << 1, 10;

  //   std::shared_ptr<MyOptimization::ConvexOptimization::NewtonMethod<double,
  //   2>>
  //       newton_method_ptr = std::make_shared<
  //           MyOptimization::ConvexOptimization::NewtonMethod<double, 2>>(
  //           iteration_error, init_pos);

  //   newton_method_ptr->Step<decltype(TwoDimensionalRosenBlockFunc)>(
  //       x_epsilon, y_epsilon, TwoDimensionalRosenBlockFunc);

  //   auto res = newton_method_ptr->GetSearchResults();

  const double iteration_error = 1e-3;
  const double parameter = 0.1;
  const double step = 10.0;
  Eigen::VectorXd init_pos(2);
  init_pos << 50, 30;

  std::shared_ptr<MyOptimization::ConvexOptimization::BFGS<double, 2>>
      bfgs_ptr =
          std::make_shared<MyOptimization::ConvexOptimization::BFGS<double, 2>>(
              init_pos, iteration_error, parameter, step);

  bfgs_ptr->Step<decltype(TwoDimensionalRosenBlockFunc)>(
      TwoDimensionalRosenBlockFunc);

  auto res = bfgs_ptr->GetSearchResults();
  MLOG_ERROR("res " << res);

  MLOG_ERROR("res " << res);
}