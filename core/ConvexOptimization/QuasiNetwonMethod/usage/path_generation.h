#ifndef CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_USAGE_PATH_GENERATION_H
#define CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_USAGE_PATH_GENERATION_H

#include <eigen3/Eigen/Dense>
#include <memory>
#include <random>
#include <vector>

#include "core/ConvexOptimization/QuasiNetwonMethod/lbfgs_with_lewis_and_overton.h"
#include "logging/log.h"

namespace MyOptimization {
namespace ConvexOptimization {

template <typename T>
struct Obstacle {
  T obs_x;
  T obs_y;
  T radius;

  Obstacle() = default;

  Obstacle(T obs_x, T obs_y, T radius)
      : obs_x(obs_x), obs_y(obs_y), radius(radius) {}
};

template <typename T>
struct Path {
  std::vector<T> xs;
  std::vector<T> ys;

  Path() = default;
};

template <typename T>
std::vector<Obstacle<T>> CreateObstacle(const T lower_bound,
                                        const T upper_bound, size_t N) {
  std::random_device rd;
  std::mt19937 generator(rd());

  std::uniform_real_distribution<T> distribution(lower_bound, upper_bound);
  std::vector<Obstacle<T>> res;
  for (auto i = 0; i < N; i++) {
    Obstacle<T> obs;
    obs.obs_x = static_cast<T>(distribution(generator));
    obs.obs_y = static_cast<T>(distribution(generator));
    obs.radius = static_cast<T>(1.0);
    res.emplace_back(obs);
  }
  return res;
}

// ! TODO: add D coeff here
template <typename T, size_t N>
T getEnergyPart(const Eigen::Matrix<T, 8 * N, 1>& coefficients) {
  // * generate small matrix
  constexpr int one_segment_coefficients_size = 4;

  if (coefficients.size() % one_segment_coefficients_size != 0) {
    MLOG_ERROR("input coefficients' size is not times of four");
    return 0.0;
  }

  Eigen::Matrix<T, one_segment_coefficients_size, one_segment_coefficients_size>
      one_segment_coeff;
  one_segment_coeff << 1, 1 / 2, 1 / 3, 1 / 4, 1 / 2, 1 / 3, 1 / 4, 1 / 5,
      1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 4, 1 / 5, 1 / 6, 1 / 7;

  Eigen::Matrix<T, 8 * N, 8 * N> energy_function_matrix;
  for (auto i = 0; i < 2 * N; ++i) {
    energy_function_matrix.block(
        i * one_segment_coefficients_size, i * one_segment_coefficients_size,
        one_segment_coefficients_size, one_segment_coefficients_size) =
        one_segment_coeff;
  }

  T energy_res =
      coefficients.transpose() * energy_function_matrix * coefficients;
  return energy_res;
};

template <typename T, size_t N>
T getPotentialPart(const Eigen::Matrix<T, 8 * N, 1>& coefficients,
                   const std::vector<Obstacle<T>> obstacles) {
  T potential_part{0.0};

  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < 8; ++j) {
      // * generate current position

      auto first_point_x = coefficients.coeff(8 * i);
      auto first_point_y = coefficients.coeff(8 * i + 3);

      auto second_point_x =
          coefficients.coeff(8 * i) + coefficients.coeff(8 * i + 1) +
          coefficients.coeff(8 * i + 2) + coefficients.coeff(8 * i + 3);
      auto second_point_y =
          coefficients.coeff(8 * i + 4) + coefficients.coeff(8 * i + 5) +
          coefficients.coeff(8 * i + 6) + coefficients.coeff(8 * i + 7);
      // * generate current Obstacle
      for (auto i = 0; i < obstacles.size(); ++i) {
        auto obstacle = obstacles[i];
        auto square_dis_first_point =
            (obstacle.obs_x - first_point_x) *
                (obstacle.obs_x - first_point_x) +
            (obstacle.obs_y - first_point_y) * (obstacle.obs_y - first_point_y);
        auto dis_to_obstacle_first = std::sqrt(square_dis_first_point);

        auto square_dis_second_point = (obstacle.obs_x - second_point_x) *
                                           (obstacle.obs_x - second_point_x) +
                                       (obstacle.obs_y - second_point_y) *
                                           (obstacle.obs_y - second_point_y);
        auto dis_to_obstacle_second = std::sqrt(square_dis_first_point);

        potential_part +=
            1000 * std::fmax(obstacle.radius - dis_to_obstacle_first, 0.0);
        potential_part +=
            1000 * std::fmax(obstacle.radius - dis_to_obstacle_second, 0.0);
      }
    }
  }
  return potential_part;
}

template <typename T, size_t N>
T getOptimizationFunction(const Eigen::Matrix<T, 8 * N, 1>& coefficients) {
  T lower_bound = 1.0;
  T upper_bound = 10.0;
  auto obstacles = CreateObstacle<T>(lower_bound, upper_bound, 2);
  return getEnergyPart<T, N>(coefficients) +
         getPotentialPart<T, N>(coefficients, obstacles);
}

// ! TODO: visualize path
template <typename T, size_t N>
class PathOptimization {
 public:
  PathOptimization() = default;
  ~PathOptimization() {}

  void GenerateCoefficient() {
    Eigen::Matrix<T, 8 * N, 1> init_coeff =
        Eigen::Matrix<T, 8 * N, 1>::Ones(8 * N);

    const T iteration_error = 1e-4;
    const T parameter = 0.1;
    const T step = 10000.0;

    std::shared_ptr<LimitMemoryCautiousBFGSWithLewis<T, 8 * N, 6>> bfgs_ptr =
        std::make_shared<LimitMemoryCautiousBFGSWithLewis<T, 8 * N, 6>>(
            init_coeff, iteration_error, step);

    bfgs_ptr->Step(getOptimizationFunction<T, N>);

    coeff_vec_.noalias() = bfgs_ptr->GetSearchResults();
    MLOG_ERROR("coeff_vec_ " << coeff_vec_);
  }

  void GeneratePath(Path<T>& path) {
    path.xs.clear();
    path.ys.clear();
    T step = 0.1;
    for (auto i = 0; i < N; i++) {
      for (i = 0; i < 10; i++) {
        T current_step = step * i;
        T x = coeff_vec_.coeff(8 * i) +
              coeff_vec_.coeff(8 * i + 1) * current_step +
              coeff_vec_.coeff(8 * i + 2) * current_step * current_step +
              coeff_vec_.coeff(8 * i + 3) * current_step * current_step *
                  current_step;
        T y = coeff_vec_.coeff(8 * i + 4) +
              coeff_vec_.coeff(8 * i + 5) * current_step +
              coeff_vec_.coeff(8 * i + 6) * current_step * current_step +
              coeff_vec_.coeff(8 * i + 7) * current_step * current_step *
                  current_step;
        path.xs.emplace_back(x);
        path.ys.emplace_back(y);
      }
    }
    return;
  }

 private:
  Eigen::Matrix<T, 8 * N, 1> coeff_vec_;
};

}  // namespace ConvexOptimization
}  // namespace MyOptimization

#endif  // CORE_CONVEXOPTIMIZATION_QUASINETWONMETHOD_USAGE_PATH_GENERATION_H
