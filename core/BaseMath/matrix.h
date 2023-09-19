#ifndef CORE_BASEMATH_MATRIX_H
#define CORE_BASEMATH_MATRIX_H
#include <array>
#include <eigen3/Eigen/Core>

#include "logging/log.h"

namespace MyOptimization {
namespace BaseMath {

template <typename T, size_t N>
std::array<T, N> ForwardSubstitutionSolveLowerTriangleEquations(
    const std::array<std::array<T, N>, N>& L, const std::array<T, N>& b) {
  std::array<T, N> res;
  res.fill(static_cast<T>(0.0));

  for (size_t i = 0; i < N; i++) {
    T tem_sum{0.0};
    for (size_t j = 0; j < i; j++) {
      tem_sum += L[i][j] * res[j];
    }

    res[i] = (b[i] - tem_sum) / L[i][i];
  }

  return res;
}

template <typename T, size_t N>
std::array<T, N> ForwardSubstitutionSolveUpperTriangleEquations(
    const std::array<std::array<T, N>, N>& L, const std::array<T, N>& b) {
  std::array<T, N> res;
  res.fill(static_cast<T>(0.0));

  for (int i = N - 1; i >= 0; i--) {
    T tem_sum{0.0};
    for (int j = N - 1; j > i; j--) {
      tem_sum += L[i][j] * res[j];
    }
    res[i] = (b[i] - tem_sum) / L[i][i];
    MLOG_ERROR("b[i] " << b[i] << ", tem_sum " << tem_sum << " L[i][i] "
                       << L[i][i] << "res[i] " << res[i]);
  }

  return res;
}

template <typename EigenMatrix, typename EigenVector>
EigenVector ForwardSubstitutionSolveLowerTriangleEquations(
    const EigenMatrix& eigen_matrix, const EigenVector& eigen_vector) {
  // size check
  if (eigen_matrix.rows() != eigen_vector.size()) {
    try {
      throw std::runtime_error("Matrix row size not same with vector size!!!");
    } catch (const std::exception& e) {
      MLOG_ERROR("[RUNTIME_ERROR]: " << e.what());
    }
  }

  if (eigen_matrix.size() == 0 || eigen_vector.size()) {
    return {};
  }

  EigenVector res;
  res.setZero();

  const size_t N = eigen_vector.size();
  for (size_t i = 0; i < N; i++) {
    decltype(eigen_vector(0)) tem_sum{0.0};
    for (size_t j = 0; j < i; j++) {
      tem_sum += eigen_matrix(i, j) * res(j);
    }

    res[i] = (eigen_vector(i) - tem_sum) / eigen_matrix(i, i);
  }

  return res;
}

// template <typename T, size_t N>
// std::array<T, N> ForwardSubstitutionSolveUpperTriangleEquations(
//     const std::array<std::array<T, N>, N>& L, const std::array<T, N>& b) {
//   std::array<T, N> res;
//   res.fill(static_cast<T>(0.0));

//   for (int i = N - 1; i >= 0; i--) {
//     T tem_sum{0.0};
//     for (int j = N - 1; j > i; j--) {
//       tem_sum += L[i][j] * res[j];
//     }
//     res[i] = (b[i] - tem_sum) / L[i][i];
//     MLOG_ERROR("b[i] " << b[i] << ", tem_sum " << tem_sum << " L[i][i] "
//                        << L[i][i] << "res[i] " << res[i]);
//   }

//   return res;
// }

}  // namespace BaseMath
}  // namespace MyOptimization

#endif  // CORE_BASEMATH_MATRIX_H
