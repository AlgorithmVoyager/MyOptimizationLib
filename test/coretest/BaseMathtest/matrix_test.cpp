#include "core/BaseMath/matrix.h"

#include <gtest/gtest.h>

namespace MyOptimization {
namespace BaseMath {
namespace {

TEST(FUNCTIONTEST, ForwardSubstitutionSolveLowerTriangleEquations) {
  /**
   * 1 0 0 1
   * 2 1 0 3
   * 3 1 1 5
   */

  std::array<std::array<float, 3>, 3> l_matrix;
  std::array<float, 3> b;
  for (auto i = 0; i < 3; i++) {
    l_matrix[i][i] = 1;
  }
  l_matrix[1][0] = 2;
  l_matrix[2][0] = 3;
  l_matrix[2][1] = 1;

  for (auto j = 0; j < 3; j++) {
    b[j] = 1 + 2 * j;
  }

  auto res =
      ForwardSubstitutionSolveLowerTriangleEquations<float, 3>(l_matrix, b);
  const float expected_res{1.0};
  for (auto i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(res[i], expected_res);
  }

  /**
   * 3 0 0 0 9
   * 2 4 0 0 8
   * 1 -2 5 0 5
   * 4 1 -3 6 27
   */
  std::array<std::array<float, 4>, 4> l_matrix_four;
  std::array<float, 4> b_four;

  l_matrix_four[0][0] = 3;
  l_matrix_four[1][0] = 2;
  l_matrix_four[1][1] = 4;
  l_matrix_four[2][0] = 1;
  l_matrix_four[2][1] = -2;
  l_matrix_four[2][2] = 5;
  l_matrix_four[3][0] = 4;
  l_matrix_four[3][1] = 1;
  l_matrix_four[3][2] = -3;
  l_matrix_four[3][3] = 6;

  b_four[0] = 9;
  b_four[1] = 8;
  b_four[2] = 5;
  b_four[3] = 27;

  auto res_four = ForwardSubstitutionSolveLowerTriangleEquations<float, 4>(
      l_matrix_four, b_four);
  const std::array<float, 4> expected_results{3.0, 0.5, 0.6, 16.3 / 6};
  for (auto i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(res_four[i], expected_results[i]);
  }
}

TEST(FUNCTIONTEST, ForwardSubstitutionSolveUpperTriangleEquations) {
  /**
   * 1 2 3 6
   * 0 1 2 3
   * 0 0 1 1
   */

  std::array<std::array<float, 3>, 3> l_matrix;
  std::array<float, 3> b;
  for (auto i = 0; i < 3; i++) {
    l_matrix[i][i] = 1;
  }
  l_matrix[0][1] = 2;
  l_matrix[0][2] = 3;
  l_matrix[1][2] = 2;

  b[0] = 6;
  b[1] = 3;
  b[2] = 1;

  auto res =
      ForwardSubstitutionSolveUpperTriangleEquations<float, 3>(l_matrix, b);
  const float expected_res{1.0};
  for (auto i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(res[i], expected_res);
  }

  /**
   * 6 -3 1 4 27
   * 0 5 -2 1 5
   * 0 0 4 2 8
   * 0 0 0 3 9
   */
  std::array<std::array<float, 4>, 4> l_matrix_four;
  std::array<float, 4> b_four;

  l_matrix_four[3][3] = 3;
  l_matrix_four[2][3] = 2;
  l_matrix_four[2][2] = 4;
  l_matrix_four[1][3] = 1;
  l_matrix_four[1][2] = -2;
  l_matrix_four[1][1] = 5;
  l_matrix_four[0][3] = 4;
  l_matrix_four[0][2] = 1;
  l_matrix_four[0][1] = -3;
  l_matrix_four[0][0] = 6;

  b_four[3] = 9;
  b_four[2] = 8;
  b_four[1] = 5;
  b_four[0] = 27;

  auto res_four = ForwardSubstitutionSolveUpperTriangleEquations<float, 4>(
      l_matrix_four, b_four);
  const std::array<float, 4> expected_results{16.3 / 6, 0.6, 0.5, 3.0};
  for (auto i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(res_four[i], expected_results[i]);
  }
}

}  // namespace
}  // namespace BaseMath
}  // namespace MyOptimization
