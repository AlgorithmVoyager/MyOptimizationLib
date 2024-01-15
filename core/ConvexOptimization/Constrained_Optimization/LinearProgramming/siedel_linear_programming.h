#ifndef CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_LINEARPROGRAMMING_SIEDEL_LINEAR_PROGRAMMING_H
#define CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_LINEARPROGRAMMING_SIEDEL_LINEAR_PROGRAMMING_H

#include <eigen3/Eigen/Dense>

/// @brief : //* linear problem is the problem with form:
// *             min f(x) = c_{T} * x
// *             s.t. A*x <= b
// * so, for the input, we mainly need c, A, & b
/// @tparam T
/// @tparam N : dimension of optimization objects
/// @tparam M : dimension of optimization constraints
template <typename T, size_t N, size_t M>
class SiedelLinearProgramming {
 public:
  using ObjectVector = Eigen::Matrix<T, N, 1>;
  using ConstriantsMatrix = Eigen::Matrix<T, M, N>;
  using ConstriantsVector = Eigen::Matrix<T, M, 1>;

  // construction
  SiedelLinearProgramming() = delete;
  ~SiedelLinearProgramming() {}
  SiedelLinearProgramming(const ObjectVector& object_coeff,
                          const ConstriantsMatrix& constraints_matrix,
                          const ConstriantsVector& constraints_vector,
                          const T epsilon = 1e-3)
      : epsilon_(epsilon) {
    object_coeff_.noalias() = object_coeff;
    constraints_matrix_.noalias() = constraints_matrix;
    constraints_vector_.noalias() = constraints_vector;
  };

  // Get interface
  const ObjectVector GetOptimumObjectVector() const { return object_vector_; }
  const T GetOptimumRes() const {
    return object_coeff_.transpose().dot(object_vector_);
  }

  // function main loop interface
  //   void Step() {

  //   }

 private:
  ObjectVector object_vector_;
  ObjectVector object_coeff_;
  ConstriantsMatrix constraints_matrix_;
  ConstriantsVector constraints_vector_;

  T epsilon_;
}

#endif  // CORE_CONVEXOPTIMIZATION_CONSTRAINED_OPTIMIZATION_LINEARPROGRAMMING_SIEDEL_LINEAR_PROGRAMMING_H
