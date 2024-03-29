#include "core/BaseMath/function.h"

#include <gtest/gtest.h>

#include <eigen3/Eigen/Dense>

namespace MyOptimization {
namespace BaseMath {
namespace {

inline int normalAddOneFunction(int i) { return i + 1; }
inline int normalMultiAddFunction(int x, int y, int z) { return x + y + z; }
inline auto normalMultiTypeAddFunction(int x, float y, double z)
    -> decltype(x + y + z) {
  return x + y + z;
}

template <typename T, size_t N>
double GetEigenRes(Eigen::Matrix<T, N, 1> a) {
  double res = 0.0;
  for (auto it = 0; it < a.size(); ++it) {
    res += a.coeff(it);
  }
  return res;
}

template <typename T>
auto normalMultiTypeAddFunctiontem(T x, T y, T z) -> decltype(x + y + z) {
  return x + y + z;
}

float highOrderFunctionForThreeVariableFunc(const float x, const float y,
                                            const float z) {
  return x * x + 2 * y * y * y + 1 / z;
}

double highOrderFunctionForThreeVariableFunc2(const double x, const double y,
                                              const double z) {
  return x * x + 2 * x * y + y / z;
}

TEST(FUNCTIONTEST, FunctionInput) {
  auto lambda_func = [](int i) -> decltype(i + 1) { return i + 1; };
  // lambda_func normal anonymous, so decltype(lambda) will lead to a Error
  std::function<int(int)> std_function_of_lambda = lambda_func;
  std::function<decltype(normalAddOneFunction)> std_function_of_normal_func =
      normalAddOneFunction;

  auto lambda_res =
      MyOptimization::BaseMath::GetValue<int(int), int>(lambda_func, 1);
  auto normal_function_res = MyOptimization::BaseMath::GetValue<int(int), int>(
      normalAddOneFunction, 1);
  auto std_function_of_lambda_res =
      MyOptimization::BaseMath::GetValue<decltype(std_function_of_lambda), int>(
          std_function_of_lambda, 1);
  auto std_function_of_normal_func_res =
      MyOptimization::BaseMath::GetValue<decltype(std_function_of_normal_func),
                                         int>(std_function_of_normal_func, 1);

  const int expected_res = 2;
  EXPECT_EQ(lambda_res, expected_res);
  EXPECT_EQ(normal_function_res, expected_res);
  EXPECT_EQ(std_function_of_lambda_res, expected_res);
  EXPECT_EQ(std_function_of_normal_func_res, expected_res);
}

TEST(FUNCTIONTEST, MultiFunctionInput) {
  auto lambda_func = [](int i, int j) { return i + j; };
  auto std_function_of_lambda = lambda_func;
  auto std_function_of_normal_func = normalMultiTypeAddFunction;

  auto lambda_res =
      MyOptimization::BaseMath::GetFuncValueForMultiInput<int(int, int), int>(
          lambda_func, 1, 1);
  auto normal_multi_add_function_res =
      MyOptimization::BaseMath::GetFuncValueForMultiInput(
          normalMultiAddFunction, 1, 1, 1);
  auto normal_multi_type_add_function_res =
      MyOptimization::BaseMath::GetFuncValueForMultiInput(
          normalMultiTypeAddFunction, 1, 1.0f, 2.0);
  auto std_function_of_lambda_res =
      MyOptimization::BaseMath::GetFuncValueForMultiInput<
          decltype(std_function_of_lambda), int>(std_function_of_lambda, 1, 2);
  auto std_function_of_normal_func_res =
      MyOptimization::BaseMath::GetFuncValueForMultiInput(
          std_function_of_normal_func, 1, 1, 2.0f);

  const int lambda_expected_res = 2;
  const int normal_multi_add_function_expected_res = 3;
  const auto normal_multi_type_add_function_expected_res = 4.0;

  EXPECT_EQ(lambda_res, lambda_expected_res);
  EXPECT_EQ(normal_multi_add_function_res,
            normal_multi_add_function_expected_res);
  EXPECT_EQ(normal_multi_type_add_function_res,
            normal_multi_type_add_function_expected_res);
  EXPECT_EQ(std_function_of_lambda_res, normal_multi_add_function_expected_res);
  EXPECT_EQ(std_function_of_normal_func_res,
            normal_multi_type_add_function_expected_res);
}

TEST(FUNCTIONTEST, ContainerCovert) {
  std::vector<int> lambda_input{2, 3, 4, 5, 6};

  auto tuple =
      CreateTupleFromContainer<decltype(lambda_input), 5>(lambda_input);
  auto tuple_r = CreateTupleFromContainer<decltype(lambda_input), 5>(
      std::vector<int>{2, 3, 4, 5, 6});
  EXPECT_EQ(std::get<0>(tuple), lambda_input[0]);
  EXPECT_EQ(std::get<1>(tuple), lambda_input[1]);
  EXPECT_EQ(std::get<2>(tuple), lambda_input[2]);
  EXPECT_EQ(std::get<3>(tuple), lambda_input[3]);
  EXPECT_EQ(std::get<4>(tuple), lambda_input[4]);
  EXPECT_EQ(std::get<0>(tuple_r), lambda_input[0]);
  EXPECT_EQ(std::get<1>(tuple_r), lambda_input[1]);
  EXPECT_EQ(std::get<2>(tuple_r), lambda_input[2]);
  EXPECT_EQ(std::get<3>(tuple_r), lambda_input[3]);
  EXPECT_EQ(std::get<4>(tuple_r), lambda_input[4]);

  std::array<int, 5> array_intput{2, 3, 4, 5, 6};
  auto tuple_by_array = CreateTupleFromArray(array_intput);
  EXPECT_EQ(std::get<0>(tuple_by_array), array_intput[0]);
  EXPECT_EQ(std::get<1>(tuple_by_array), array_intput[1]);
  EXPECT_EQ(std::get<2>(tuple_by_array), array_intput[2]);
  EXPECT_EQ(std::get<3>(tuple_by_array), array_intput[3]);
  EXPECT_EQ(std::get<4>(tuple_by_array), array_intput[4]);
  ;
  auto tuple_by_array_r =
      CreateTupleFromArray(std::array<int, 5>{2, 3, 4, 5, 6});
  EXPECT_EQ(std::get<0>(tuple_by_array_r), array_intput[0]);
  EXPECT_EQ(std::get<1>(tuple_by_array_r), array_intput[1]);
  EXPECT_EQ(std::get<2>(tuple_by_array_r), array_intput[2]);
  EXPECT_EQ(std::get<3>(tuple_by_array_r), array_intput[3]);
  EXPECT_EQ(std::get<4>(tuple_by_array_r), array_intput[4]);
}

TEST(FUNCTIONTEST, ContainerInput) {
  auto lambda_func = [](const int i, int j) { return i + j; };
  std::function<int(int, int)> std_function_for_lambda = lambda_func;
  std::function<decltype(normalMultiTypeAddFunction)>
      std_function_for_normal_func = normalMultiTypeAddFunction;

  std::vector<int> lambda_input{2, 3};

  auto lambda_res_r =
      MyOptimization::BaseMath::GetFuncValueForContainer<decltype(lambda_func),
                                                         std::vector<int>, 2>(
          lambda_func, std::vector<int>{2, 3});
  auto lambda_res_l = MyOptimization::BaseMath::GetFuncValueForContainer<
      decltype(lambda_func), decltype(lambda_input), 2>(lambda_func,
                                                        lambda_input);
  auto std_function_for_lambda_res =
      MyOptimization::BaseMath::GetFuncValueForContainer<
          decltype(std_function_for_lambda), decltype(lambda_input), 2>(
          std_function_for_lambda, lambda_input);

  const int lambda_expected_res = 5.0;
  EXPECT_EQ(lambda_res_r, lambda_expected_res);
  EXPECT_EQ(lambda_res_l, lambda_expected_res);
  EXPECT_EQ(std_function_for_lambda_res, lambda_expected_res);

  std::vector<int> normal_input{2, 3, 4};
  auto normal_function_res = MyOptimization::BaseMath::GetFuncValueForContainer<
      decltype(normalMultiTypeAddFunction), std::vector<int>, 3>(
      normalMultiTypeAddFunction, normal_input);
  auto std_function_for_normal_func_res =
      MyOptimization::BaseMath::GetFuncValueForContainer<
          decltype(std_function_for_normal_func), std::vector<int>, 3>(
          std_function_for_normal_func, normal_input);
  const int normal_function_expected_res = 9.0;
  EXPECT_EQ(normal_function_res, normal_function_expected_res);
  EXPECT_EQ(std_function_for_normal_func_res, normal_function_expected_res);
}

TEST(FUNCTIONTEST, ArrayInput) {
  auto lambda_func = [](const int i, int j) { return i + j; };
  std::function<int(int, int)> std_function_lambda = lambda_func;
  std::array<int, 2> lambda_input{2, 3};

  auto lambda_res =
      MyOptimization::BaseMath::GetFuncValueForArray(lambda_func, lambda_input);
  auto std_function_lambda_res = MyOptimization::BaseMath::GetFuncValueForArray(
      std_function_lambda, lambda_input);

  const int lambda_expected_res = 5.0;
  EXPECT_EQ(lambda_res, lambda_expected_res);
  EXPECT_EQ(std_function_lambda_res, lambda_expected_res);

  std::array<float, 3> normal_input{2.f, 3.f, 6.f};
  auto normal_function_res = MyOptimization::BaseMath::GetFuncValueForArray(
      normalMultiTypeAddFunctiontem<float>, normal_input);
  const int normal_function_expected_res = 11.0f;
  EXPECT_EQ(normal_function_res, normal_function_expected_res);
}

TEST(FUNCTIONTEST, ContainerInputGradientForwardStep) {
  auto lambda_func = [](const float i, float j) { return 2 * i + 3 * j; };
  std::vector<float> lambda_input{2.0, 3.0};
  const double epsilon = 1e-3;
  const double tolerance_error = 1e-1;

  auto lambda_res = MyOptimization::BaseMath::
      GetNumericnGrandientForContainerByForwardDifference<
          decltype(lambda_func), decltype(lambda_input), float, 2>(
          lambda_func, lambda_input, epsilon);

  const std::vector<float> expected_lambda_gradients_output{2.0f, 3.0f};
  for (auto i = 0U; i < lambda_res.size(); ++i) {
    EXPECT_LT(std::abs(lambda_res[i] - expected_lambda_gradients_output[i]),
              tolerance_error);
  }

  // highOrderFunctionForThreeVariableFunc : x * x + 2 * y * y * y + 1 / z;
  std::vector<float> func_input{2.0, 3.0, 4.0};
  auto func_res = MyOptimization::BaseMath::
      GetNumericnGrandientForContainerByForwardDifference<
          decltype(highOrderFunctionForThreeVariableFunc), decltype(func_input),
          float, 3>(highOrderFunctionForThreeVariableFunc, func_input, epsilon);
  const std::vector<float> expected_func_gradients_output{4.0f, 54.0f,
                                                          -(1 / 16)};
  for (auto i = 0U; i < func_res.size(); ++i) {
    EXPECT_LT(std::abs(func_res[i] - expected_func_gradients_output[i]),
              tolerance_error);
  }
}

TEST(FUNCTIONTEST, ArrayInputGradientForwardStep) {
  auto lambda_func = [](const double i, double j) { return 2 * i + 3 * j; };
  std::array<double, 2> lambda_input{2.0, 3.0};
  const double epsilon = 1e-4;
  const double tolerance_error = 2 * 1e-1;

  auto lambda_res =
      MyOptimization::BaseMath::GetNumericGrandientForArrayByForwardDifference<
          decltype(lambda_func), double, 2>(lambda_func, lambda_input, epsilon);

  const std::vector<double> expected_lambda_gradients_output{2.0f, 3.0f};
  for (auto i = 0U; i < lambda_res.size(); ++i) {
    EXPECT_LT(std::abs(lambda_res[i] - expected_lambda_gradients_output[i]),
              tolerance_error);
  }

  // highOrderFunctionForThreeVariableFunc2 : x * x + 2 * x * y + y / z;
  std::array<double, 3> func_input{2.0, 3.0, 4.0};
  auto func_res =
      MyOptimization::BaseMath::GetNumericGrandientForArrayByForwardDifference<
          decltype(highOrderFunctionForThreeVariableFunc2), double, 3>(
          highOrderFunctionForThreeVariableFunc2, func_input, epsilon);
  const std::vector<double> expected_func_gradients_output{10.0f, 4.25f,
                                                           -(3 / 16)};
  for (auto i = 0U; i < func_res.size(); ++i) {
    EXPECT_LT(std::abs(func_res[i] - expected_func_gradients_output[i]),
              tolerance_error);
  }
}

TEST(FUNCTIONTEST, ContainerInputGradientCentralStep) {
  auto lambda_func = [](const float i, float j) { return 2 * i + 3 * j; };
  std::vector<float> lambda_input{2.0, 3.0};
  const double epsilon = 1e-3;
  const double tolerance_error = 1e-1;

  auto lambda_res = MyOptimization::BaseMath::
      GetNumericnGrandientForContainerByCenteralDifference<
          decltype(lambda_func), decltype(lambda_input), float, 2>(
          lambda_func, lambda_input, epsilon);

  const std::vector<float> expected_lambda_gradients_output{2.0f, 3.0f};
  for (auto i = 0U; i < lambda_res.size(); ++i) {
    EXPECT_LT(std::abs(lambda_res[i] - expected_lambda_gradients_output[i]),
              tolerance_error);
  }

  // highOrderFunctionForThreeVariableFunc : x * x + 2 * y * y * y + 1 / z;
  std::vector<float> func_input{2.0, 3.0, 4.0};
  auto func_res = MyOptimization::BaseMath::
      GetNumericnGrandientForContainerByCenteralDifference<
          decltype(highOrderFunctionForThreeVariableFunc), decltype(func_input),
          float, 3>(highOrderFunctionForThreeVariableFunc, func_input, epsilon);
  const std::vector<float> expected_func_gradients_output{4.0f, 54.0f,
                                                          -(1 / 16)};
  for (auto i = 0U; i < func_res.size(); ++i) {
    EXPECT_LT(std::abs(func_res[i] - expected_func_gradients_output[i]),
              tolerance_error);
  }
}

TEST(FUNCTIONTEST, ArrayInputGradientCenteralStep) {
  auto lambda_func = [](const double i, double j) { return 2 * i + 3 * j; };
  std::array<double, 2> lambda_input{2.0, 3.0};
  const double epsilon = 1e-4;
  const double tolerance_error = 2 * 1e-1;

  auto lambda_res =
      MyOptimization::BaseMath::GetNumericGrandientForArrayByCenteralDifference<
          decltype(lambda_func), double, 2>(lambda_func, lambda_input, epsilon);

  const std::vector<double> expected_lambda_gradients_output{2.0f, 3.0f};
  for (auto i = 0U; i < lambda_res.size(); ++i) {
    EXPECT_LT(std::abs(lambda_res[i] - expected_lambda_gradients_output[i]),
              tolerance_error);
  }

  // highOrderFunctionForThreeVariableFunc2 : x * x + 2 * x * y + y / z;
  std::array<double, 3> func_input{2.0, 3.0, 4.0};
  auto func_res =
      MyOptimization::BaseMath::GetNumericGrandientForArrayByCenteralDifference<
          decltype(highOrderFunctionForThreeVariableFunc2), double, 3>(
          highOrderFunctionForThreeVariableFunc2, func_input, epsilon);
  const std::vector<double> expected_func_gradients_output{10.0f, 4.25f,
                                                           -(3 / 16)};
  for (auto i = 0U; i < func_res.size(); ++i) {
    EXPECT_LT(std::abs(func_res[i] - expected_func_gradients_output[i]),
              tolerance_error);
  }
}

TEST(FUNCTIONTEST, StdFunctionArrayInputGradientCenteralStep) {
  std::function<decltype(highOrderFunctionForThreeVariableFunc2)>
      std_high_order_fun = highOrderFunctionForThreeVariableFunc2;
  const float epsilon = 1e-3;
  const double tolerance_error = 2 * 1e-1;
  // highOrderFunctionForThreeVariableFunc2 : x * x + 2 * x * y + y / z;
  std::array<double, 3> func_input{2.0, 3.0, 4.0};
  auto func_res =
      MyOptimization::BaseMath::GetNumericGrandientForArrayByCenteralDifference<
          decltype(std_high_order_fun), double, 3>(std_high_order_fun,
                                                   func_input, epsilon);
  const std::vector<double> expected_func_gradients_output{10.0f, 4.25f,
                                                           -(3 / 16)};
  for (auto i = 0U; i < func_res.size(); ++i) {
    EXPECT_LT(std::abs(func_res[i] - expected_func_gradients_output[i]),
              tolerance_error);
  }
}

TEST(FUNCTIONTEST, GetEigenResTEST) {
  Eigen::Matrix<double, 3, 1> a;
  a << 1.0, 2.0, 3.0;

  auto getRes = [](Eigen::Matrix<double, 3, 1> a) -> double { return a(0); };

  double func_res = MyOptimization::BaseMath::GetFuncValueForEigenVector<
      decltype(GetEigenRes<double, 3>), double, 3>(GetEigenRes<double, 3>, a);

  MLOG_ERROR("func_res == " << func_res);

  double func_res_lambda =
      MyOptimization::BaseMath::GetFuncValueForEigenVector<decltype(getRes),
                                                           double, 3>(getRes,
                                                                      a);

  MLOG_ERROR("func_res_lambda == " << func_res_lambda);

  double func_res_new = MyOptimization::BaseMath::GetFuncValueForEigenVector<
      decltype(normalMultiAddFunction), double, 3>(normalMultiAddFunction, a);
  MLOG_ERROR("func_res_new == " << func_res_new);
}

}  // namespace
}  // namespace BaseMath
}  // namespace MyOptimization
