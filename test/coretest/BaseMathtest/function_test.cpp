#include "core/BaseMath/function.h"

#include <gtest/gtest.h>

namespace MyOptimization {
namespace BaseMath {
namespace {

inline int normalAddOneFunction(int i) { return i + 1; }
inline int normalMultiAddFunction(int x, int y, int z) { return x + y + z; }
inline auto normalMultiTypeAddFunction(int x, float y, double z)
    -> decltype(x + y + z) {
  return x + y + z;
}

template <typename T>
auto normalMultiTypeAddFunctiontem(T x, T y, T z) -> decltype(x + y + z) {
  return x + y + z;
}

TEST(FUNCTIONTEST, FunctionInput) {
  auto lambda_func = [](int i) { return i + 1; };
  auto lambda_res =
      MyOptimization::BaseMath::GetValue<int(int), int>(lambda_func, 1);
  auto normal_function_res = MyOptimization::BaseMath::GetValue<int(int), int>(
      normalAddOneFunction, 1);

  const int expected_res = 2;
  EXPECT_EQ(lambda_res, expected_res);
  EXPECT_EQ(normal_function_res, expected_res);
}

TEST(FUNCTIONTEST, MultiFunctionInput) {
  auto lambda_func = [](int i, int j) { return i + j; };

  auto lambda_res =
      MyOptimization::BaseMath::GetFuncValueForMultiInput<int(int, int), int>(
          lambda_func, 1, 1);
  auto normal_multi_add_function_res =
      MyOptimization::BaseMath::GetFuncValueForMultiInput(
          normalMultiAddFunction, 1, 1, 1);
  auto normal_multi_type_add_function_res =
      MyOptimization::BaseMath::GetFuncValueForMultiInput(
          normalMultiTypeAddFunction, 1, 1.0f, 2.0);

  const int lambda_expected_res = 2;
  const int normal_multi_add_function_expected_res = 3;
  const auto normal_multi_type_add_function_expected_res = 4.0;

  EXPECT_EQ(lambda_res, lambda_expected_res);
  EXPECT_EQ(normal_multi_add_function_res,
            normal_multi_add_function_expected_res);
  EXPECT_EQ(normal_multi_type_add_function_res,
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
  std::vector<int> lambda_input{2, 3};

  auto lambda_res_r =
      MyOptimization::BaseMath::GetFuncValueForContainer<decltype(lambda_func),
                                                         std::vector<int>, 2>(
          lambda_func, std::vector<int>{2, 3});
  auto lambda_res_l = MyOptimization::BaseMath::GetFuncValueForContainer<
      decltype(lambda_func), decltype(lambda_input), 2>(lambda_func,
                                                        lambda_input);

  const int lambda_expected_res = 5.0;
  EXPECT_EQ(lambda_res_r, lambda_expected_res);
  EXPECT_EQ(lambda_res_l, lambda_expected_res);

  std::vector<int> normal_input{2, 3, 4};
  auto normal_function_res = MyOptimization::BaseMath::GetFuncValueForContainer<
      decltype(normalMultiTypeAddFunction), std::vector<int>, 3>(
      normalMultiTypeAddFunction, normal_input);
  const int normal_function_expected_res = 9.0;
  EXPECT_EQ(normal_function_res, normal_function_expected_res);
}

TEST(FUNCTIONTEST, ArrayInput) {
  auto lambda_func = [](const int i, int j) { return i + j; };
  std::array<int, 2> lambda_input{2, 3};

  auto lambda_res =
      MyOptimization::BaseMath::GetFuncValueForArray(lambda_func, lambda_input);

  const int lambda_expected_res = 5.0;
  EXPECT_EQ(lambda_res, lambda_expected_res);

  std::array<float, 3> normal_input{2.f, 3.f, 6.f};
  auto normal_function_res = MyOptimization::BaseMath::GetFuncValueForArray(
      normalMultiTypeAddFunctiontem<float>, normal_input);
  const int normal_function_expected_res = 11.0f;
  EXPECT_EQ(normal_function_res, normal_function_expected_res);
}

}  // namespace
}  // namespace BaseMath
}  // namespace MyOptimization
