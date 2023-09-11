#include <glog/logging.h>
#include <stdexcept>
#include <array>
#include <algorithm>
#include <numeric>

namespace MyOptimization
{
    namespace ConvexOptimization
    {
        namespace utils
        {
            template <typename Function>
            void ValueFunctionCheck(const Function &value_function);

            template <typename Function, typename GradientFunction>
            void ValueAndGradientFunctionCheck(const Function &value_function,
                                               const GradientFunction &gradient_fucntion);

            template <typename Container, size_t N>
            void ContainerSizeCheck(const Container &container);

            template <typename Function, typename... Args, size_t N>
            void ArgsCheck(const Function &value_function, const Args &...args);

            template <typename T, size_t N>
            std::array<T, N> GetNextStepInputValue(const std::array<T, N> &init_input_value, const std::array<T, N> &gradients, const float step);

            template <typename T, size_t N>
            T GetDotProductOfTwoArray(const std::array<T, N> &lhs, const std::array<T, N> &rhs);

            template <typename T, typename Lambda, size_t N>
            void GetArrayMultiLambda(std::array<T, N> &init_input_value, const Lambda &lambda);

            template <typename T, typename Lambda, size_t N>
            std::array<T, N> GetArrayMultiLambda(std::array<T, N> &init_input_value, const Lambda &lambda);

            template <typename T, typename Lambda, size_t N>
            void AddArray(const std::array<T, N> &increment_input_value, std::array<T, N> &input_value);

        } // namespace utils
    }     // namespace ConvexOptimization

} // namespace MyOptimization