#include "core/ConvexOptimization/SteepestGradientDescent/armijo_condition.h"

namespace MyOptimization
{
    namespace ConvexOptimization
    {
        void ArmijoParameterCheck(const float parameter)
        {
            if (parameter <= 0.0 || parameter >= 1.0)
            {
                try
                {
                    throw std::runtime_error("the param must be between (0,1)");
                }
                catch (const std::exception &e)
                {
                    LOG(ERROR) << "[Runtime Error]:" << e.what();
                }
            }
            return;
        }

        template <typename Func, typename T, size_t N>
        void LinearSearchWithArmijoCondition(
            const Func &func, const std::array<T, N> &current_step_input_value,
            const std::array<T, N> &negative_gradients, const float armijo_parameter,
            float &step)
        {
            static_assert(std::is_function_v<Func>, "Func must be a function");

            const float stop_step = 1e-8;

            // c * t * d^T * Df(x^k), d = - Df(x^k)
            while (GetFuncValueForArray(
                       func,
                       MyOptimization::ConvexOptimization::utils::GetNextStepInputValue(
                           current_step_input_value, negative_gradients, step)) >
                   (GetFuncValueForArray(func, current_step_input_value) +
                    (-1.0 * armijo_parameter * step *
                     MyOptimization::ConvexOptimization::utils::GetDotProductOfTwoArray(
                         negative_gradients, negative_gradients))))
            {
                step /= 2;
                if (step < stop_step)
                {
                    LOG(ERROR) << "step has minize als " << stop_step
                               << ", but linear search still not finished !!!";
                    break;
                }
            }

            return;
        }

    } // namespace ConvexOptimization

} // namespace MyOptimization
