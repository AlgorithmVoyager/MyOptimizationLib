cc_test(
    name = "steepest_gradient_descent_test",
    srcs = [
        # "armijo_test.cpp",
        "steepest_gradient_descent_test.cpp",
        # "utils/utils_test.cpp",
    ],
    copts = ["-std=c++17"],
    deps = [
        # "//core/ConvexOptimization/SteepestGradientDescent:armijo",
        "//core/ConvexOptimization/SteepestGradientDescent:steepest_gradient_descent",
        # "//core/ConvexOptimization/SteepestGradientDescent/utils",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)
