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

cc_test(
    name = "utils_test",
    srcs = [
        "utils/utils_test.cpp",
    ],
    copts = ["-std=c++17"],
    deps = [
        "//core/ConvexOptimization/SteepestGradientDescent/utils",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "armijo_test",
    srcs = [
        "armijo_test.cpp",
    ],
    copts = ["-std=c++17"],
    deps = [
        "//core/ConvexOptimization/SteepestGradientDescent:armijo",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)
