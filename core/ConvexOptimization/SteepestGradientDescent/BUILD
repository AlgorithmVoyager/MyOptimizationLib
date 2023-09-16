cc_library(
    name = "armijo",
    srcs = [],
    hdrs = ["armijo_condition.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/BaseMath:function",
        "//core/ConvexOptimization/SteepestGradientDescent/utils",
        "//logging:log",
    ],
)

cc_library(
    name = "steepest_gradient_descent",
    srcs = [],
    hdrs = ["steepest_gradient_descent.h"],
    copts = [
        "-std=c++17",
        "-Wall",
        "-Werror",
        "-Wextra",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/BaseMath:function",
        "//core/ConvexOptimization/SteepestGradientDescent:armijo",
        "//core/ConvexOptimization/SteepestGradientDescent/utils",
    ],
)
