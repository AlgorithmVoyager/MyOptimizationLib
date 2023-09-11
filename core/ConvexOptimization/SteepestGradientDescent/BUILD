cc_library(
    name = "armijo",
    srcs = ["armijo_condition.cpp"],
    hdrs = ["armijo_condition.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/BaseMath:function",
        "//core/ConvexOptimization/SteepestGradientDescent/utils",
        "@com_github_google_glog//:glog",
    ],
)

cc_library(
    name = "steepest_gradient_descent",
    srcs = ["steepest_gradient_descent.cpp"],
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
        "@com_github_google_glog//:glog",
    ],
)
