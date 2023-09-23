cc_test(
    name = "armijo_test",
    srcs = [
        "armijo_test.cpp",
    ],
    copts = ["-std=c++17"],
    deps = [
        "//core/ConvexOptimization/StepSearch:armijo",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "wolfe_test",
    srcs = [
        "wolfe_condition_test.cpp",
    ],
    copts = ["-std=c++17"],
    deps = [
        "//core/ConvexOptimization/StepSearch:wolfe",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "levis_and_overton_condition_test",
    srcs = [
        "levis_and_overton_condition_test.cpp",
    ],
    copts = ["-std=c++17"],
    deps = [
        "//core/ConvexOptimization/StepSearch:lewis_overton_condition",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)
