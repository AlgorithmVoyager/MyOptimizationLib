cc_test(
    name = "lbfgs_test",
    srcs = [
        # "bfgs_test.cpp",
        # "cautious_bfgs_with_wolfe_test.cpp",
        "lbfgs_test.cpp",
    ],
    deps = [
        "//core/ConvexOptimization/QuasiNetwonMethod:quasi_newton_method",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "bfgs_test",
    srcs = [
        "bfgs_test.cpp",
        # "cautious_bfgs_with_wolfe_test.cpp",
        # "lbfgs_test.cpp",
    ],
    deps = [
        "//core/ConvexOptimization/QuasiNetwonMethod:quasi_newton_method",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "lbfgs_levis_test",
    srcs = [
        "lbfgs_with_lewis_and_overton_test.cpp",
    ],
    deps = [
        "//core/ConvexOptimization/QuasiNetwonMethod:quasi_newton_method",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "conjugate_gradient_test",
    srcs = [
        "linear_newton_conjugate_gradient_method_test.cpp",
    ],
    deps = [
        "//core/ConvexOptimization/QuasiNetwonMethod:quasi_newton_method",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "truncated_newton_cg_test",
    srcs = [
        "truncated_newton_cg_test.cpp",
    ],
    deps = [
        "//core/ConvexOptimization/QuasiNetwonMethod:quasi_newton_method",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)
