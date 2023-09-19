cc_test(
    name = "function_test",
    srcs = ["function_test.cpp"],
    copts = ["-std=c++17"],
    deps = [
        "//core/BaseMath:container",
        "//core/BaseMath:function",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "matrix_test",
    srcs = ["matrix_test.cpp"],
    copts = ["-std=c++17"],
    deps = [
        "//core/BaseMath:matrix",
        "@gtest",
        "@gtest//:gtest_main",
    ],
)
