cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        "//core/ConvexOptimization/QuasiNetwonMethod:quasi_newton_method",
        "//logging:log",
    ],
)

cc_library(
    name = "path_generation",
    hdrs = ["path_generation.h"],
    deps = [
        "//core/ConvexOptimization/QuasiNetwonMethod:quasi_newton_method",
        "//logging:log",
    ],
)

cc_binary(
    name = "path_generation_main",
    srcs = ["path_generation.cpp"],
    deps = [
        "//core/ConvexOptimization/QuasiNetwonMethod:quasi_newton_method",
        "//core/ConvexOptimization/QuasiNetwonMethod/usage:path_generation",
        "//logging:log",
    ],
)
