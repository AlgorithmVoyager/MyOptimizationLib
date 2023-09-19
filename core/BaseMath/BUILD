cc_library(
    name = "container",
    srcs = [],
    hdrs = ["container.h"],
    copts = [
        "-std=c++17",
        "-Wall",
        "-Werror",
        "-Wextra",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "function",
    srcs = [],
    hdrs = ["function.h"],
    copts = [
        "-std=c++17",
        "-Wall",
        "-Werror",
        "-Wextra",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/BaseMath:container",
        "//logging:log",
        "@eigen",
    ],
)

cc_library(
    name = "matrix",
    srcs = [],
    hdrs = ["matrix.h"],
    copts = [
        "-std=c++17",
        "-Wall",
        "-Werror",
        "-Wextra",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//logging:log",
        "@eigen",
    ],
)
