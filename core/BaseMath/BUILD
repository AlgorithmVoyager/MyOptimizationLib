cc_library(
    name = "container",
	hdrs = ["container.h"],
    srcs = ["container.cpp"],
    copts = ["-std=c++17","-Wall", "-Werror", "-Wextra"],
     visibility = ["//visibility:public"],
)

cc_library(
    name = "function",
	hdrs = ["function.h"],
    srcs = ["function.cpp"],
    copts = ["-std=c++17","-Wall", "-Werror", "-Wextra"],
    deps =["//core/BaseMath:container"],
    visibility = ["//visibility:public"],
)


cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = ["//core/BaseMath:function"],
    copts = ["-std=c++17","-Wall", "-Werror", "-Wextra"],
    visibility = ["//visibility:public"],
)
