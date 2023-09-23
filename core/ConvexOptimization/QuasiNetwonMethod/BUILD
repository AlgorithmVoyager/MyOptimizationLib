cc_library(
    name = "quasi_newton_method",
    hdrs = [
        "bfgs.h",
        "cautious_bfgs_with_wolfe.h",
        "lbfgs_with_lewis_and_overton.h",
        "limit_memory_newton_method.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/BaseMath:function",
        "//core/ConvexOptimization/StepSearch:armijo",
        "//core/ConvexOptimization/StepSearch:lewis_overton_condition",
        "//core/ConvexOptimization/StepSearch:wolfe",
    ],
)
