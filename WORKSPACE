load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

http_archive(
    name = "gtest",
    strip_prefix = "googletest-5ab508a01f9eb089207ee87fd547d290da39d015",
    urls = ["https://github.com/google/googletest/archive/5ab508a01f9eb089207ee87fd547d290da39d015.zip"],
)

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
    strip_prefix = "glog-0.6.0",
    urls = ["https://github.com/google/glog/archive/v0.6.0.zip"],
)

http_archive(
    name = "rules_eigen",
    url = "https://github.com/AlgorithmVoyager/rules_eigen/archive/eigen-3.4.0-v0.tar.gz",
    sha256 = "668d7503e44daa76e68278373afb04e14e962068dbe461e5ad636bb5c9ea9e5e",
    strip_prefix = "rules_eigen-eigen-3.4.0-v0",
)

load("@rules_eigen//bzl:repositories.bzl", "eigen_repositories")

eigen_repositories()