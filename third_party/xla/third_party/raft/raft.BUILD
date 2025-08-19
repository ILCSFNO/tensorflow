load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

BASE_COPTS = [
    "-fexceptions",
    "-Wno-unused-variable",
    "-Wno-ctad-maybe-unsupported",
    "-Wno-reorder-ctor",
    "-Wno-non-virtual-dtor",
    "-Wno-uninitialized",
    "-Wno-pass-failed",
    "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
    "-DRAFT_SYSTEM_LITTLE_ENDIAN",
]

cuda_library(
    name = "raft_matrix_select_k",
    copts = BASE_COPTS,
    includes = ["cpp/include"],
    textual_hdrs = glob([
        "cpp/include/**/*.cuh",
        "cpp/include/**/*.hpp",
        "cpp/include/**/*.h",
        "cpp/src/**/*.cuh",
        "cpp/internal/**/*.cuh",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "@kokkos//:mdspan",
        "@rapids_logger",
        "@rmm",
    ],
)

cuda_library(
    name = "select_k_runner",
    srcs = ["select_k_runner.cu.cc"],
    hdrs = ["select_k_runner.hpp"],
    copts = BASE_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":raft_matrix_select_k",
    ],
)

cc_test(
    name = "select_k_smoke_test",
    srcs = ["select_k_smoke_test.cu.cc"],
    copts = BASE_COPTS,
    deps = [
        ":select_k_runner",
        "@com_google_googletest//:gtest_main",
    ],
)
