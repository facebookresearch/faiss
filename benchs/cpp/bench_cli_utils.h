/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <benchmark/benchmark.h>

/// Define the standard dataset-location flags shared by every bench that can
/// load an on-disk {train, base, query, groundtruth} dataset. Invoke once at
/// global scope in a bench's .cpp (alongside its other DEFINE_* flags):
///
///     BENCH_DEFINE_DATASET_FILE_FLAGS();
///
/// The defaults reproduce the historical SIFT1M behavior, so a bench that
/// simply points --data_dir at a SIFT1M directory keeps working unchanged.
/// Overriding the four filename flags lets the same bench load any dataset
/// with matching framing (e.g. cohere, deep, gist, bigann) without renaming
/// files. Vectors are read as .fvecs (float32) or .bvecs (uint8) based on the
/// filename extension; ground truth is always .ivecs. The resulting flags are
/// FLAGS_data_dir, FLAGS_train_file, FLAGS_base_file, FLAGS_query_file and
/// FLAGS_gt_file, passed positionally to DatasetSIFT1M::load().
#define BENCH_DEFINE_DATASET_FILE_FLAGS()                                       \
    DEFINE_string(data_dir, "sift1M", "path to the dataset directory");         \
    DEFINE_string(                                                              \
            train_file,                                                         \
            "sift_learn.fvecs",                                                 \
            "train/learn vectors filename under --data_dir (.fvecs/.bvecs)");   \
    DEFINE_string(                                                              \
            base_file,                                                          \
            "sift_base.fvecs",                                                  \
            "base/database vectors filename under --data_dir (.fvecs/.bvecs)"); \
    DEFINE_string(                                                              \
            query_file,                                                         \
            "sift_query.fvecs",                                                 \
            "query vectors filename under --data_dir (.fvecs/.bvecs)");         \
    DEFINE_string(                                                              \
            gt_file,                                                            \
            "sift_groundtruth.ivecs",                                           \
            "ground-truth neighbors filename under --data_dir (.ivecs)")

namespace benchmarks {

/// Parse a comma-separated list flag such as "16,32,64". An empty flag
/// value returns `defaults`, so the built-in sweep runs unless the user
/// overrides it.
inline std::vector<int> int_list(
        const std::string& flag_value,
        std::vector<int> defaults) {
    if (flag_value.empty()) {
        return defaults;
    }
    std::vector<int> values;
    std::stringstream ss(flag_value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            size_t pos = 0;
            int v = std::stoi(token, &pos);
            if (pos != token.size()) {
                throw std::invalid_argument(token);
            }
            values.push_back(v);
        } catch (const std::exception&) {
            fprintf(stderr,
                    "benchmarks: cannot parse '%s' as an integer list\n",
                    flag_value.c_str());
            exit(1);
        }
    }
    if (values.empty()) {
        fprintf(stderr,
                "benchmarks: empty integer list '%s'\n",
                flag_value.c_str());
        exit(1);
    }
    return values;
}

/// Same as int_list, for comma-separated strings.
inline std::vector<std::string> str_list(
        const std::string& flag_value,
        std::vector<std::string> defaults) {
    if (flag_value.empty()) {
        return defaults;
    }
    std::vector<std::string> values;
    std::stringstream ss(flag_value);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(token);
        }
    }
    if (values.empty()) {
        fprintf(stderr,
                "benchmarks: empty string list '%s'\n",
                flag_value.c_str());
        exit(1);
    }
    return values;
}

inline void print_help_and_exit(
        const char* argv0,
        const char* description,
        const char* example) {
    const char* prog = strrchr(argv0, '/');
    prog = prog ? prog + 1 : argv0;

    printf("%s — %s\n\n", prog, description);
    printf("Usage: %s [flags]\n", prog);
    printf("Without flags, the full built-in parameter sweep is run.\n\n");

    printf("Parameter flags (defaults define the built-in sweep):\n");
    std::vector<gflags::CommandLineFlagInfo> flags;
    gflags::GetAllFlags(&flags);
    for (const auto& f : flags) {
        // Only this binary's own flags, not gflags' built-ins.
        if (f.filename.find("bench_") == std::string::npos) {
            continue;
        }
        if (f.default_value.empty()) {
            printf("  --%s=<%s>\n", f.name.c_str(), f.type.c_str());
        } else {
            printf("  --%s=<%s>  (default: %s)\n",
                   f.name.c_str(),
                   f.type.c_str(),
                   f.default_value.c_str());
        }
        printf("        %s\n", f.description.c_str());
    }

    printf("\nCommon Google Benchmark flags:\n");
    printf("  --benchmark_list_tests            list all benchmark names\n");
    printf("  --benchmark_filter=<regex>        run only matching benchmarks\n");
    printf("  --benchmark_repetitions=<n>       repeat each benchmark n times\n");
    printf("  --benchmark_min_time=<n>s         min running time per benchmark\n");
    printf("  --benchmark_out=<file> --benchmark_out_format=json|csv|console\n");

    printf("\nExamples:\n");
    printf("  %s                                # full sweep\n", prog);
    printf("  %s --benchmark_list_tests         # see what would run\n", prog);
    if (example != nullptr) {
        printf("  %s %s\n", prog, example);
    }
    exit(0);
}

/// Call at the top of every benchmarks main(). Implements --help/-h
/// (program description, this binary's flags, benchmark-library flags and
/// example invocations), then initializes Google Benchmark and parses the
/// remaining gflags.
///
/// `example` is an optional example argument string shown under "Examples:",
/// e.g. "--d=128,256 --benchmark_filter=fvec_L2sqr".
inline void bench_init(
        int* argc,
        char*** argv,
        const char* description,
        const char* example = nullptr) {
    for (int i = 1; i < *argc; i++) {
        const char* a = (*argv)[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0 ||
            strcmp(a, "-help") == 0) {
            print_help_and_exit((*argv)[0], description, example);
        }
    }
    benchmark::Initialize(argc, *argv);
    gflags::AllowCommandLineReparsing();
    gflags::ParseCommandLineFlags(argc, argv, true);
}

} // namespace benchmarks
