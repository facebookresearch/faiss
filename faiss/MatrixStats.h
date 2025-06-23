/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <stdint.h>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace faiss {

/** Reports some statistics on a dataset and comments on them.
 *
 * It is a class rather than a function so that all stats can also be
 * accessed from code */

struct MatrixStats {
    MatrixStats(size_t n, size_t d, const float* x);
    std::string comments;

    // raw statistics
    size_t n = 0, d = 0;
    size_t n_collision = 0;
    size_t n_valid = 0;
    size_t n0 = 0;
    double min_norm2 = HUGE_VALF;
    double max_norm2 = 0;
    uint64_t hash_value = 0;

    struct PerDimStats {
        /// counts of various special entries
        size_t n = 0;
        size_t n_nan = 0;
        size_t n_inf = 0;
        size_t n0 = 0;

        /// to get min/max and stddev values
        float min = HUGE_VALF;
        float max = -HUGE_VALF;
        double sum = 0;
        double sum2 = 0;

        size_t n_valid = 0;
        double mean = NAN;
        double stddev = NAN;

        void add(float x);
        void compute_mean_std();
    };

    std::vector<PerDimStats> per_dim_stats;
    struct Occurrence {
        size_t first;
        size_t count;
    };
    std::unordered_map<uint64_t, Occurrence> occurrences;

    char* buf;
    size_t nbuf;
    void do_comment(const char* fmt, ...);
};

} // namespace faiss
