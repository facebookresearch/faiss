/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

/* Random generators. Implemented here for speed and to make
 * sequences reproducible.
 */

#pragma once

#include <stdint.h>
#include <random>

namespace faiss {

/**************************************************
 * Random data generation functions
 **************************************************/

/// random generator that can be used in multithreaded contexts
struct RandomGenerator {
    std::mt19937 mt;

    /// random positive integer
    int rand_int();

    /// random int64_t
    int64_t rand_int64();

    /// generate random integer between 0 and max-1
    int rand_int(int max);

    /// between 0 and 1
    float rand_float();

    double rand_double();

    explicit RandomGenerator(int64_t seed = 1234);
};

/// fast random generator that cannot be used in multithreaded contexts.
/// based on https://prng.di.unimi.it/
struct SplitMix64RandomGenerator {
    uint64_t state;

    /// random positive integer
    int rand_int();

    /// random int64_t
    int64_t rand_int64();

    /// generate random integer between 0 and max-1
    int rand_int(int max);

    /// between 0 and 1
    float rand_float();

    double rand_double();

    explicit SplitMix64RandomGenerator(int64_t seed = 1234);

    uint64_t next();
};

/* Generate an array of uniform random floats / multi-threaded implementation */
void float_rand(float* x, size_t n, int64_t seed);
void float_randn(float* x, size_t n, int64_t seed);
void int64_rand(int64_t* x, size_t n, int64_t seed);
void byte_rand(uint8_t* x, size_t n, int64_t seed);
// max is actually the maximum value + 1
void int64_rand_max(int64_t* x, size_t n, uint64_t max, int64_t seed);

/* random permutation */
void rand_perm(int* perm, size_t n, int64_t seed);
void rand_perm_splitmix64(int* perm, size_t n, int64_t seed);

/* Random set of vectors with intrinsic dimensionality 10 that is harder to
 * index than a subspace of dim 10 but easier than uniform data in dimension d
 * */
void rand_smooth_vectors(size_t n, size_t d, float* x, int64_t seed);

} // namespace faiss
