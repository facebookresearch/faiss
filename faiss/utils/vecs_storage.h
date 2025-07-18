/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Declares functions to read or write in C/C++ the bvecs, ivecs and fvecs
 * files from http://corpus-texmex.irisa.fr/
 */

#pragma once
#include <sys/types.h>

namespace faiss {

/// read vectors from file in bvec format
float* bvecs_read(const char* fileName, size_t num, size_t numOffset, int* dim);

/// read vectors from file in ivec format
int* ivecs_read(const char* fileName, size_t num, size_t numOffset, int* dim);

/// read vectors from file in fvec format
float* fvecs_read(const char* fileName, size_t num, size_t numOffset, int* dim);

/// write vectors to file in bvecs format
void bvecs_write(const char* fileName, size_t num, int dim, float* vecs);

/// write vectors to file in ivecs format
void ivecs_write(const char* fileName, size_t num, int dim, int* vecs);

/// write vectors to file in fvecs format
void fvecs_write(const char* fileName, size_t num, int dim, float* vecs);

} // namespace faiss
