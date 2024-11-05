/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Utils for index_read

#ifndef FAISS_INDEX_READ_UTILS_H
#define FAISS_INDEX_READ_UTILS_H

#include <faiss/IndexIVF.h>
#include <faiss/impl/io.h>

#pragma once

namespace faiss {
struct ProductQuantizer;
struct ScalarQuantizer;

void read_index_header(Index* idx, IOReader* f);
void read_direct_map(DirectMap* dm, IOReader* f);
void read_ivf_header(
        IndexIVF* ivf,
        IOReader* f,
        std::vector<std::vector<idx_t>>* ids = nullptr);
void read_InvertedLists(IndexIVF* ivf, IOReader* f, int io_flags);
ArrayInvertedLists* set_array_invlist(
        IndexIVF* ivf,
        std::vector<std::vector<idx_t>>& ids);
void read_ProductQuantizer(ProductQuantizer* pq, IOReader* f);
void read_ScalarQuantizer(ScalarQuantizer* ivsc, IOReader* f);

} // namespace faiss

#endif
