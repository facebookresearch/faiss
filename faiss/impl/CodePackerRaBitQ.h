/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/impl/CodePacker.h>

namespace faiss {

/** CodePacker for RaBitQ that allocates enlarged blocks.
 *
 * Each block contains the standard PQ4 packed codes region (bbs * nsq / 2
 * bytes) followed by an auxiliary data region for per-vector factors.
 * The pack_1/unpack_1 operations transfer BOTH the PQ4 codes and the
 * auxiliary data, so callers such as BlockInvertedLists::remove_ids()
 * and add_entries() automatically preserve auxiliary data.
 *
 * code_size = PQ4 flat bytes + aux_size_per_vec, which must match the
 * buffer sizes allocated by callers (e.g. the index's code_size field).
 */
struct CodePackerRaBitQ : CodePacker {
    size_t nsq;
    size_t aux_size_per_vec;

    /** Construct a RaBitQ code packer.
     * @param nsq              number of sub-quantizers (M2)
     * @param bbs              block size (number of vectors per block)
     * @param aux_per_vector   bytes of auxiliary data per vector
     */
    CodePackerRaBitQ(size_t nsq, size_t bbs, size_t aux_per_vector);

    CodePacker* clone() const final;

    void pack_1(const uint8_t* flat_code, size_t offset, uint8_t* block)
            const final;
    void unpack_1(const uint8_t* block, size_t offset, uint8_t* flat_code)
            const final;
};

} // namespace faiss
