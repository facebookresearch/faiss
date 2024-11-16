/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>

namespace faiss {

/**
 * Packing consists in combining a fixed number of codes of constant size
 * (code_size) into a block of data where they may (or may not) be interleaved
 * for efficient consumption by distance computation kernels. This exists for
 * the "fast_scan" indexes on CPU and for some GPU kernels.
 */
struct CodePacker {
    size_t code_size;  // input code size in bytes
    size_t nvec;       // number of vectors per block
    size_t block_size; // size of one block in bytes (>= code_size * nvec)

    // pack a single code to a block
    virtual void pack_1(
            const uint8_t*
                    flat_code, // code to write to the block, size code_size
            size_t offset,     // offset in the block (0 <= offset < nvec)
            uint8_t* block     // block to write to (size block_size)
    ) const = 0;

    // unpack a single code from a block
    virtual void unpack_1(
            const uint8_t* block, // block to read from (size block_size)
            size_t offset,        // offset in the block (0 <= offset < nvec)
            uint8_t* flat_code    // where to write the resulting code, size
                                  // code_size
    ) const = 0;

    // pack all code in a block
    virtual void pack_all(
            const uint8_t* flat_codes, // codes to write to the block, size
                                       // (nvec * code_size)
            uint8_t* block             // block to write to (size block_size)
    ) const;

    // unpack all code in a block
    virtual void unpack_all(
            const uint8_t* block, // block to read from (size block_size)
            uint8_t* flat_codes // where to write the resulting codes size (nvec
                                // * code_size)
    ) const;

    virtual ~CodePacker() {}
};

/** Trivial code packer where codes are stored one by one */
struct CodePackerFlat : CodePacker {
    explicit CodePackerFlat(size_t code_size);

    void pack_1(const uint8_t* flat_code, size_t offset, uint8_t* block)
            const final;
    void unpack_1(const uint8_t* block, size_t offset, uint8_t* flat_code)
            const final;

    void pack_all(const uint8_t* flat_codes, uint8_t* block) const final;
    void unpack_all(const uint8_t* block, uint8_t* flat_codes) const final;
};

} // namespace faiss
