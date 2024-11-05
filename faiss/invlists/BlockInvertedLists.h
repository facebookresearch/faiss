/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/index_io.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/InvertedListsIOHook.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

struct CodePacker;
struct IDSelector;

/** Inverted Lists that are organized by blocks.
 *
 * Different from the regular inverted lists, the codes are organized by blocks
 * of size block_size bytes that reprsent a set of n_per_block. Therefore, code
 * allocations are always rounded up to block_size bytes. The codes are also
 * aligned on 32-byte boundaries for use with SIMD.
 *
 * To avoid misinterpretations, the code_size is set to (size_t)(-1), even if
 * arguably the amount of memory consumed by code is block_size / n_per_block.
 *
 * The writing functions add_entries and update_entries operate on block-aligned
 * data.
 */
struct BlockInvertedLists : InvertedLists {
    size_t n_per_block = 0; // nb of vectors stored per block
    size_t block_size = 0;  // nb bytes per block

    // required to interpret the content of the blocks (owned by this)
    const CodePacker* packer = nullptr;

    std::vector<AlignedTable<uint8_t>> codes;
    std::vector<std::vector<idx_t>> ids;

    BlockInvertedLists(size_t nlist, size_t vec_per_block, size_t block_size);
    BlockInvertedLists(size_t nlist, const CodePacker* packer);

    BlockInvertedLists();

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;
    /// remove ids from the InvertedLists
    size_t remove_ids(const IDSelector& sel);

    // works only on empty BlockInvertedLists
    // the codes should be of size ceil(n_entry / n_per_block) * block_size
    // and padded with 0s
    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    /// not implemented
    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    // also pads new data with 0s
    void resize(size_t list_no, size_t new_size) override;

    ~BlockInvertedLists() override;
};

struct BlockInvertedListsIOHook : InvertedListsIOHook {
    BlockInvertedListsIOHook();
    void write(const InvertedLists* ils, IOWriter* f) const override;
    InvertedLists* read(IOReader* f, int io_flags) const override;
};

} // namespace faiss
