/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDGrouper.h>

namespace faiss {

/***********************************************************************
 * IDGrouperBitmap
 ***********************************************************************/

IDGrouperBitmap::IDGrouperBitmap(size_t n, uint64_t* bitmap)
        : n(n), bitmap(bitmap) {}

idx_t IDGrouperBitmap::get_group(idx_t id) const {
    assert(id >= 0 && "id shouldn't be less than zero");
    assert(id < this->n * 64 && "is should be less than total number of bits");

    idx_t index = id >> 6; // div by 64
    uint64_t block = this->bitmap[index] >>
            (id & 63); // Equivalent of words[i] >> (index % 64)
    // block is non zero after right shift, it means, next set bit is in current
    // block The index of set bit is "given index" + "trailing zero in the right
    // shifted word"
    if (block != 0) {
        return id + __builtin_ctzll(block);
    }

    while (++index < this->n) {
        block = this->bitmap[index];
        if (block != 0) {
            return (index << 6) + __builtin_ctzll(block);
        }
    }

    return NO_MORE_DOCS;
}

void IDGrouperBitmap::set_group(idx_t group_id) {
    idx_t index = group_id >> 6;
    this->bitmap[index] |= 1ULL
            << (group_id & 63); // Equivalent of 1ULL << (value % 64)
}

} // namespace faiss
