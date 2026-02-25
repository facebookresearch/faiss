/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/CodePackerRaBitQ.h>
#include <faiss/impl/pq4_fast_scan.h>

#include <cstring>

namespace faiss {

CodePackerRaBitQ::CodePackerRaBitQ(
        size_t nsq,
        size_t bbs,
        size_t aux_per_vector) {
    this->nsq = nsq;
    this->aux_size_per_vec = aux_per_vector;
    nvec = bbs;
    const size_t pq4_bytes = (nsq * 4 + 7) / 8;
    // code_size covers PQ4 codes + auxiliary data so that callers
    // (BlockInvertedLists::remove_ids, add_entries, etc.) allocate
    // buffers large enough and pack_1/unpack_1 transfer everything.
    code_size = pq4_bytes + aux_per_vector;
    // block_size = PQ4 packed codes + auxiliary data region
    block_size = ((nsq + 1) / 2) * bbs + aux_per_vector * bbs;
}

void CodePackerRaBitQ::pack_1(
        const uint8_t* flat_code,
        size_t offset,
        uint8_t* block) const {
    const size_t bbs = nvec;
    const size_t pq4_bytes = (nsq * 4 + 7) / 8;
    if (offset >= nvec) {
        block += (offset / nvec) * block_size;
        offset = offset % nvec;
    }
    for (size_t i = 0; i < pq4_bytes; i++) {
        uint8_t code = flat_code[i];
        pq4_set_packed_element(block, code & 15, bbs, nsq, offset, 2 * i);
        pq4_set_packed_element(block, code >> 4, bbs, nsq, offset, 2 * i + 1);
    }
    // Pack auxiliary data (factors, ex-codes) into the block aux region
    if (aux_size_per_vec > 0) {
        const size_t packed_block_size = ((nsq + 1) / 2) * bbs;
        uint8_t* dst = block + packed_block_size + offset * aux_size_per_vec;
        memcpy(dst, flat_code + pq4_bytes, aux_size_per_vec);
    }
}

void CodePackerRaBitQ::unpack_1(
        const uint8_t* block,
        size_t offset,
        uint8_t* flat_code) const {
    const size_t bbs = nvec;
    const size_t pq4_bytes = (nsq * 4 + 7) / 8;
    if (offset >= nvec) {
        block += (offset / nvec) * block_size;
        offset = offset % nvec;
    }
    for (size_t i = 0; i < pq4_bytes; i++) {
        uint8_t code0, code1;
        code0 = pq4_get_packed_element(block, bbs, nsq, offset, 2 * i);
        code1 = pq4_get_packed_element(block, bbs, nsq, offset, 2 * i + 1);
        flat_code[i] = code0 | (code1 << 4);
    }
    // Unpack auxiliary data from the block aux region
    if (aux_size_per_vec > 0) {
        const size_t packed_block_size = ((nsq + 1) / 2) * bbs;
        const uint8_t* src =
                block + packed_block_size + offset * aux_size_per_vec;
        memcpy(flat_code + pq4_bytes, src, aux_size_per_vec);
    }
}

CodePacker* CodePackerRaBitQ::clone() const {
    return new CodePackerRaBitQ(*this);
}

} // namespace faiss
