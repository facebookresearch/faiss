/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/CodePacker.h>

#include <cassert>
#include <cstring>

namespace faiss {

/*********************************************
 * CodePacker
 * default of pack_all / unpack_all loops over the _1 versions
 */

void CodePacker::pack_all(const uint8_t* flat_codes, uint8_t* block) const {
    for (size_t i = 0; i < nvec; i++) {
        pack_1(flat_codes + code_size * i, i, block);
    }
}

void CodePacker::unpack_all(const uint8_t* block, uint8_t* flat_codes) const {
    for (size_t i = 0; i < nvec; i++) {
        unpack_1(block, i, flat_codes + code_size * i);
    }
}

/*********************************************
 * CodePackerFlat
 */

CodePackerFlat::CodePackerFlat(size_t code_size) {
    this->code_size = code_size;
    nvec = 1;
    block_size = code_size;
}

void CodePackerFlat::pack_all(const uint8_t* flat_codes, uint8_t* block) const {
    memcpy(block, flat_codes, code_size);
}

void CodePackerFlat::unpack_all(const uint8_t* block, uint8_t* flat_codes)
        const {
    memcpy(flat_codes, block, code_size);
}

void CodePackerFlat::pack_1(
        const uint8_t* flat_code,
        size_t offset,
        uint8_t* block) const {
    assert(offset == 0);
    pack_all(flat_code, block);
}

void CodePackerFlat::unpack_1(
        const uint8_t* block,
        size_t offset,
        uint8_t* flat_code) const {
    assert(offset == 0);
    unpack_all(block, flat_code);
}

} // namespace faiss
