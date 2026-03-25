/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Huawei Technologies Co., Ltd.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NEON-optimized matrix block transpose

#ifdef COMPILE_SIMD_ARM_NEON

#include <arm_neon.h>
#include <cstring>

namespace faiss {

// Migrated from krl_matrix_block_transpose_kernel
static void matrix_block_transpose_kernel_neon(
        const uint32_t* src,
        size_t dim,
        size_t blocksize,
        uint32_t* block) {
    uint32x4_t matrix[16];
    uint64x2_t tmp[4];
    for (size_t i = 0; i < blocksize; i += 16) {
        for (size_t j = 0; j < 16; j += 4) {
            matrix[j] = vld1q_u32(src + (i + j) * dim);
            matrix[j + 1] = vld1q_u32(src + (i + j + 1) * dim);
            matrix[j + 2] = vld1q_u32(src + (i + j + 2) * dim);
            matrix[j + 3] = vld1q_u32(src + (i + j + 3) * dim);
            tmp[0] = vreinterpretq_u64_u32(vtrn1q_u32(matrix[j], matrix[j + 1]));
            tmp[1] = vreinterpretq_u64_u32(vtrn2q_u32(matrix[j], matrix[j + 1]));
            tmp[2] = vreinterpretq_u64_u32(vtrn1q_u32(matrix[j + 2], matrix[j + 3]));
            tmp[3] = vreinterpretq_u64_u32(vtrn2q_u32(matrix[j + 2], matrix[j + 3]));
            matrix[j] = vreinterpretq_u32_u64(vtrn1q_u64(tmp[0], tmp[2]));
            matrix[j + 1] = vreinterpretq_u32_u64(vtrn1q_u64(tmp[1], tmp[3]));
            matrix[j + 2] = vreinterpretq_u32_u64(vtrn2q_u64(tmp[0], tmp[2]));
            matrix[j + 3] = vreinterpretq_u32_u64(vtrn2q_u64(tmp[1], tmp[3]));
        }
        vst1q_u32(block + i, matrix[0]);
        vst1q_u32(block + i + 4, matrix[4]);
        vst1q_u32(block + i + 8, matrix[8]);
        vst1q_u32(block + i + 12, matrix[12]);
        vst1q_u32(block + i + blocksize, matrix[1]);
        vst1q_u32(block + i + blocksize + 4, matrix[5]);
        vst1q_u32(block + i + blocksize + 8, matrix[9]);
        vst1q_u32(block + i + blocksize + 12, matrix[13]);
        vst1q_u32(block + i + 2 * blocksize, matrix[2]);
        vst1q_u32(block + i + 2 * blocksize + 4, matrix[6]);
        vst1q_u32(block + i + 2 * blocksize + 8, matrix[10]);
        vst1q_u32(block + i + 2 * blocksize + 12, matrix[14]);
        vst1q_u32(block + i + 3 * blocksize, matrix[3]);
        vst1q_u32(block + i + 3 * blocksize + 4, matrix[7]);
        vst1q_u32(block + i + 3 * blocksize + 8, matrix[11]);
        vst1q_u32(block + i + 3 * blocksize + 12, matrix[15]);
    }
}

// Migrated from krl_matrix_block_transpose
void matrix_block_transpose_neon(
        const uint32_t* src,
        size_t ny,
        size_t dim,
        size_t blocksize,
        uint32_t* block) {
    size_t i = 0;
    size_t bid = 0;
    for (; i + blocksize <= ny; i += blocksize) {
        size_t d = 0;
        for (; d + 4 <= dim; d += 4) {
            matrix_block_transpose_kernel_neon(
                    src + i * dim + d, dim, blocksize, block + bid);
            bid += 4 * blocksize;
        }
        for (; d < dim; ++d) {
            for (size_t j = 0; j < blocksize; ++j) {
                block[bid + j] = src[(i + j) * dim + d];
            }
            bid += blocksize;
        }
    }
    if (i < ny) {
        const size_t left = ny - i;
        for (size_t d = 0; d < dim; ++d) {
            for (size_t j = 0; j < left; ++j) {
                block[bid + j] = src[(i + j) * dim + d];
            }
            std::memset(block + bid + left, 0, (blocksize - left) * sizeof(uint32_t));
            bid += blocksize;
        }
    }
}

} // namespace faiss

#endif // COMPILE_SIMD_ARM_NEON
