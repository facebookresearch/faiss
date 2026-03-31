/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Huawei Technologies Co., Ltd.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NEON-optimized ProductQuantizer distance table computation.
// precision=0 (FP32): block-transposed fp32 + continuous_transpose kernels
// precision=1 (INT8): quantized int8 centroids + int8 dot-product kernels
// precision=2 (FP16): quantized fp16 centroids + fp16 kernels

#ifdef COMPILE_SIMD_ARM_NEON

#include <arm_neon.h>
#include <cstring>
#include <vector>

#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
void matrix_block_transpose_neon(
        const uint32_t* src,
        size_t ny,
        size_t dim,
        size_t blocksize,
        uint32_t* block);
} // namespace faiss

namespace faiss {

// blocksize == 16
static void ip_continuous_transpose_16(
        float* dis,
        const float* x,
        const float* y,
        size_t dsub) {
    float32x4_t res[4];
    float32x4_t q = vdupq_n_f32(x[0]);
    res[0] = vmulq_f32(vld1q_f32(y), q);
    res[1] = vmulq_f32(vld1q_f32(y + 4), q);
    res[2] = vmulq_f32(vld1q_f32(y + 8), q);
    res[3] = vmulq_f32(vld1q_f32(y + 12), q);
    for (size_t i = 1; i < dsub; ++i) {
        q = vdupq_n_f32(x[i]);
        res[0] = vmlaq_f32(res[0], vld1q_f32(y + 16 * i), q);
        res[1] = vmlaq_f32(res[1], vld1q_f32(y + 16 * i + 4), q);
        res[2] = vmlaq_f32(res[2], vld1q_f32(y + 16 * i + 8), q);
        res[3] = vmlaq_f32(res[3], vld1q_f32(y + 16 * i + 12), q);
    }
    vst1q_f32(dis, res[0]);
    vst1q_f32(dis + 4, res[1]);
    vst1q_f32(dis + 8, res[2]);
    vst1q_f32(dis + 12, res[3]);
}

// blocksize == 32
static void ip_continuous_transpose_32(
        float* dis,
        const float* x,
        const float* y,
        size_t dsub) {
    float32x4_t res[8];
    float32x4_t q = vdupq_n_f32(x[0]);
    for (int j = 0; j < 8; j++) {
        res[j] = vmulq_f32(vld1q_f32(y + j * 4), q);
    }
    for (size_t i = 1; i < dsub; ++i) {
        q = vdupq_n_f32(x[i]);
        for (int j = 0; j < 8; j++) {
            res[j] = vmlaq_f32(res[j], vld1q_f32(y + 32 * i + j * 4), q);
        }
    }
    for (int j = 0; j < 8; j++) {
        vst1q_f32(dis + j * 4, res[j]);
    }
}

// blocksize == 64
static void ip_continuous_transpose_64(
        float* dis,
        const float* x,
        const float* y,
        size_t dsub) {
    float32x4_t res[16];
    float32x4_t q = vdupq_n_f32(x[0]);
    for (int j = 0; j < 16; j++) {
        res[j] = vmulq_f32(vld1q_f32(y + j * 4), q);
    }
    for (size_t i = 1; i < dsub; ++i) {
        q = vdupq_n_f32(x[i]);
        for (int j = 0; j < 16; j++) {
            res[j] = vmlaq_f32(res[j], vld1q_f32(y + 64 * i + j * 4), q);
        }
    }
    for (int j = 0; j < 16; j++) {
        vst1q_f32(dis + j * 4, res[j]);
    }
}

// blocksize == 16, computes ||centroid[j] - x||^2 for j=0..15
static void l2_continuous_transpose_16(
        float* dis,
        const float* x,
        const float* y,
        size_t dsub) {
    // Initialize with (y[j][0] - x[0])^2 for j=0..15
    float32x4_t q0 = vdupq_n_f32(x[0]);
    float32x4_t d0 = vsubq_f32(vld1q_f32(y), q0);
    float32x4_t d1 = vsubq_f32(vld1q_f32(y + 4), q0);
    float32x4_t d2 = vsubq_f32(vld1q_f32(y + 8), q0);
    float32x4_t d3 = vsubq_f32(vld1q_f32(y + 12), q0);
    float32x4_t res0 = vmulq_f32(d0, d0);
    float32x4_t res1 = vmulq_f32(d1, d1);
    float32x4_t res2 = vmulq_f32(d2, d2);
    float32x4_t res3 = vmulq_f32(d3, d3);
    for (size_t i = 1; i < dsub; ++i) {
        q0 = vdupq_n_f32(x[i]);
        d0 = vsubq_f32(vld1q_f32(y + 16 * i), q0);
        d1 = vsubq_f32(vld1q_f32(y + 16 * i + 4), q0);
        d2 = vsubq_f32(vld1q_f32(y + 16 * i + 8), q0);
        d3 = vsubq_f32(vld1q_f32(y + 16 * i + 12), q0);
        res0 = vmlaq_f32(res0, d0, d0);
        res1 = vmlaq_f32(res1, d1, d1);
        res2 = vmlaq_f32(res2, d2, d2);
        res3 = vmlaq_f32(res3, d3, d3);
    }
    vst1q_f32(dis, res0);
    vst1q_f32(dis + 4, res1);
    vst1q_f32(dis + 8, res2);
    vst1q_f32(dis + 12, res3);
}

// blocksize == 32
static void l2_continuous_transpose_32(
        float* dis,
        const float* x,
        const float* y,
        size_t dsub) {
    float32x4_t res[8];
    float32x4_t q = vdupq_n_f32(x[0]);
    for (int j = 0; j < 8; j++) {
        float32x4_t d = vsubq_f32(vld1q_f32(y + j * 4), q);
        res[j] = vmulq_f32(d, d);
    }
    for (size_t i = 1; i < dsub; ++i) {
        q = vdupq_n_f32(x[i]);
        for (int j = 0; j < 8; j++) {
            float32x4_t d = vsubq_f32(vld1q_f32(y + 32 * i + j * 4), q);
            res[j] = vmlaq_f32(res[j], d, d);
        }
    }
    for (int j = 0; j < 8; j++) {
        vst1q_f32(dis + j * 4, res[j]);
    }
}

// blocksize == 64
static void l2_continuous_transpose_64(
        float* dis,
        const float* x,
        const float* y,
        size_t dsub) {
    float32x4_t res[16];
    float32x4_t q = vdupq_n_f32(x[0]);
    for (int j = 0; j < 16; j++) {
        float32x4_t d = vsubq_f32(vld1q_f32(y + j * 4), q);
        res[j] = vmulq_f32(d, d);
    }
    for (size_t i = 1; i < dsub; ++i) {
        q = vdupq_n_f32(x[i]);
        for (int j = 0; j < 16; j++) {
            float32x4_t d = vsubq_f32(vld1q_f32(y + 64 * i + j * 4), q);
            res[j] = vmlaq_f32(res[j], d, d);
        }
    }
    for (int j = 0; j < 16; j++) {
        vst1q_f32(dis + j * 4, res[j]);
    }
}

static void compute_inner_prod_table_neon(
        size_t M,
        size_t ksub,
        size_t dsub,
        const float* x,          // query, shape (M, dsub)
        const float* y,          // NEON transposed centroids, layout per subq: (dsub, ceil_ksub)
        size_t ceil_ksub,        // ksub rounded up to blocksize
        size_t blocksize,        // 16, 32, or 64
        float* dis) {            // output, shape (M, ksub)
    const size_t left = ksub & (blocksize - 1);
    const float* yp = y;

    for (size_t m = 0; m < M; m++) {
        float* disp = dis + m * ksub;
        const float* xp = x + m * dsub;

        if (blocksize == 64) {
            if (left) {
                float tmp[64];
                size_t i = 0;
                for (; i + 64 <= ksub; i += 64) {
                    ip_continuous_transpose_64(disp + i, xp, yp + i * dsub, dsub);
                }
                ip_continuous_transpose_64(tmp, xp, yp + i * dsub, dsub);
                std::memcpy(disp + i, tmp, left * sizeof(float));
            } else {
                for (size_t i = 0; i < ksub; i += 64) {
                    ip_continuous_transpose_64(disp + i, xp, yp + i * dsub, dsub);
                }
            }
        } else if (blocksize == 32) {
            if (left) {
                float tmp[32];
                size_t i = 0;
                for (; i + 32 <= ksub; i += 32) {
                    ip_continuous_transpose_32(disp + i, xp, yp + i * dsub, dsub);
                }
                ip_continuous_transpose_32(tmp, xp, yp + i * dsub, dsub);
                std::memcpy(disp + i, tmp, left * sizeof(float));
            } else {
                for (size_t i = 0; i < ksub; i += 32) {
                    ip_continuous_transpose_32(disp + i, xp, yp + i * dsub, dsub);
                }
            }
        } else { // blocksize == 16
            if (left) {
                float tmp[16];
                size_t i = 0;
                for (; i + 16 <= ksub; i += 16) {
                    ip_continuous_transpose_16(disp + i, xp, yp + i * dsub, dsub);
                }
                ip_continuous_transpose_16(tmp, xp, yp + i * dsub, dsub);
                std::memcpy(disp + i, tmp, left * sizeof(float));
            } else {
                for (size_t i = 0; i < ksub; i += 16) {
                    ip_continuous_transpose_16(disp + i, xp, yp + i * dsub, dsub);
                }
            }
        }
        yp += ceil_ksub * dsub;
    }
}

static void compute_distance_table_neon(
        size_t M,
        size_t ksub,
        size_t dsub,
        const float* x,          // query, shape (M, dsub)
        const float* y,          // NEON transposed centroids
        size_t ceil_ksub,
        size_t blocksize,
        float* dis) {
    const size_t left = ksub & (blocksize - 1);
    const float* yp = y;

    for (size_t m = 0; m < M; m++) {
        float* disp = dis + m * ksub;
        const float* xp = x + m * dsub;

        if (blocksize == 64) {
            if (left) {
                float tmp[64];
                size_t i = 0;
                for (; i + 64 <= ksub; i += 64) {
                    l2_continuous_transpose_64(disp + i, xp, yp + i * dsub, dsub);
                }
                l2_continuous_transpose_64(tmp, xp, yp + i * dsub, dsub);
                std::memcpy(disp + i, tmp, left * sizeof(float));
            } else {
                for (size_t i = 0; i < ksub; i += 64) {
                    l2_continuous_transpose_64(disp + i, xp, yp + i * dsub, dsub);
                }
            }
        } else if (blocksize == 32) {
            if (left) {
                float tmp[32];
                size_t i = 0;
                for (; i + 32 <= ksub; i += 32) {
                    l2_continuous_transpose_32(disp + i, xp, yp + i * dsub, dsub);
                }
                l2_continuous_transpose_32(tmp, xp, yp + i * dsub, dsub);
                std::memcpy(disp + i, tmp, left * sizeof(float));
            } else {
                for (size_t i = 0; i < ksub; i += 32) {
                    l2_continuous_transpose_32(disp + i, xp, yp + i * dsub, dsub);
                }
            }
        } else { // blocksize == 16
            if (left) {
                float tmp[16];
                size_t i = 0;
                for (; i + 16 <= ksub; i += 16) {
                    l2_continuous_transpose_16(disp + i, xp, yp + i * dsub, dsub);
                }
                l2_continuous_transpose_16(tmp, xp, yp + i * dsub, dsub);
                std::memcpy(disp + i, tmp, left * sizeof(float));
            } else {
                for (size_t i = 0; i < ksub; i += 16) {
                    l2_continuous_transpose_16(disp + i, xp, yp + i * dsub, dsub);
                }
            }
        }
        yp += ceil_ksub * dsub;
    }
}

void ProductQuantizer::initialize_neon_transposed_centroids(
        size_t blocksize,
        int precision_mode) {
    FAISS_THROW_IF_NOT_MSG(!centroids.empty(), "centroids must be set first");

    using Precision = NeonTransposedCentroids::Precision;

    // Clear all storage first
    neon_transposed.data.clear();
    neon_transposed.data_f16.clear();
    neon_transposed.data_u8.clear();

    if (precision_mode == 0) {
        // FP32: block-transpose centroids for continuous_transpose kernels.
        // Output layout per subquantizer: (num_blocks, dsub, blocksize)
        // i.e. matrix_block_transpose_neon output for (ksub, dsub) input.
        FAISS_THROW_IF_NOT_MSG(
                blocksize == 16 || blocksize == 32 || blocksize == 64,
                "blocksize must be 16, 32, or 64");
        const size_t ceil_ksub = (ksub + blocksize - 1) & ~(blocksize - 1);
        neon_transposed.precision = Precision::FP32;
        neon_transposed.blocksize = blocksize;
        neon_transposed.ceil_ksub = ceil_ksub;
        // Total: M * ceil_ksub * dsub floats
        neon_transposed.data.resize(M * ceil_ksub * dsub, 0.0f);
        for (size_t m = 0; m < M; m++) {
            matrix_block_transpose_neon(
                    reinterpret_cast<const uint32_t*>(get_centroids(m, 0)),
                    ksub,
                    dsub,
                    blocksize,
                    reinterpret_cast<uint32_t*>(
                            neon_transposed.data.data() +
                            m * ceil_ksub * dsub));
        }
    } else if (precision_mode == 1) {
        // INT8: quantize centroids to uint8 (L2) or int8 (IP).
        // Layout: (M, ksub, dsub) uint8 — same as original centroids, just quantized.
        // Query will be quantized on-the-fly at search time.
        neon_transposed.precision = Precision::INT8;
        neon_transposed.blocksize = 0;
        neon_transposed.ceil_ksub = 0;
        const size_t total = M * ksub * dsub;
        neon_transposed.data_u8.resize(total);
        // Quantize: map float to uint8 via round-to-nearest, clamp [0,255]
        // Uses same approach as krl quant_u8: vcvtaq_u32_f32 + narrow
        const float* src = centroids.data();
        uint8_t* dst = neon_transposed.data_u8.data();
        size_t i = 0;
        for (; i + 16 <= total; i += 16) {
            float32x4_t a0 = vld1q_f32(src + i);
            float32x4_t a1 = vld1q_f32(src + i + 4);
            float32x4_t a2 = vld1q_f32(src + i + 8);
            float32x4_t a3 = vld1q_f32(src + i + 12);
            uint32x4_t u0 = vcvtaq_u32_f32(a0);
            uint32x4_t u1 = vcvtaq_u32_f32(a1);
            uint32x4_t u2 = vcvtaq_u32_f32(a2);
            uint32x4_t u3 = vcvtaq_u32_f32(a3);
            uint16x8_t s01 = vcombine_u16(vqmovn_u32(u0), vqmovn_u32(u1));
            uint16x8_t s23 = vcombine_u16(vqmovn_u32(u2), vqmovn_u32(u3));
            uint8x16_t b = vcombine_u8(vqmovn_u16(s01), vqmovn_u16(s23));
            vst1q_u8(dst + i, b);
        }
        for (; i < total; i++) {
            float v = src[i];
            dst[i] = (v < 0.f) ? 0 : (v > 255.f) ? 255 : (uint8_t)(v + 0.5f);
        }
    } else {
        // FP16: quantize centroids to float16.
        // Layout: (M, ksub, dsub) uint16 — same as original centroids, just quantized.
        neon_transposed.precision = Precision::FP16;
        neon_transposed.blocksize = 0;
        neon_transposed.ceil_ksub = 0;
        const size_t total = M * ksub * dsub;
        neon_transposed.data_f16.resize(total);
        const float* src = centroids.data();
        uint16_t* dst = neon_transposed.data_f16.data();
        size_t i = 0;
        for (; i + 16 <= total; i += 16) {
            float32x4_t a0 = vld1q_f32(src + i);
            float32x4_t a1 = vld1q_f32(src + i + 4);
            float32x4_t a2 = vld1q_f32(src + i + 8);
            float32x4_t a3 = vld1q_f32(src + i + 12);
            vst1_u16(dst + i,      vreinterpret_u16_f16(vcvt_f16_f32(a0)));
            vst1_u16(dst + i + 4,  vreinterpret_u16_f16(vcvt_f16_f32(a1)));
            vst1_u16(dst + i + 8,  vreinterpret_u16_f16(vcvt_f16_f32(a2)));
            vst1_u16(dst + i + 12, vreinterpret_u16_f16(vcvt_f16_f32(a3)));
        }
        for (; i < total; i++) {
            float16x4_t tmp = vcvt_f16_f32(vdupq_n_f32(src[i]));
            vst1_lane_u16(dst + i, vreinterpret_u16_f16(tmp), 0);
        }
    }
}

static float ip_f16_scalar(
        const float16_t* qx,
        const float16_t* cy,
        size_t dsub) {
    float32x4_t acc = vdupq_n_f32(0.f);
    size_t i = 0;
    for (; i + 4 <= dsub; i += 4) {
        float32x4_t xv = vcvt_f32_f16(vld1_f16(qx + i));
        float32x4_t yv = vcvt_f32_f16(vld1_f16(cy + i));
        acc = vmlaq_f32(acc, xv, yv);
    }
    float res = vaddvq_f32(acc);
    for (; i < dsub; i++) {
        res += (float)qx[i] * (float)cy[i];
    }
    return res;
}

// FP16 L2sqr: ||qx - cy||^2 in fp16 arithmetic, result float.
static float l2_f16_scalar(
        const float16_t* qx,
        const float16_t* cy,
        size_t dsub) {
    float32x4_t acc = vdupq_n_f32(0.f);
    size_t i = 0;
    for (; i + 4 <= dsub; i += 4) {
        float32x4_t xv = vcvt_f32_f16(vld1_f16(qx + i));
        float32x4_t yv = vcvt_f32_f16(vld1_f16(cy + i));
        float32x4_t dv = vsubq_f32(xv, yv);
        acc = vmlaq_f32(acc, dv, dv);
    }
    float res = vaddvq_f32(acc);
    for (; i < dsub; i++) {
        float d = (float)qx[i] - (float)cy[i];
        res += d * d;
    }
    return res;
}

// INT8 inner product: one int8 query subvector vs one uint8 centroid subvector.
static int32_t ip_u8s8_scalar(
        const int8_t* qx,
        const uint8_t* cy,
        size_t dsub) {
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
    for (; i + 16 <= dsub; i += 16) {
        int8x16_t xv = vld1q_s8(qx + i);
        // Reinterpret uint8 centroid as int8 for dot product
        int8x16_t yv = vreinterpretq_s8_u8(vld1q_u8(cy + i));
        acc = vdotq_s32(acc, xv, yv);
    }
    int32_t res = vaddvq_s32(acc);
    for (; i < dsub; i++) {
        res += (int32_t)qx[i] * (int32_t)(int8_t)cy[i];
    }
    return res;
}

// INT8 L2sqr: ||qx - cy||^2 with uint8 operands, result uint32.
static uint32_t l2_u8_scalar(
        const uint8_t* qx,
        const uint8_t* cy,
        size_t dsub) {
    uint32x4_t acc = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 16 <= dsub; i += 16) {
        uint8x16_t xv = vld1q_u8(qx + i);
        uint8x16_t yv = vld1q_u8(cy + i);
        uint8x16_t dv = vabdq_u8(xv, yv);
        acc = vdotq_u32(acc, dv, dv);
    }
    uint32_t res = vaddvq_u32(acc);
    for (; i < dsub; i++) {
        int32_t d = (int32_t)qx[i] - (int32_t)cy[i];
        res += (uint32_t)(d * d);
    }
    return res;
}

void pq_compute_inner_prod_table_neon(
        const ProductQuantizer& pq,
        const float* x,
        float* dis_table) {
    using Precision = ProductQuantizer::NeonTransposedCentroids::Precision;
    const auto& nt = pq.neon_transposed;

    if (nt.precision == Precision::FP32) {
        // Block-transposed fp32 path — highest accuracy, uses continuous_transpose kernels
        compute_inner_prod_table_neon(
                pq.M,
                pq.ksub,
                pq.dsub,
                x,
                nt.data.data(),
                nt.ceil_ksub,
                nt.blocksize,
                dis_table);
    } else if (nt.precision == Precision::FP16) {
        // FP16 path: quantize query x on-the-fly, compute IP vs fp16 centroids.
        std::vector<float16_t> qx_f16(pq.M * pq.dsub);
        // quant_f16: convert fp32 query to fp16
        size_t total_q = pq.M * pq.dsub;
        size_t qi = 0;
        for (; qi + 4 <= total_q; qi += 4) {
            vst1_f16(qx_f16.data() + qi, vcvt_f16_f32(vld1q_f32(x + qi)));
        }
        for (; qi < total_q; qi++) {
            float16x4_t tmp = vcvt_f16_f32(vdupq_n_f32(x[qi]));
            vst1_lane_f16(qx_f16.data() + qi, tmp, 0);
        }
        const float16_t* cy = reinterpret_cast<const float16_t*>(nt.data_f16.data());
        for (size_t m = 0; m < pq.M; m++) {
            const float16_t* qxm = qx_f16.data() + m * pq.dsub;
            const float16_t* cym = cy + m * pq.ksub * pq.dsub;
            float* disp = dis_table + m * pq.ksub;
            for (size_t k = 0; k < pq.ksub; k++) {
                disp[k] = ip_f16_scalar(qxm, cym + k * pq.dsub, pq.dsub);
            }
        }
    } else {
        // INT8 path: quantize query x to int8 on-the-fly, compute IP vs uint8 centroids.
        std::vector<int8_t> qx_s8(pq.M * pq.dsub);
        // quant_s8: vcvtaq_s32_f32 + narrow to int8
        size_t total_q = pq.M * pq.dsub;
        size_t qi = 0;
        for (; qi + 16 <= total_q; qi += 16) {
            float32x4_t a0 = vld1q_f32(x + qi);
            float32x4_t a1 = vld1q_f32(x + qi + 4);
            float32x4_t a2 = vld1q_f32(x + qi + 8);
            float32x4_t a3 = vld1q_f32(x + qi + 12);
            int32x4_t i0 = vcvtaq_s32_f32(a0);
            int32x4_t i1 = vcvtaq_s32_f32(a1);
            int32x4_t i2 = vcvtaq_s32_f32(a2);
            int32x4_t i3 = vcvtaq_s32_f32(a3);
            int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            int8x16_t b = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
            vst1q_s8(qx_s8.data() + qi, b);
        }
        for (; qi < total_q; qi++) {
            float v = x[qi];
            qx_s8[qi] = (v < -128.f) ? -128 : (v > 127.f) ? 127 : (int8_t)(v + (v >= 0.f ? 0.5f : -0.5f));
        }
        const uint8_t* cy = nt.data_u8.data();
        for (size_t m = 0; m < pq.M; m++) {
            const int8_t* qxm = qx_s8.data() + m * pq.dsub;
            const uint8_t* cym = cy + m * pq.ksub * pq.dsub;
            float* disp = dis_table + m * pq.ksub;
            for (size_t k = 0; k < pq.ksub; k++) {
                disp[k] = (float)ip_u8s8_scalar(qxm, cym + k * pq.dsub, pq.dsub);
            }
        }
    }
}

void pq_compute_distance_table_neon(
        const ProductQuantizer& pq,
        const float* x,
        float* dis_table) {
    using Precision = ProductQuantizer::NeonTransposedCentroids::Precision;
    const auto& nt = pq.neon_transposed;

    if (nt.precision == Precision::FP32) {
        // Block-transposed fp32 path
        compute_distance_table_neon(
                pq.M,
                pq.ksub,
                pq.dsub,
                x,
                nt.data.data(),
                nt.ceil_ksub,
                nt.blocksize,
                dis_table);
    } else if (nt.precision == Precision::FP16) {
        // FP16 path: quantize query, compute L2sqr vs fp16 centroids.
        std::vector<float16_t> qx_f16(pq.M * pq.dsub);
        size_t total_q = pq.M * pq.dsub;
        size_t qi = 0;
        for (; qi + 4 <= total_q; qi += 4) {
            vst1_f16(qx_f16.data() + qi, vcvt_f16_f32(vld1q_f32(x + qi)));
        }
        for (; qi < total_q; qi++) {
            float16x4_t tmp = vcvt_f16_f32(vdupq_n_f32(x[qi]));
            vst1_lane_f16(qx_f16.data() + qi, tmp, 0);
        }
        const float16_t* cy = reinterpret_cast<const float16_t*>(nt.data_f16.data());
        for (size_t m = 0; m < pq.M; m++) {
            const float16_t* qxm = qx_f16.data() + m * pq.dsub;
            const float16_t* cym = cy + m * pq.ksub * pq.dsub;
            float* disp = dis_table + m * pq.ksub;
            for (size_t k = 0; k < pq.ksub; k++) {
                disp[k] = l2_f16_scalar(qxm, cym + k * pq.dsub, pq.dsub);
            }
        }
    } else {
        // INT8 path: quantize query to uint8, compute L2sqr vs uint8 centroids.
        std::vector<uint8_t> qx_u8(pq.M * pq.dsub);
        size_t total_q = pq.M * pq.dsub;
        size_t qi = 0;
        for (; qi + 16 <= total_q; qi += 16) {
            float32x4_t a0 = vld1q_f32(x + qi);
            float32x4_t a1 = vld1q_f32(x + qi + 4);
            float32x4_t a2 = vld1q_f32(x + qi + 8);
            float32x4_t a3 = vld1q_f32(x + qi + 12);
            uint32x4_t u0 = vcvtaq_u32_f32(a0);
            uint32x4_t u1 = vcvtaq_u32_f32(a1);
            uint32x4_t u2 = vcvtaq_u32_f32(a2);
            uint32x4_t u3 = vcvtaq_u32_f32(a3);
            uint16x8_t s01 = vcombine_u16(vqmovn_u32(u0), vqmovn_u32(u1));
            uint16x8_t s23 = vcombine_u16(vqmovn_u32(u2), vqmovn_u32(u3));
            uint8x16_t b = vcombine_u8(vqmovn_u16(s01), vqmovn_u16(s23));
            vst1q_u8(qx_u8.data() + qi, b);
        }
        for (; qi < total_q; qi++) {
            float v = x[qi];
            qx_u8[qi] = (v < 0.f) ? 0 : (v > 255.f) ? 255 : (uint8_t)(v + 0.5f);
        }
        const uint8_t* cy = nt.data_u8.data();
        for (size_t m = 0; m < pq.M; m++) {
            const uint8_t* qxm = qx_u8.data() + m * pq.dsub;
            const uint8_t* cym = cy + m * pq.ksub * pq.dsub;
            float* disp = dis_table + m * pq.ksub;
            for (size_t k = 0; k < pq.ksub; k++) {
                disp[k] = (float)l2_u8_scalar(qxm, cym + k * pq.dsub, pq.dsub);
            }
        }
    }
}

} // namespace faiss

#endif // COMPILE_SIMD_ARM_NEON
