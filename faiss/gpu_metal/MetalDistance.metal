// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Metal Shading Language kernels for distance computation and top-k selection.
 *
 * Kernel organization:
 * - Distance kernels: l2_squared_matrix, ip_matrix (tiled GEMM-style)
 * - Top-k selection: topk_threadgroup_K (parallel bitonic sort, K <= 256)
 */

#include <metal_stdlib>
using namespace metal;

kernel void l2_squared_matrix(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_M = 32;
    constexpr uint TILE_N = 32;
    constexpr uint TILE_K = 16;
    constexpr uint TG_THREADS = 256;

    uint nq = params[0], nb = params[1], d = params[2];
    uint row0 = tgid.y * TILE_M;
    uint col0 = tgid.x * TILE_N;
    uint ty = ltid.y, tx = ltid.x;
    uint tid = ty * 16 + tx;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    threadgroup float tgQ[TILE_M * TILE_K];
    threadgroup float tgV[TILE_N * TILE_K];

    for (uint dk = 0; dk < d; dk += TILE_K) {
        uint kLen = min(TILE_K, d - dk);

        for (uint i = tid; i < TILE_M * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gRow = row0 + mr;
            tgQ[i] = (gRow < nq && mk < kLen) ? queries[gRow * d + dk + mk] : 0.0f;
        }
        for (uint i = tid; i < TILE_N * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gCol = col0 + mr;
            tgV[i] = (gCol < nb && mk < kLen) ? vectors[gCol * d + dk + mk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE_K; kk++) {
            float q0 = tgQ[(ty * 2) * TILE_K + kk];
            float q1 = tgQ[(ty * 2 + 1) * TILE_K + kk];
            float v0 = tgV[(tx * 2) * TILE_K + kk];
            float v1 = tgV[(tx * 2 + 1) * TILE_K + kk];
            float d00 = q0 - v0; acc00 += d00 * d00;
            float d01 = q0 - v1; acc01 += d01 * d01;
            float d10 = q1 - v0; acc10 += d10 * d10;
            float d11 = q1 - v1; acc11 += d11 * d11;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint r0 = row0 + ty * 2, r1 = r0 + 1;
    uint c0 = col0 + tx * 2, c1 = c0 + 1;
    if (r0 < nq && c0 < nb) distances[r0 * nb + c0] = acc00;
    if (r0 < nq && c1 < nb) distances[r0 * nb + c1] = acc01;
    if (r1 < nq && c0 < nb) distances[r1 * nb + c0] = acc10;
    if (r1 < nq && c1 < nb) distances[r1 * nb + c1] = acc11;
}

kernel void ip_matrix(
    device const float* queries [[buffer(0)]],
    device const float* vectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 ltid [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE_M = 32;
    constexpr uint TILE_N = 32;
    constexpr uint TILE_K = 16;
    constexpr uint TG_THREADS = 256;

    uint nq = params[0], nb = params[1], d = params[2];
    uint row0 = tgid.y * TILE_M;
    uint col0 = tgid.x * TILE_N;
    uint ty = ltid.y, tx = ltid.x;
    uint tid = ty * 16 + tx;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    threadgroup float tgQ[TILE_M * TILE_K];
    threadgroup float tgV[TILE_N * TILE_K];

    for (uint dk = 0; dk < d; dk += TILE_K) {
        uint kLen = min(TILE_K, d - dk);

        for (uint i = tid; i < TILE_M * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gRow = row0 + mr;
            tgQ[i] = (gRow < nq && mk < kLen) ? queries[gRow * d + dk + mk] : 0.0f;
        }
        for (uint i = tid; i < TILE_N * TILE_K; i += TG_THREADS) {
            uint mr = i / TILE_K, mk = i % TILE_K;
            uint gCol = col0 + mr;
            tgV[i] = (gCol < nb && mk < kLen) ? vectors[gCol * d + dk + mk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE_K; kk++) {
            float q0 = tgQ[(ty * 2) * TILE_K + kk];
            float q1 = tgQ[(ty * 2 + 1) * TILE_K + kk];
            float v0 = tgV[(tx * 2) * TILE_K + kk];
            float v1 = tgV[(tx * 2 + 1) * TILE_K + kk];
            acc00 += q0 * v0; acc01 += q0 * v1;
            acc10 += q1 * v0; acc11 += q1 * v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint r0 = row0 + ty * 2, r1 = r0 + 1;
    uint c0 = col0 + tx * 2, c1 = c0 + 1;
    if (r0 < nq && c0 < nb) distances[r0 * nb + c0] = acc00;
    if (r0 < nq && c1 < nb) distances[r0 * nb + c1] = acc01;
    if (r1 < nq && c0 < nb) distances[r1 * nb + c0] = acc10;
    if (r1 < nq && c1 < nb) distances[r1 * nb + c1] = acc11;
}

// ============================================================
//  Parallel threadgroup-based top-k (bitonic sort)
//  One threadgroup (256 threads) per query, 4 candidates per thread = 1024.
// ============================================================

#define TOPK_THREADGROUP_VARIANT(K) \
kernel void topk_threadgroup_##K( \
    device const float* distances [[buffer(0)]], \
    device float* outDistances [[buffer(1)]], \
    device int* outIndices [[buffer(2)]], \
    device const uint* params [[buffer(3)]], \
    uint qi [[threadgroup_position_in_grid]], \
    uint tid [[thread_position_in_threadgroup]] \
) { \
    constexpr uint TG_SIZE = 256; \
    constexpr uint R = 4; \
    constexpr uint CANDIDATES = TG_SIZE * R; \
    threadgroup float tgDist[CANDIDATES]; \
    threadgroup int tgIdx[CANDIDATES]; \
    uint nq = params[0], nb = params[1], k = params[2], want_min = params[3]; \
    if (qi >= nq || k == 0) return; \
    const device float* row = distances + qi * nb; \
    uint kk = min(k, nb); \
    uint K_out = min((uint)K, kk); \
    \
    float localDist[R]; \
    int localIdx[R]; \
    uint localCount = 0; \
    \
    for (uint j = tid; j < nb; j += TG_SIZE) { \
        float v = row[j]; \
        if (localCount < R) { \
            uint pos = localCount; \
            while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) { \
                localDist[pos] = localDist[pos-1]; \
                localIdx[pos] = localIdx[pos-1]; \
                pos--; \
            } \
            localDist[pos] = v; \
            localIdx[pos] = (int)j; \
            localCount++; \
        } else { \
            bool better = want_min ? (v < localDist[R-1]) : (v > localDist[R-1]); \
            if (better) { \
                uint pos = R - 1; \
                while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) { \
                    localDist[pos] = localDist[pos-1]; \
                    localIdx[pos] = localIdx[pos-1]; \
                    pos--; \
                } \
                localDist[pos] = v; \
                localIdx[pos] = (int)j; \
            } \
        } \
    } \
    \
    for (uint i = 0; i < R; i++) { \
        uint idx = tid * R + i; \
        if (i < localCount) { \
            tgDist[idx] = localDist[i]; \
            tgIdx[idx] = localIdx[i]; \
        } else { \
            tgDist[idx] = want_min ? 1e38f : -1e38f; \
            tgIdx[idx] = -1; \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    \
    for (uint k2 = 2; k2 <= CANDIDATES; k2 *= 2) { \
        for (uint j = k2 >> 1; j > 0; j >>= 1) { \
            for (uint idx = tid; idx < CANDIDATES; idx += TG_SIZE) { \
                uint partner = idx ^ j; \
                if (partner < CANDIDATES && partner > idx) { \
                    bool ascending = ((idx & k2) == 0); \
                    bool partnerBetter = want_min \
                        ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])) \
                        : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx])); \
                    bool idxBetter = want_min \
                        ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])) \
                        : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner])); \
                    bool swap = ascending ? partnerBetter : idxBetter; \
                    if (swap) { \
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td; \
                        int ti = tgIdx[idx]; tgIdx[idx] = tgIdx[partner]; tgIdx[partner] = ti; \
                    } \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    \
    for (uint i = tid; i < K_out; i += TG_SIZE) { \
        outDistances[qi * k + i] = tgDist[i]; \
        outIndices[qi * k + i] = tgIdx[i]; \
    } \
    for (uint i = tid; i < k - K_out; i += TG_SIZE) { \
        outDistances[qi * k + K_out + i] = want_min ? 1e38f : -1e38f; \
        outIndices[qi * k + K_out + i] = -1; \
    } \
}

TOPK_THREADGROUP_VARIANT(32)
TOPK_THREADGROUP_VARIANT(64)
TOPK_THREADGROUP_VARIANT(128)
TOPK_THREADGROUP_VARIANT(256)
#undef TOPK_THREADGROUP_VARIANT
