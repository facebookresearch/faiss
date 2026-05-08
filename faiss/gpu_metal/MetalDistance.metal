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
TOPK_THREADGROUP_VARIANT(512)
TOPK_THREADGROUP_VARIANT(1024)
#undef TOPK_THREADGROUP_VARIANT
kernel void topk_threadgroup_2048(
    device const float* distances [[buffer(0)]],
    device float* outDistances [[buffer(1)]],
    device int* outIndices [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint qi [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint R = 16;
    constexpr uint CANDIDATES = TG_SIZE * R; // 4096
    threadgroup float tgDist[CANDIDATES];
    threadgroup int tgIdx[CANDIDATES];
    uint nq = params[0], nb = params[1], k = params[2], want_min = params[3];
    if (qi >= nq || k == 0) return;
    const device float* row = distances + qi * nb;
    uint kk = min(k, nb);
    uint K_out = min((uint)2048, kk);

    if (nb <= CANDIDATES) {
        for (uint i = tid; i < CANDIDATES; i += TG_SIZE) {
            if (i < nb) {
                tgDist[i] = row[i];
                tgIdx[i] = (int)i;
            } else {
                tgDist[i] = want_min ? 1e38f : -1e38f;
                tgIdx[i] = -1;
            }
        }
    } else {
        float localDist[R];
        int localIdx[R];
        uint localCount = 0;

        for (uint j = tid; j < nb; j += TG_SIZE) {
            float v = row[j];
            if (localCount < R) {
                uint pos = localCount;
                while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) {
                    localDist[pos] = localDist[pos-1];
                    localIdx[pos] = localIdx[pos-1];
                    pos--;
                }
                localDist[pos] = v;
                localIdx[pos] = (int)j;
                localCount++;
            } else {
                bool better = want_min ? (v < localDist[R-1]) : (v > localDist[R-1]);
                if (better) {
                    uint pos = R - 1;
                    while (pos > 0 && ((want_min && v < localDist[pos-1]) || (!want_min && v > localDist[pos-1]))) {
                        localDist[pos] = localDist[pos-1];
                        localIdx[pos] = localIdx[pos-1];
                        pos--;
                    }
                    localDist[pos] = v;
                    localIdx[pos] = (int)j;
                }
            }
        }

        for (uint i = 0; i < R; i++) {
            uint idx = tid * R + i;
            if (i < localCount) {
                tgDist[idx] = localDist[i];
                tgIdx[idx] = localIdx[i];
            } else {
                tgDist[idx] = want_min ? 1e38f : -1e38f;
                tgIdx[idx] = -1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CANDIDATES; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CANDIDATES; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CANDIDATES && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool partnerBetter = want_min ? (tgDist[partner] < tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]))
                                      : (tgDist[partner] > tgDist[idx] || (tgDist[partner] == tgDist[idx] && tgIdx[partner] < tgIdx[idx]));
                    bool idxBetter = want_min ? (tgDist[idx] < tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]))
                                  : (tgDist[idx] > tgDist[partner] || (tgDist[idx] == tgDist[partner] && tgIdx[idx] < tgIdx[partner]));
                    bool swap = ascending ? partnerBetter : idxBetter;
                    if (swap) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int ti = tgIdx[idx]; tgIdx[idx] = tgIdx[partner]; tgIdx[partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = tid; i < K_out; i += TG_SIZE) {
        outDistances[qi * k + i] = tgDist[i];
        outIndices[qi * k + i] = tgIdx[i];
    }
    for (uint i = tid; i < k - K_out; i += TG_SIZE) {
        outDistances[qi * k + K_out + i] = want_min ? 1e38f : -1e38f;
        outIndices[qi * k + K_out + i] = -1;
    }
}

// Bitonic merge kernel: merges two sorted lists of length K′ into one sorted list of length K′.
// Input: two buffers A and B, each (nq, K′) sorted (ascending for L2, descending for IP).
// Output: one buffer C (nq, K′) sorted, containing best K′ from A and B combined.
// Uses one threadgroup per query with bitonic merge network (compare-exchange pattern).
#define BITONIC_MERGE_TWO_SORTED_VARIANT(K) \

kernel void compute_norms(
    device const float* vectors [[buffer(0)]],
    device float*       norms   [[buffer(1)]],
    device const uint*  params  [[buffer(2)]],  // nb, d
    uint vid [[thread_position_in_grid]]
) {
    uint nb = params[0], d = params[1];
    if (vid >= nb) return;
    const device float* v = vectors + vid * d;
    float sum = 0.0f;
    uint d4 = d / 4;
    const device float4* v4 = (const device float4*)v;
    for (uint t = 0; t < d4; t++) {
        sum += dot(v4[t], v4[t]);
    }
    for (uint t = d4 * 4; t < d; t++) {
        sum += v[t] * v[t];
    }
    norms[vid] = sum;
}

// L2 distance using pre-computed vector (centroid) norms:
// dist[i][j] = queryNorm[i] + vecNorm[j] - 2 * dot(query[i], vec[j])
// We compute queryNorm on-the-fly (cheap, nq rows) but reuse vecNorms.
kernel void l2_with_norms(
    device const float* queries    [[buffer(0)]],
    device const float* vectors    [[buffer(1)]],
    device float*       distances  [[buffer(2)]],
    device const uint*  params     [[buffer(3)]],  // nq, nb, d
    device const float* vecNorms   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint nq = params[0], nb = params[1], d = params[2];
    uint i = gid.y;
    uint j = gid.x;
    if (i >= nq || j >= nb) return;

    const device float* q = queries + i * d;
    const device float* v = vectors + j * d;
    float dot_val = 0.0f;
    uint d4 = d / 4;
    const device float4* q4 = (const device float4*)q;
    const device float4* v4 = (const device float4*)v;
    for (uint t = 0; t < d4; t++) {
        dot_val += dot(q4[t], v4[t]);
    }
    for (uint t = d4 * 4; t < d; t++) {
        dot_val += q[t] * v[t];
    }

    // Compute query norm inline (avoids separate query norm buffer).
    float qNorm = 0.0f;
    for (uint t = 0; t < d4; t++) {
        qNorm += dot(q4[t], q4[t]);
    }
    for (uint t = d4 * 4; t < d; t++) {
        qNorm += q[t] * q[t];
    }

    distances[i * nb + j] = qNorm + vecNorms[j] - 2.0f * dot_val;
}

// ============================================================

// ============================================================
// IVF Flat scan — two-pass design (mirrors CUDA IVF):
//
//   Pass 1  ivf_scan_list
//       Grid: (nq * nprobe) threadgroups, 256 threads each.
//       Each threadgroup scans ONE inverted list for one query
//       and writes a per-list top-k.
//       Output: perListDist/perListIdx — (nq * nprobe * k).
//
//   Pass 2  ivf_merge_lists
//       Grid: (nq) threadgroups, 256 threads each.
//       Merges nprobe per-list top-k into final top-k per query.
//
// Both use float4 vectorised loads for memory throughput.
// ============================================================
// Shared params layout (device const uint*):
//   [0] nq   [1] d   [2] k   [3] nprobe   [4] want_min

inline bool ivf_better_int(float aDist, int aIdx, float bDist, int bIdx, uint want_min) {
    constexpr float tie_eps = 1e-6f;
    float delta = aDist - bDist;
    bool strictly = want_min ? (delta < -tie_eps) : (delta > tie_eps);
    if (strictly) return true;
    if (fabs(delta) <= tie_eps) return aIdx < bIdx;
    return false;
}

inline bool ivf_better_long(float aDist, long aIdx, float bDist, long bIdx, uint want_min) {
    constexpr float tie_eps = 1e-6f;
    float delta = aDist - bDist;
    bool strictly = want_min ? (delta < -tie_eps) : (delta > tie_eps);
    if (strictly) return true;
    if (fabs(delta) <= tie_eps) return aIdx < bIdx;
    return false;
}

kernel void ivf_scan_list(
    device const float*      queries       [[buffer(0)]],
    device const float*      codes         [[buffer(1)]],
    device const long*  ids           [[buffer(2)]],
    device const uint*  listOffset    [[buffer(3)]],
    device const uint*  listLength    [[buffer(4)]],
    device const int*   coarseAssign  [[buffer(5)]],
    device       float* perListDist   [[buffer(6)]],
    device       long*  perListIdx    [[buffer(7)]],
    device const uint*  params        [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 256;
    constexpr uint LOCAL_K = 4;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    // Cache query vector in threadgroup memory (read once from device, reused
    // by all threads for every vector in this list).
    threadgroup float tgQuery[2048]; // max d supported; only first d floats used
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sort internally by (dist, vecIdx) using 32-bit vecIdx for speed;
    // resolve to 64-bit ID only when writing output.
    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    uint d4 = d / 4;

    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        const device float* vvec = codes + vecIdx * d;

        float dist = 0.0f;
        if (want_min) {
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            const device float4* v4 = (const device float4*)vvec;
            for (uint t = 0; t < d4; t++) {
                float4 diff = q4[t] - v4[t];
                dist += dot(diff, diff);
            }
            for (uint t = d4 * 4; t < d; t++) {
                float diff = tgQuery[t] - vvec[t];
                dist += diff * diff;
            }
        } else {
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            const device float4* v4 = (const device float4*)vvec;
            for (uint t = 0; t < d4; t++) {
                dist += dot(q4[t], v4[t]);
            }
            for (uint t = d4 * 4; t < d; t++) {
                dist += tgQuery[t] * vvec[t];
            }
        }

        int vi = (int)vecIdx;

        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = ivf_better_int(
                        localDist[pos], localIdx[pos],
                        localDist[pos - 1], localIdx[pos - 1],
                        want_min);
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K; // 1024
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];

    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = ivf_better_int(
                            tgDist[partner], tgIdx[partner],
                            tgDist[idx], tgIdx[idx],
                            want_min);
                    bool iB = ivf_better_int(
                            tgDist[idx], tgIdx[idx],
                            tgDist[partner], tgIdx[partner],
                            want_min);
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write output: resolve 32-bit vecIdx → 64-bit ID from ids buffer.
    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

// ---- Small-list variant: 32 threads, 32-element bitonic sort ----
// Used when avg list size ≤ 32 (most threads in 256-thread version idle).
// Saves ~90% of bitonic sort barriers and threadgroup memory.
kernel void ivf_scan_list_small(
    device const float*      queries       [[buffer(0)]],
    device const float*      codes         [[buffer(1)]],
    device const long*  ids           [[buffer(2)]],
    device const uint*       listOffset    [[buffer(3)]],
    device const uint*       listLength    [[buffer(4)]],
    device const int*        coarseAssign  [[buffer(5)]],
    device       float*      perListDist   [[buffer(6)]],
    device       long*  perListIdx    [[buffer(7)]],
    device const uint*       params        [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE = 32;
    constexpr uint LOCAL_K = 1;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    uint lOff = listOffset[(uint)list_no];
    uint lLen = listLength[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1;
        }
        return;
    }

    // Query cache in threadgroup memory.
    threadgroup float tgQuery[2048];
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bestDist = sentinel;
    int   bestIdx  = -1;
    uint d4 = d / 4;

    for (uint li = tid; li < lLen; li += TG_SIZE) {
        uint vecIdx = lOff + li;
        const device float* vvec = codes + vecIdx * d;

        float dist = 0.0f;
        if (want_min) {
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            const device float4* v4 = (const device float4*)vvec;
            for (uint t = 0; t < d4; t++) {
                float4 diff = q4[t] - v4[t];
                dist += dot(diff, diff);
            }
            for (uint t = d4 * 4; t < d; t++) {
                float diff = tgQuery[t] - vvec[t];
                dist += diff * diff;
            }
        } else {
            const threadgroup float4* q4 = (const threadgroup float4*)tgQuery;
            const device float4* v4 = (const device float4*)vvec;
            for (uint t = 0; t < d4; t++) {
                dist += dot(q4[t], v4[t]);
            }
            for (uint t = d4 * 4; t < d; t++) {
                dist += tgQuery[t] * vvec[t];
            }
        }

        int vi = (int)vecIdx;

        bool better = ivf_better_int(dist, vi, bestDist, bestIdx, want_min);
        if (better) {
            bestDist = dist;
            bestIdx  = vi;
        }
    }

    constexpr uint CAND = TG_SIZE; // 32
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];
    tgDist[tid] = bestDist;
    tgIdx [tid] = bestIdx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            uint partner = tid ^ j;
            if (partner < CAND && partner > tid) {
                bool ascending = ((tid & k2) == 0);
                bool pB = ivf_better_int(
                        tgDist[partner], tgIdx[partner],
                        tgDist[tid], tgIdx[tid],
                        want_min);
                bool iB = ivf_better_int(
                        tgDist[tid], tgIdx[tid],
                        tgDist[partner], tgIdx[partner],
                        want_min);
                if (ascending ? pB : iB) {
                    float td = tgDist[tid]; tgDist[tid] = tgDist[partner]; tgDist[partner] = td;
                    int   ti = tgIdx [tid]; tgIdx [tid] = tgIdx [partner]; tgIdx [partner] = ti;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

// ---- Interleaved layout scan: 32-vector blocks, dimensions interleaved ----
// Memory layout per block: [v0d0 v1d0 .. v31d0] [v0d1 v1d1 .. v31d1] ...
// Each simdgroup (32 threads) processes one block cooperatively,
// achieving coalesced reads: all threads read the same dimension simultaneously.
kernel void ivf_scan_list_interleaved(
    device const float*      queries           [[buffer(0)]],
    device const float*      codes             [[buffer(1)]],
    device const long*       ids               [[buffer(2)]],
    device const uint*       listOffset        [[buffer(3)]],
    device const uint*       listLength        [[buffer(4)]],
    device const int*        coarseAssign      [[buffer(5)]],
    device       float*      perListDist       [[buffer(6)]],
    device       long*       perListIdx        [[buffer(7)]],
    device const uint*       params            [[buffer(8)]],
    device const uint*       ilCodesOffset     [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE   = 256;
    constexpr uint SIMD_W    = 32;
    constexpr uint NUM_SIMDS = TG_SIZE / SIMD_W; // 8
    constexpr uint LOCAL_K   = 4;

    uint nq       = params[0];
    uint d        = params[1];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];

    uint qi = tgid / nprobe;
    uint pi = tgid % nprobe;
    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint outBase = (qi * nprobe + pi) * k;

    int list_no = coarseAssign[qi * nprobe + pi];
    if (list_no < 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1L;
        }
        return;
    }

    uint idOff = listOffset[(uint)list_no];
    uint lLen  = listLength[(uint)list_no];
    uint cOff  = ilCodesOffset[(uint)list_no];
    if (lLen == 0) {
        for (uint i = tid; i < k; i += TG_SIZE) {
            perListDist[outBase + i] = sentinel;
            perListIdx [outBase + i] = -1L;
        }
        return;
    }

    uint numBlocks = (lLen + SIMD_W - 1) / SIMD_W;
    uint laneId    = tid % SIMD_W;
    uint simdId    = tid / SIMD_W;

    threadgroup float tgQuery[2048];
    const device float* qvecDev = queries + qi * d;
    for (uint i = tid; i < d; i += TG_SIZE) {
        tgQuery[i] = qvecDev[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localDist[LOCAL_K];
    int   localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1;
    }

    for (uint blk = simdId; blk < numBlocks; blk += NUM_SIMDS) {
        uint vecInList = blk * SIMD_W + laneId;
        bool valid = vecInList < lLen;

        const device float* blockPtr = codes + cOff + blk * SIMD_W * d;

        float dist = 0.0f;
        if (want_min) {
            for (uint dd = 0; dd < d; dd += 4) {
                float v0 = valid ? blockPtr[(dd + 0) * SIMD_W + laneId] : 0.0f;
                float v1 = valid ? blockPtr[(dd + 1) * SIMD_W + laneId] : 0.0f;
                float v2 = valid ? blockPtr[(dd + 2) * SIMD_W + laneId] : 0.0f;
                float v3 = valid ? blockPtr[(dd + 3) * SIMD_W + laneId] : 0.0f;
                float4 diff = float4(
                    tgQuery[dd]     - v0,
                    tgQuery[dd + 1] - v1,
                    tgQuery[dd + 2] - v2,
                    tgQuery[dd + 3] - v3);
                dist += dot(diff, diff);
            }
        } else {
            for (uint dd = 0; dd < d; dd += 4) {
                float v0 = valid ? blockPtr[(dd + 0) * SIMD_W + laneId] : 0.0f;
                float v1 = valid ? blockPtr[(dd + 1) * SIMD_W + laneId] : 0.0f;
                float v2 = valid ? blockPtr[(dd + 2) * SIMD_W + laneId] : 0.0f;
                float v3 = valid ? blockPtr[(dd + 3) * SIMD_W + laneId] : 0.0f;
                dist += tgQuery[dd]     * v0
                      + tgQuery[dd + 1] * v1
                      + tgQuery[dd + 2] * v2
                      + tgQuery[dd + 3] * v3;
            }
        }

        if (!valid) dist = sentinel;
        int vi = valid ? (int)(idOff + vecInList) : -1;

        bool better = want_min ? (dist < localDist[LOCAL_K-1])
                               : (dist > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = dist;
            localIdx [pos] = vi;
            while (pos > 0) {
                bool sw = ivf_better_int(
                        localDist[pos], localIdx[pos],
                        localDist[pos - 1], localIdx[pos - 1],
                        want_min);
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                int   ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    constexpr uint CAND = TG_SIZE * LOCAL_K; // 1024
    threadgroup float tgDist[CAND];
    threadgroup int   tgIdx [CAND];

    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = ivf_better_int(
                            tgDist[partner], tgIdx[partner],
                            tgDist[idx], tgIdx[idx],
                            want_min);
                    bool iB = ivf_better_int(
                            tgDist[idx], tgIdx[idx],
                            tgDist[partner], tgIdx[partner],
                            want_min);
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        int   ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        int vi = tgIdx[i];
        perListDist[outBase + i] = tgDist[i];
        perListIdx [outBase + i] = (vi < 0) ? -1L : ids[vi];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        perListDist[outBase + kk + i] = sentinel;
        perListIdx [outBase + kk + i] = -1L;
    }
}

// ---- Pass 2: merge nprobe per-list results per query ----
// Strided-scan pattern: each thread scans every TG_SIZE-th candidate across
// all nprobe×k entries, keeping its best LOCAL_K in registers. Then dump to
// threadgroup memory and bitonic-sort. Handles any nprobe×k without overflow.
kernel void ivf_merge_lists(
    device const float*      perListDist  [[buffer(0)]],
    device const long*       perListIdx   [[buffer(1)]],
    device       float*      outDistances [[buffer(2)]],
    device       long*       outIndices   [[buffer(3)]],
    device const uint*       params       [[buffer(4)]],
    uint qi  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    constexpr uint TG_SIZE  = 256;
    constexpr uint LOCAL_K  = 4;
    constexpr uint CAND     = TG_SIZE * LOCAL_K; // 1024

    uint nq       = params[0];
    uint k        = params[2];
    uint nprobe   = params[3];
    uint want_min = params[4];

    if (qi >= nq || k == 0) return;

    float sentinel = want_min ? 1e38f : -1e38f;
    uint totalCand = nprobe * k;
    uint inputBase = qi * totalCand;

    float localDist[LOCAL_K];
    long  localIdx [LOCAL_K];
    uint  localCount = 0;
    for (uint i = 0; i < LOCAL_K; i++) {
        localDist[i] = sentinel;
        localIdx [i] = -1L;
    }

    for (uint i = tid; i < totalCand; i += TG_SIZE) {
        float d = perListDist[inputBase + i];
        long  v = perListIdx [inputBase + i];

        bool better = want_min ? (d < localDist[LOCAL_K-1])
                               : (d > localDist[LOCAL_K-1]);
        if (localCount < LOCAL_K || better) {
            if (v < 0 && localCount >= LOCAL_K) continue;
            uint pos = (localCount < LOCAL_K) ? localCount : LOCAL_K - 1;
            localDist[pos] = d;
            localIdx [pos] = v;
            while (pos > 0) {
                bool sw = ivf_better_long(
                        localDist[pos], localIdx[pos],
                        localDist[pos - 1], localIdx[pos - 1],
                        want_min);
                if (!sw) break;
                float td = localDist[pos]; localDist[pos] = localDist[pos-1]; localDist[pos-1] = td;
                long  ti = localIdx [pos]; localIdx [pos] = localIdx [pos-1]; localIdx [pos-1] = ti;
                pos--;
            }
            if (localCount < LOCAL_K) localCount++;
        }
    }

    threadgroup float tgDist[CAND];
    threadgroup long  tgIdx [CAND];

    for (uint i = 0; i < LOCAL_K; i++) {
        tgDist[tid * LOCAL_K + i] = (i < localCount) ? localDist[i] : sentinel;
        tgIdx [tid * LOCAL_K + i] = (i < localCount) ? localIdx [i] : -1L;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k2 = 2; k2 <= CAND; k2 *= 2) {
        for (uint j = k2 >> 1; j > 0; j >>= 1) {
            for (uint idx = tid; idx < CAND; idx += TG_SIZE) {
                uint partner = idx ^ j;
                if (partner < CAND && partner > idx) {
                    bool ascending = ((idx & k2) == 0);
                    bool pB = ivf_better_long(
                            tgDist[partner], tgIdx[partner],
                            tgDist[idx], tgIdx[idx],
                            want_min);
                    bool iB = ivf_better_long(
                            tgDist[idx], tgIdx[idx],
                            tgDist[partner], tgIdx[partner],
                            want_min);
                    if (ascending ? pB : iB) {
                        float td = tgDist[idx]; tgDist[idx] = tgDist[partner]; tgDist[partner] = td;
                        long  ti = tgIdx [idx]; tgIdx [idx] = tgIdx [partner]; tgIdx [partner] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint kk = min(k, CAND);
    for (uint i = tid; i < kk; i += TG_SIZE) {
        outDistances[qi * k + i] = tgDist[i];
        outIndices  [qi * k + i] = tgIdx [i];
    }
    for (uint i = tid; i < k - kk; i += TG_SIZE) {
        outDistances[qi * k + kk + i] = sentinel;
        outIndices  [qi * k + kk + i] = -1L;
    }
}

// ============================================================
//  IVF Scalar Quantizer scan kernels
// ============================================================
//
// QT_8bit:  codes are uchar[d] per vector; decode: vmin[dim] + code * vdiff[dim]
//           SQ tables buffer layout: vmin[0..d-1], vdiff[0..d-1] (2*d floats)
// QT_4bit:  packed 2 values per byte, low/high nibble
// QT_6bit:  packed 4 values per 3 bytes
// QT_fp16:  codes are half[d] per vector; no SQ tables needed
//
// Both share the same top-k selection as ivf_scan_list.

inline float sq4_decode_component(device const uchar* code, uint i) {
    uchar b = code[i >> 1];
    uchar bits = (i & 1u) ? (b >> 4) : (b & 0x0Fu);
    return (float(bits) + 0.5f) / 15.0f;
}

inline float sq6_decode_component(device const uchar* code, uint i) {
    const device uchar* p = code + (i >> 2) * 3;
    uchar bits = 0;
    switch (i & 3u) {
        case 0:
            bits = p[0] & 0x3Fu;
            break;
        case 1:
            bits = (p[0] >> 6) | ((p[1] & 0x0Fu) << 2);
            break;
        case 2:
            bits = (p[1] >> 4) | ((p[2] & 0x03u) << 4);
            break;
        default:
            bits = p[2] >> 2;
            break;
    }
    return (float(bits) + 0.5f) / 63.0f;
}

