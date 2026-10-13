/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/pipnn/HashPrune.h>

#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/random.h>

namespace faiss {
namespace pipnn {

uint16_t distance_key(float v) {
    // Normalized on the bit pattern, so -ffast-math cannot remove the checks.
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    if ((u & 0x7fffffffu) == 0) {
        u = 0; // -0 -> +0
    } else if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) {
        u = 0x7f800000u; // NaN -> +inf
    }
    std::memcpy(&v, &u, sizeof(v));
    // Scalar `encode_bf16` only: the SIMD path rounds differently, so keys
    // would depend on the CPU.
    const uint16_t b = encode_bf16(v);
    return (b & 0x8000) ? static_cast<uint16_t>(~b)
                        : static_cast<uint16_t>(b | 0x8000);
}

bool ReservoirRef::offer(int32_t id, uint16_t hash, uint16_t key) {
    const int n = *count;
    // A full reservoir rejects anything not below its maximum (rules 1 and 3).
    if (n == capacity && *farthest != kNoFarthest) {
        const Slot& f = slots[*farthest];
        if (!slot_less(key, id, f.key, f.id)) {
            return false;
        }
    }

    int lo = 0;
    int hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (slots[mid].hash < hash) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Rule 1: bucket hit. The cached farthest stays valid only when a
    // non-farthest slot is lowered.
    if (lo < n && slots[lo].hash == hash) {
        Slot& s = slots[lo];
        if (!slot_less(key, id, s.key, s.id)) {
            return false;
        }
        s.id = id;
        s.key = key;
        if (*farthest == lo) {
            *farthest = kNoFarthest;
        }
        return true;
    }

    // Rule 2: miss with room.
    if (n < capacity) {
        std::memmove(
                slots + lo + 1,
                slots + lo,
                sizeof(Slot) * static_cast<size_t>(n - lo));
        slots[lo] = Slot{id, hash, key};
        *count = static_cast<uint8_t>(n + 1);
        *farthest = kNoFarthest;
        return true;
    }

    // Rule 3: miss, full. Compare against the slot_less-maximum.
    int z = *farthest;
    if (z == kNoFarthest) {
        z = 0;
        for (int i = 1; i < n; i++) {
            if (slot_less(
                        slots[z].key, slots[z].id, slots[i].key, slots[i].id)) {
                z = i;
            }
        }
        *farthest = static_cast<uint8_t>(z);
    }
    const Slot& zs = slots[z];
    if (!slot_less(key, id, zs.key, zs.id)) {
        return false;
    }
    if (z < lo) {
        std::memmove(
                slots + z,
                slots + z + 1,
                sizeof(Slot) * static_cast<size_t>(lo - 1 - z));
        slots[lo - 1] = Slot{id, hash, key};
    } else {
        std::memmove(
                slots + lo + 1,
                slots + lo,
                sizeof(Slot) * static_cast<size_t>(z - lo));
        slots[lo] = Slot{id, hash, key};
    }
    *farthest = kNoFarthest;
    return true;
}

Hyperplanes::Hyperplanes(int d, int m, int64_t seed) : d(d), m(m) {
    FAISS_THROW_IF_NOT_FMT(d >= 1, "Hyperplanes: d must be >= 1, got %d", d);
    FAISS_THROW_IF_NOT_FMT(
            m >= 1 && m <= 16, "Hyperplanes: m must be in [1, 16], got %d", m);
    std::vector<float> h(static_cast<size_t>(m) * d);
    float_randn(h.data(), h.size(), seed);
    ht.assign(static_cast<size_t>(d) * 16, 0.0f);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
            ht[static_cast<size_t>(j) * 16 + i] =
                    h[static_cast<size_t>(i) * d + j];
        }
    }
}

FAISS_NOINLINE void Hyperplanes::sketch(const float* x, float* out16) const {
    // Local accumulators, j ascending: the result depends only on the values
    // of x, never on its buffer, alignment or caller.
    float acc[16] = {};
    const float* h = ht.data();
    for (int j = 0; j < d; j++) {
        const float xj = x[j];
        const float* hj = h + static_cast<size_t>(j) * 16;
        for (int i = 0; i < 16; i++) {
            acc[i] += xj * hj[i];
        }
    }
    std::memcpy(out16, acc, sizeof(acc));
}

uint16_t residual_hash(const float* sketch_p, const float* sketch_c, int m) {
    uint16_t h = 0;
    for (int i = 0; i < m; i++) {
        if (sketch_c[i] - sketch_p[i] >= 0) {
            h = static_cast<uint16_t>(h | (1u << i));
        }
    }
    return h;
}

} // namespace pipnn
} // namespace faiss
