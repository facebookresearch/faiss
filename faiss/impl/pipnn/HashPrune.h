/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace faiss {
namespace pipnn {

/// One reservoir slot: candidate id, residual-LSH bucket, order-preserving key.
struct Slot {
    int32_t id;
    uint16_t hash;
    uint16_t key;
};
static_assert(sizeof(Slot) == 8, "PiPNN reservoir slots must be 8 bytes");

/// "No cached farthest slot"; never a slot index since capacity <= 255.
constexpr uint8_t kNoFarthest = 0xFF;

/// Monotone 16-bit key of a distance, which may be negative, from its scalar
/// bf16 encoding. NaN maps to +inf and -0 to +0.
uint16_t distance_key(float v);

/// Strict (key, id) order. Deciding key ties by id makes the reservoir
/// independent of the offer order.
inline bool slot_less(uint16_t ka, int32_t ia, uint16_t kb, int32_t ib) {
    return ka < kb || (ka == kb && ia < ib);
}

/// One point's reservoir in caller-owned storage; not thread-safe. Slots are
/// sorted by distinct hash, and *farthest caches the slot_less-maximum.
struct ReservoirRef {
    Slot* slots;
    uint8_t* count;    // 0 when fresh
    uint8_t* farthest; // kNoFarthest when fresh or unknown
    int capacity;      // 1..255

    /// Keeps the smaller offer on a bucket hit, else inserts or evicts the
    /// maximum. An id must always come with the same hash. True iff changed.
    bool offer(int32_t id, uint16_t hash, uint16_t key);
};

/// m <= 16 Gaussian hyperplanes from float_randn(seed), stored transposed
/// and zero-padded: ht[j * 16 + i].
struct Hyperplanes {
    int d = 0;
    int m = 0;
    std::vector<float> ht; // d * 16

    Hyperplanes() = default;
    Hyperplanes(int d, int m, int64_t seed);

    /// out16[i] = sum over ascending j of x[j] * ht[j * 16 + i]. Not inlined,
    /// so all call sites produce bitwise-identical sketches.
    void sketch(const float* x, float* out16) const;
};

/// Bit i (i < m <= 16) is set iff sketch_c[i] - sketch_p[i] >= 0.
uint16_t residual_hash(const float* sketch_p, const float* sketch_c, int m);

} // namespace pipnn
} // namespace faiss
