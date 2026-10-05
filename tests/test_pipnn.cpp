/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include <faiss/impl/FaissException.h>
#include <faiss/impl/pipnn/HashPrune.h>
#include <faiss/utils/random.h>

using faiss::pipnn::distance_key;
using faiss::pipnn::Hyperplanes;
using faiss::pipnn::kNoFarthest;
using faiss::pipnn::ReservoirRef;
using faiss::pipnn::residual_hash;
using faiss::pipnn::Slot;
using faiss::pipnn::slot_less;

TEST(PiPNNDistanceKey, SpecialValues) {
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(distance_key(0.0f), uint16_t(0x8000));
    EXPECT_EQ(distance_key(-0.0f), distance_key(0.0f));
    EXPECT_EQ(distance_key(1.0f), uint16_t(0xBF80));
    EXPECT_EQ(distance_key(-1.0f), uint16_t(0x407F));
    EXPECT_EQ(distance_key(inf), uint16_t(0xFF80));
    EXPECT_EQ(
            distance_key(std::numeric_limits<float>::quiet_NaN()),
            distance_key(inf));
}

TEST(PiPNNDistanceKey, Monotone) {
    // Strictly increasing over all non-NaN bf16 values.
    std::vector<float> grid;
    for (uint32_t b = 0; b < 65536; b++) {
        const uint32_t bits = b << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        if (!std::isnan(f)) {
            grid.push_back(f);
        }
    }
    std::sort(grid.begin(), grid.end());
    for (size_t i = 1; i < grid.size(); i++) {
        if (grid[i - 1] != grid[i]) { // skips the -0, +0 pair
            ASSERT_LT(distance_key(grid[i - 1]), distance_key(grid[i]))
                    << grid[i - 1] << " vs " << grid[i];
        }
    }

    // Non-decreasing over arbitrary floats, which bf16 rounding maps between
    // grid points.
    std::vector<float> v;
    faiss::SplitMix64RandomGenerator rng(42);
    for (int i = 0; i < 100000; i++) {
        const uint32_t bits = static_cast<uint32_t>(rng.next());
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        if (!std::isnan(f)) {
            v.push_back(f);
        }
    }
    std::sort(v.begin(), v.end());
    for (size_t i = 1; i < v.size(); i++) {
        ASSERT_LE(distance_key(v[i - 1]), distance_key(v[i]))
                << v[i - 1] << " vs " << v[i];
    }
}

namespace {

struct TestReservoir {
    std::vector<Slot> slots;
    uint8_t count = 0;
    uint8_t farthest = kNoFarthest;
    ReservoirRef ref;

    explicit TestReservoir(int capacity)
            : slots(capacity), ref{slots.data(), &count, &farthest, capacity} {}
    TestReservoir(const TestReservoir&) = delete;
    TestReservoir& operator=(const TestReservoir&) = delete;

    bool offer(int32_t id, uint16_t hash, uint16_t key) {
        return ref.offer(id, hash, key);
    }
};

using SlotTuple = std::tuple<int32_t, uint16_t, uint16_t>; // (id, hash, key)

std::vector<SlotTuple> contents(const TestReservoir& r) {
    std::vector<SlotTuple> out;
    out.reserve(r.count);
    for (int i = 0; i < r.count; i++) {
        out.emplace_back(r.slots[i].id, r.slots[i].hash, r.slots[i].key);
    }
    return out;
}

/// Checks the ReservoirRef invariant documented in HashPrune.h.
void check_invariant(const TestReservoir& r) {
    ASSERT_LE(int(r.count), r.ref.capacity);
    std::set<int32_t> ids;
    for (int i = 0; i < r.count; i++) {
        if (i > 0) {
            ASSERT_LT(r.slots[i - 1].hash, r.slots[i].hash);
        }
        ASSERT_TRUE(ids.insert(r.slots[i].id).second);
    }
    if (r.farthest != kNoFarthest) {
        ASSERT_LT(int(r.farthest), int(r.count));
        const Slot& f = r.slots[r.farthest];
        for (int i = 0; i < r.count; i++) {
            if (i != r.farthest) {
                ASSERT_TRUE(
                        slot_less(r.slots[i].key, r.slots[i].id, f.key, f.id));
            }
        }
    }
}

} // namespace

TEST(PiPNNReservoir, Rules) {
    TestReservoir r(3);
    EXPECT_TRUE(r.offer(1, 30, 0x9000));
    EXPECT_TRUE(r.offer(2, 10, 0x9300));
    EXPECT_FALSE(r.offer(3, 10, 0x9400)); // bucket hit, larger key
    EXPECT_FALSE(r.offer(4, 10, 0x9300)); // bucket hit, equal key, larger id
    EXPECT_TRUE(r.offer(0, 10, 0x9300));  // bucket hit, equal key, smaller id
    EXPECT_TRUE(r.offer(5, 20, 0x9100));
    std::vector<SlotTuple> expected = {
            {0, 10, 0x9300}, {5, 20, 0x9100}, {1, 30, 0x9000}};
    EXPECT_EQ(contents(r), expected);
    EXPECT_FALSE(r.offer(6, 40, 0x9300)); // full: ties the max, larger id
    EXPECT_TRUE(r.offer(7, 40, 0x9200));  // full: evicts the max, id 0
    expected = {{5, 20, 0x9100}, {1, 30, 0x9000}, {7, 40, 0x9200}};
    EXPECT_EQ(contents(r), expected);
}

TEST(PiPNNReservoir, FarthestCache) {
    TestReservoir r(3);
    ASSERT_TRUE(r.offer(1, 10, 0x9000));
    ASSERT_TRUE(r.offer(2, 20, 0x9300));
    ASSERT_TRUE(r.offer(3, 30, 0x9100));
    EXPECT_FALSE(r.offer(4, 40, 0x9400)); // the scan caches the max, id 2
    EXPECT_EQ(int(r.farthest), 1);
    EXPECT_TRUE(r.offer(3, 30, 0x8F00)); // lowering another slot keeps it
    EXPECT_EQ(int(r.farthest), 1);
    EXPECT_TRUE(r.offer(2, 20, 0x8E00)); // lowering the max clears it
    EXPECT_EQ(int(r.farthest), int(kNoFarthest));
}

namespace {

struct Offer {
    int32_t id;
    uint16_t hash;
    uint16_t key;
};

/// Reference model of the reservoir (Lemma A.2 of the PiPNN paper, with key
/// ties decided by id): the min(capacity, B) slot_less-smallest bucket
/// minima, listed in hash order.
std::vector<SlotTuple> model_reservoir(
        const std::vector<Offer>& offers,
        int capacity) {
    std::map<uint16_t, std::pair<uint16_t, int32_t>> minima; // hash->(key,id)
    for (const Offer& o : offers) {
        auto it = minima.find(o.hash);
        if (it == minima.end()) {
            minima.emplace(o.hash, std::make_pair(o.key, o.id));
        } else if (slot_less(
                           o.key, o.id, it->second.first, it->second.second)) {
            it->second = std::make_pair(o.key, o.id);
        }
    }
    std::vector<std::tuple<uint16_t, int32_t, uint16_t>> ranked; // key,id,hash
    ranked.reserve(minima.size());
    for (const auto& [hash, key_id] : minima) {
        ranked.emplace_back(key_id.first, key_id.second, hash);
    }
    std::sort(ranked.begin(), ranked.end());
    if (ranked.size() > static_cast<size_t>(capacity)) {
        ranked.resize(capacity);
    }
    std::vector<SlotTuple> out;
    out.reserve(ranked.size());
    for (const auto& [key, id, hash] : ranked) {
        out.emplace_back(id, hash, key);
    }
    std::sort(
            out.begin(), out.end(), [](const SlotTuple& a, const SlotTuple& b) {
                return std::get<1>(a) < std::get<1>(b);
            });
    return out;
}

/// Ids are bucket * 4 + r (r < 4), so an id always lands in the same bucket,
/// as the residual hash guarantees in the build. Ids and keys repeat, so the
/// offers include re-offers and key ties.
std::vector<Offer> random_offers(
        size_t n,
        int n_buckets,
        int n_keys,
        faiss::SplitMix64RandomGenerator& rng) {
    std::vector<Offer> offers(n);
    for (Offer& o : offers) {
        const int bucket = static_cast<int>(rng.next() % n_buckets);
        o.hash = static_cast<uint16_t>(bucket * 37);
        o.id = static_cast<int32_t>(bucket * 4 + rng.next() % 4);
        o.key = static_cast<uint16_t>(0x8000 + rng.next() % n_keys);
    }
    return offers;
}

} // namespace

TEST(PiPNNReservoir, MatchesModelInAnyOrder) {
    faiss::SplitMix64RandomGenerator rng(77);
    for (int capacity : {1, 3, 8, 255}) {
        const int n_buckets = 2 * capacity + 8;
        for (int n_keys : {3, 4096}) {
            for (int trial = 0; trial < 3; trial++) {
                std::vector<Offer> offers =
                        random_offers(3 * n_buckets, n_buckets, n_keys, rng);
                const std::vector<SlotTuple> expected =
                        model_reservoir(offers, capacity);
                for (int shuffle = 0; shuffle < 10; shuffle++) {
                    SCOPED_TRACE(
                            testing::Message()
                            << "capacity " << capacity << " keys " << n_keys
                            << " trial " << trial << " shuffle " << shuffle);
                    for (size_t i = offers.size(); i > 1; i--) {
                        std::swap(offers[i - 1], offers[rng.next() % i]);
                    }
                    TestReservoir r(capacity);
                    for (const Offer& o : offers) {
                        const std::vector<SlotTuple> before = contents(r);
                        const bool changed = r.offer(o.id, o.hash, o.key);
                        ASSERT_EQ(changed, contents(r) != before);
                        ASSERT_NO_FATAL_FAILURE(check_invariant(r));
                    }
                    ASSERT_EQ(contents(r), expected);
                }
            }
        }
    }
}

TEST(PiPNNSketch, InvalidArgs) {
    EXPECT_THROW(Hyperplanes(0, 4, 1), faiss::FaissException);
    EXPECT_THROW(Hyperplanes(8, 0, 1), faiss::FaissException);
    EXPECT_THROW(Hyperplanes(8, 17, 1), faiss::FaissException);
}

TEST(PiPNNSketch, ProjectionIndependentOfBuffer) {
    const int d = 37;
    const int m = 12;
    Hyperplanes hp(d, m, 7);
    std::vector<float> h(size_t(m) * d);
    faiss::float_randn(h.data(), h.size(), 7);
    std::vector<float> x(d);
    faiss::float_randn(x.data(), d, 99);
    // The same vector at an odd offset of another buffer, with a misaligned
    // output, must give a bitwise-identical sketch.
    std::vector<float> shifted(3 + d);
    std::copy(x.begin(), x.end(), shifted.begin() + 3);
    float a[16];
    std::vector<float> b(17);
    hp.sketch(x.data(), a);
    hp.sketch(shifted.data() + 3, b.data() + 1);
    EXPECT_EQ(0, std::memcmp(a, b.data() + 1, sizeof(a)));
    for (int i = 0; i < m; i++) {
        double s = 0;
        for (int j = 0; j < d; j++) {
            s += double(x[j]) * double(h[size_t(i) * d + j]);
        }
        EXPECT_NEAR(a[i], s, 1e-4 * (1 + std::abs(s))) << i;
    }
    for (int i = m; i < 16; i++) {
        EXPECT_EQ(a[i], 0.0f) << i; // zero-padded hyperplanes
    }
}

TEST(PiPNNSketch, ResidualHash) {
    float p[16];
    float c[16];
    for (int i = 0; i < 16; i++) {
        p[i] = float(i);
        c[i] = i % 3 == 0 ? float(i) - 1 : float(i) + 1;
    }
    // Bit i is set iff i % 3 != 0.
    EXPECT_EQ(residual_hash(p, c, 1), uint16_t(0));
    EXPECT_EQ(residual_hash(p, c, 6), uint16_t(0x36));
    EXPECT_EQ(residual_hash(p, c, 16), uint16_t(0x6DB6));
    // A zero residual sets all m bits and none above.
    EXPECT_EQ(residual_hash(p, p, 12), uint16_t(0x0FFF));
}
