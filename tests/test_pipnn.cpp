/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/pipnn/HashPrune.h>
#include <faiss/impl/pipnn/Partition.h>
#include <faiss/impl/pipnn/kernels.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/distances.h>
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

/*************************************************************
 * Numeric kernels
 *************************************************************/

namespace pipnn_kernel_test {

using faiss::pipnn::Ranking;

const Ranking kRankings[] = {Ranking::L2, Ranking::IP, Ranking::Angle};

faiss::AlignedTable<float, 64> random_table(size_t n, int64_t seed) {
    faiss::AlignedTable<float, 64> t(n);
    faiss::float_randn(t.get(), n, seed);
    return t;
}

// Scratch buffers start as NaN, so a BLAS that reads C with beta == 0 fails.
faiss::AlignedTable<float, 64> nan_table(size_t n) {
    faiss::AlignedTable<float, 64> t(n);
    std::fill(t.get(), t.get() + n, std::numeric_limits<float>::quiet_NaN());
    return t;
}

std::vector<float> norms_of(const float* X, size_t n, size_t d) {
    std::vector<float> norms(n);
    for (size_t i = 0; i < n; i++) {
        norms[i] = faiss::fvec_norm_L2sqr(X + i * d, d);
    }
    return norms;
}

double ref_rank(const float* p, const float* l, size_t d, Ranking ranking) {
    double dot = 0, ln = 0;
    for (size_t t = 0; t < d; t++) {
        dot += double(p[t]) * double(l[t]);
        ln += double(l[t]) * double(l[t]);
    }
    if (ranking == Ranking::L2) {
        return ln - 2 * dot;
    }
    return ranking == Ranking::IP ? -dot : -dot / std::sqrt(ln);
}

std::vector<uint16_t> run_stripe(
        const float* points,
        size_t S,
        const float* leaders,
        size_t L,
        size_t d,
        Ranking ranking,
        int fanout) {
    const std::vector<float> norms = norms_of(leaders, L, d);
    faiss::AlignedTable<float, 64> scratch = nan_table(S * L);
    std::vector<uint16_t> out(S * size_t(fanout), 0xFFFF);
    faiss::pipnn::stripe_assign(
            points,
            S,
            leaders,
            ranking == Ranking::IP ? nullptr : norms.data(),
            L,
            d,
            ranking,
            fanout,
            scratch.get(),
            out.data());
    return out;
}

// Full float64 distance matrix: L2 squared, or -dot for IP.
std::vector<double> ref_distances(
        const float* X,
        size_t n,
        size_t d,
        faiss::MetricType metric) {
    std::vector<double> ref(n * n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double acc = 0;
            for (size_t c = 0; c < d; c++) {
                const double a = X[i * d + c];
                const double b = X[j * d + c];
                acc += metric == faiss::METRIC_L2 ? (a - b) * (a - b) : a * b;
            }
            ref[i * n + j] = metric == faiss::METRIC_L2 ? acc : -acc;
        }
    }
    return ref;
}

void run_leaf(
        const float* X,
        size_t s,
        size_t d,
        faiss::MetricType metric,
        int k,
        std::vector<int32_t>& idx,
        std::vector<float>& dist) {
    const std::vector<float> norms = norms_of(X, s, d);
    faiss::AlignedTable<float, 64> gram = nan_table(s * s);
    idx.assign(s * size_t(k), 7); // garbage: every slot must be written
    dist.assign(s * size_t(k), 7.0f);
    faiss::pipnn::leaf_knn(
            X,
            norms.data(),
            s,
            d,
            metric,
            k,
            gram.get(),
            idx.data(),
            dist.data());
}

} // namespace pipnn_kernel_test

// Leaders must be distinct, with ranks within 1e-3 of the float64 reference.
TEST(PiPNNKernels, StripeMatchesFloat64Reference) {
    namespace pkt = pipnn_kernel_test;
    struct Case {
        size_t S, L, d;
        std::vector<int> fanouts;
    };
    const Case cases[] = {
            {9, 20, 16, {1, 10, 20}},
            {faiss::pipnn::kStripe, 100, 32, {33, 100}}};
    for (const Case& c : cases) {
        const auto points = pkt::random_table(c.S * c.d, 123);
        const auto leaders = pkt::random_table(c.L * c.d, 456);
        for (pkt::Ranking ranking : pkt::kRankings) {
            for (int fanout : c.fanouts) {
                SCOPED_TRACE(
                        testing::Message()
                        << "L=" << c.L << " ranking=" << int(ranking)
                        << " fanout=" << fanout);
                const std::vector<uint16_t> out = pkt::run_stripe(
                        points.get(),
                        c.S,
                        leaders.get(),
                        c.L,
                        c.d,
                        ranking,
                        fanout);
                for (size_t i = 0; i < c.S; i++) {
                    std::vector<double> rank(c.L);
                    for (size_t j = 0; j < c.L; j++) {
                        rank[j] = pkt::ref_rank(
                                points.get() + i * c.d,
                                leaders.get() + j * c.d,
                                c.d,
                                ranking);
                    }
                    std::vector<double> sorted = rank;
                    std::sort(sorted.begin(), sorted.end());
                    const uint16_t* row = out.data() + i * fanout;
                    ASSERT_EQ(
                            std::set<uint16_t>(row, row + fanout).size(),
                            size_t(fanout));
                    for (int r = 0; r < fanout; r++) {
                        ASSERT_LT(row[r], c.L);
                        EXPECT_NEAR(rank[row[r]], sorted[r], 1e-3);
                    }
                }
            }
        }
    }
}

// Leaders cycle through the 4 unit directions and the points are (0, 0) and
// (2, 0): small integers make every rank, and so every tie, exact.
TEST(PiPNNKernels, StripeTiesPreferLowerLeaderIndex) {
    namespace pkt = pipnn_kernel_test;
    const size_t L = 40;
    const float dirs[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    faiss::AlignedTable<float, 64> leaders(L * 2);
    for (size_t j = 0; j < L; j++) {
        leaders[2 * j] = dirs[j % 4][0];
        leaders[2 * j + 1] = dirs[j % 4][1];
    }
    faiss::AlignedTable<float, 64> points(4);
    const float pts[4] = {0, 0, 2, 0};
    std::copy(pts, pts + 4, points.get());
    // (2, 0) ranks the +x leaders first, then the orthogonal ones, then -x.
    std::vector<uint16_t> order1;
    for (size_t first : {0, 1, 2}) {
        for (size_t j = first; j < L; j += first == 1 ? 2 : 4) {
            order1.push_back(uint16_t(j));
        }
    }
    for (pkt::Ranking ranking : pkt::kRankings) {
        for (int fanout : {10, 40}) {
            SCOPED_TRACE(
                    testing::Message()
                    << "ranking=" << int(ranking) << " fanout=" << fanout);
            std::vector<uint16_t> expected(2 * fanout);
            for (int r = 0; r < fanout; r++) {
                expected[r] = uint16_t(r);
                expected[fanout + r] = order1[r];
            }
            EXPECT_EQ(
                    pkt::run_stripe(
                            points.get(),
                            2,
                            leaders.get(),
                            L,
                            2,
                            ranking,
                            fanout),
                    expected);
        }
    }
}

// NaN ranks +inf: a NaN point still gets `fanout` leaders in index order, and
// a NaN leader is never chosen over a finite one.
TEST(PiPNNKernels, StripeNaNRanksLast) {
    namespace pkt = pipnn_kernel_test;
    const size_t S = 5, L = 40, d = 8;
    auto points = pkt::random_table(S * d, 7);
    auto leaders = pkt::random_table(L * d, 8);
    points[2 * d] = std::numeric_limits<float>::quiet_NaN();
    leaders[0] = std::numeric_limits<float>::quiet_NaN();
    for (pkt::Ranking ranking : pkt::kRankings) {
        for (int fanout : {4, 33}) {
            SCOPED_TRACE(
                    testing::Message()
                    << "ranking=" << int(ranking) << " fanout=" << fanout);
            const std::vector<uint16_t> out = pkt::run_stripe(
                    points.get(), S, leaders.get(), L, d, ranking, fanout);
            for (size_t i = 0; i < S; i++) {
                for (int r = 0; r < fanout; r++) {
                    const uint16_t leader = out[i * fanout + r];
                    if (i == 2) {
                        EXPECT_EQ(leader, r) << "NaN point, rank " << r;
                    } else {
                        EXPECT_NE(leader, 0) << "point " << i << " rank " << r;
                    }
                }
            }
        }
    }
}

// Leader 1 is zero and leader 3 underflows to a zero norm; with a positive
// dot product either would otherwise rank first at -inf under Angle.
TEST(PiPNNKernels, StripeAngleZeroLeaderRanksLast) {
    namespace pkt = pipnn_kernel_test;
    const size_t S = 5, L = 6, d = 4;
    faiss::AlignedTable<float, 64> points(S * d);
    faiss::float_rand(points.get(), S * d, 11);
    for (size_t i = 0; i < S * d; i++) {
        points[i] += 0.5f;
    }
    auto leaders = pkt::random_table(L * d, 12);
    for (size_t c = 0; c < d; c++) {
        leaders[1 * d + c] = 0;
        leaders[3 * d + c] = 1e-30f;
    }
    ASSERT_EQ(faiss::fvec_norm_L2sqr(leaders.get() + 3 * d, d), 0.0f);
    const std::vector<uint16_t> out = pkt::run_stripe(
            points.get(), S, leaders.get(), L, d, pkt::Ranking::Angle, int(L));
    for (size_t i = 0; i < S; i++) {
        EXPECT_EQ(out[i * L + 4], 1) << "point " << i;
        EXPECT_EQ(out[i * L + 5], 3) << "point " << i;
    }
}

// Exact distances: ties go to the lower index, and a NaN point is at +inf
// from everyone.
TEST(PiPNNKernels, LeafKnnExactCases) {
    namespace pkt = pipnn_kernel_test;
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<int32_t> idx;
    std::vector<float> dist;

    const float line[5] = {0, 1, 2, 3, 4};
    pkt::run_leaf(line, 5, 1, faiss::METRIC_L2, 2, idx, dist);
    EXPECT_EQ(idx, (std::vector<int32_t>{1, 2, 0, 2, 1, 3, 2, 4, 3, 2}));
    EXPECT_EQ(dist, (std::vector<float>{1, 4, 1, 1, 1, 1, 1, 1, 1, 4}));

    const float with_nan[8] = {
            0, 0, 1, 0, 0, 2, std::numeric_limits<float>::quiet_NaN(), 0};
    pkt::run_leaf(with_nan, 4, 2, faiss::METRIC_L2, 3, idx, dist);
    EXPECT_EQ(idx, (std::vector<int32_t>{1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2}));
    EXPECT_EQ(
            dist,
            (std::vector<float>{
                    1, 4, inf, 1, 5, inf, 4, 5, inf, inf, inf, inf}));
}

// The Gram entries of a copied row may round differently, so its L2 distance
// is a small value that must be clamped at 0, never negative.
TEST(PiPNNKernels, LeafL2CopiedRowIsNearestAndNonNegative) {
    namespace pkt = pipnn_kernel_test;
    const size_t s = 33, d = 24;
    const int k = 4;
    auto Y = pkt::random_table(s * d, 13);
    std::copy(Y.get() + 7 * d, Y.get() + 8 * d, Y.get() + 30 * d);
    std::vector<int32_t> idx;
    std::vector<float> dist;
    pkt::run_leaf(Y.get(), s, d, faiss::METRIC_L2, k, idx, dist);
    EXPECT_GE(*std::min_element(dist.begin(), dist.end()), 0.0f);
    EXPECT_EQ(idx[7 * k], 30);
    EXPECT_EQ(idx[30 * k], 7);
    EXPECT_LE(dist[7 * k], 1e-4f);
}

// pairwise_distances matches the float64 reference, and leaf_knn equals the
// (distance, index)-smallest non-self entries of its rows, bitwise, then -1 /
// +inf padding. The final prune relies on this agreement.
TEST(PiPNNKernels, LeafKnnMatchesPairwise) {
    namespace pkt = pipnn_kernel_test;
    const size_t d = 24;
    for (faiss::MetricType metric :
         {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        for (size_t n : {1, 3, 65}) {
            const auto X = pkt::random_table(n * d, int64_t(100 + n));
            const std::vector<float> norms = pkt::norms_of(X.get(), n, d);
            faiss::AlignedTable<float, 64> D = pkt::nan_table(n * n);
            faiss::pipnn::pairwise_distances(
                    X.get(), norms.data(), n, d, metric, D.get());
            const std::vector<double> ref =
                    pkt::ref_distances(X.get(), n, d, metric);
            for (size_t i = 0; i < n * n; i++) {
                EXPECT_EQ(D[i], D[(i % n) * n + i / n]); // symmetric
                EXPECT_NEAR(D[i], ref[i], 1e-3);
            }
            for (int k : {2, 5}) {
                SCOPED_TRACE(
                        testing::Message() << "metric=" << int(metric)
                                           << " n=" << n << " k=" << k);
                std::vector<int32_t> want_idx(n * k, -1);
                std::vector<float> want_dist(
                        n * k, std::numeric_limits<float>::infinity());
                for (size_t i = 0; i < n; i++) {
                    std::vector<std::pair<float, int32_t>> row;
                    for (size_t j = 0; j < n; j++) {
                        if (j != i) {
                            row.emplace_back(D[i * n + j], int32_t(j));
                        }
                    }
                    std::sort(row.begin(), row.end());
                    for (size_t r = 0; r < std::min(size_t(k), row.size());
                         r++) {
                        want_dist[i * k + r] = row[r].first;
                        want_idx[i * k + r] = row[r].second;
                    }
                }
                std::vector<int32_t> idx;
                std::vector<float> dist;
                pkt::run_leaf(X.get(), n, d, metric, k, idx, dist);
                EXPECT_EQ(idx, want_idx);
                EXPECT_EQ(dist, want_dist);
            }
        }
    }
}

TEST(PiPNNKernels, RejectBadArguments) {
    using faiss::FaissException;
    const auto X = pipnn_kernel_test::random_table(3 * 4, 1);
    const std::vector<float> norms(3, 1.0f);
    faiss::AlignedTable<float, 64> scratch(3 * 3);
    std::vector<uint16_t> leaders_out(3 * 3);
    std::vector<int32_t> idx(3 * 2);
    std::vector<float> dist(3 * 2);

    // 3 points against 3 leaders (the same rows), d = 4.
    auto stripe = [&](const float* nrm, size_t L, int fanout) {
        faiss::pipnn::stripe_assign(
                X.get(),
                3,
                X.get(),
                nrm,
                L,
                4,
                faiss::pipnn::Ranking::L2,
                fanout,
                scratch.get(),
                leaders_out.data());
    };
    EXPECT_THROW(stripe(norms.data(), 3, 0), FaissException);
    EXPECT_THROW(stripe(norms.data(), 3, 4), FaissException); // > L
    EXPECT_THROW(stripe(nullptr, 3, 1), FaissException);
    EXPECT_THROW(stripe(norms.data(), 65536, 1), FaissException); // > uint16

    auto leaf = [&](const float* nrm, faiss::MetricType m, int k) {
        faiss::pipnn::leaf_knn(
                X.get(),
                nrm,
                3,
                4,
                m,
                k,
                scratch.get(),
                idx.data(),
                dist.data());
    };
    EXPECT_THROW(leaf(norms.data(), faiss::METRIC_L1, 2), FaissException);
    EXPECT_THROW(leaf(norms.data(), faiss::METRIC_L2, 0), FaissException);
    EXPECT_THROW(leaf(nullptr, faiss::METRIC_L2, 2), FaissException);
    EXPECT_THROW(
            faiss::pipnn::pairwise_distances(
                    X.get(),
                    norms.data(),
                    3,
                    4,
                    faiss::METRIC_L1,
                    scratch.get()),
            FaissException);
}

/*************************************************************
 * PiPNN storage access and Randomized Ball Carving
 *************************************************************/

namespace pipnn_partition_test {

std::vector<float> gaussian_data(size_t n, size_t d, int64_t seed) {
    std::vector<float> x(n * d);
    faiss::float_randn(x.data(), x.size(), seed);
    return x;
}

/// x_i = (1 + 10 i / n) * u for one direction u: under max-IP every point ranks
/// leaders by norm, so all points pick the same `fanout` leaders.
std::vector<float> collinear_data(size_t n, size_t d, int64_t seed) {
    std::vector<float> u(d);
    faiss::float_rand(u.data(), d, seed);
    std::vector<float> x(n * d);
    for (size_t i = 0; i < n; i++) {
        const float scale = 1.0f + 10.0f * float(i) / float(n);
        for (size_t j = 0; j < d; j++) {
            x[i * d + j] = scale * u[j];
        }
    }
    return x;
}

struct FlatData {
    faiss::IndexFlat index;

    FlatData(const std::vector<float>& x, size_t d, faiss::MetricType metric)
            : index(d, metric) {
        index.add(x.size() / d, x.data());
    }
};

const uint64_t kSeed = faiss::pipnn::mix_seed(1234, 0);

faiss::pipnn::PartitionParams small_params(faiss::MetricType metric) {
    faiss::pipnn::PartitionParams p;
    p.c_max = 64;
    p.c_min = 16;
    p.metric = metric;
    return p;
}

/// Sets the OpenMP thread count for its lifetime, then restores it.
struct ScopedOmpThreads {
    const int saved = omp_get_max_threads();
    explicit ScopedOmpThreads(int n) {
        omp_set_num_threads(n);
    }
    ~ScopedOmpThreads() {
        omp_set_num_threads(saved);
    }
};

using Leaves = std::vector<std::vector<int32_t>>;

/// Leaves are non-empty, strictly increasing, at most c_max long, and cover
/// every point.
void expect_valid_cover(const Leaves& leaves, size_t n, int c_max);

/// Runs partition_and_visit, checks that its leaves are a valid cover, and
/// returns them sorted.
Leaves visit_all(
        const faiss::Index& storage,
        const faiss::pipnn::PartitionParams& p,
        uint64_t seed,
        faiss::pipnn::PartitionStats* stats = nullptr) {
    std::mutex mu;
    Leaves leaves;
    faiss::pipnn::partition_and_visit(
            storage,
            p,
            seed,
            [&](const int32_t* ids, size_t s) {
                std::vector<int32_t> leaf(ids, ids + s);
                std::lock_guard<std::mutex> lock(mu);
                leaves.push_back(std::move(leaf));
            },
            stats);
    std::sort(leaves.begin(), leaves.end());
    expect_valid_cover(leaves, size_t(storage.ntotal), p.c_max);
    return leaves;
}

void expect_valid_cover(const Leaves& leaves, size_t n, int c_max) {
    std::vector<char> seen(n, 0);
    for (const std::vector<int32_t>& leaf : leaves) {
        ASSERT_FALSE(leaf.empty());
        ASSERT_LE(leaf.size(), size_t(c_max));
        for (size_t i = 0; i < leaf.size(); i++) {
            ASSERT_GE(leaf[i], 0);
            ASSERT_LT(size_t(leaf[i]), n);
            if (i > 0) {
                ASSERT_LT(leaf[i - 1], leaf[i]);
            }
            seen[leaf[i]] = 1;
        }
    }
    for (size_t i = 0; i < n; i++) {
        ASSERT_TRUE(seen[i]) << "point " << i << " is in no leaf";
    }
}

/// Fraction of points that share a leaf with their nearest other point (L2), or
/// with the other point of largest inner product (IP).
double fraction_with_neighbour(
        const Leaves& leaves,
        const std::vector<float>& x,
        size_t d,
        faiss::MetricType metric) {
    const size_t n = x.size() / d;
    faiss::IndexFlat index(d, metric);
    index.add(n, x.data());
    std::vector<float> D(2 * n);
    std::vector<faiss::idx_t> I(2 * n);
    index.search(n, x.data(), 2, D.data(), I.data());
    std::vector<std::vector<size_t>> leaves_of(n);
    for (size_t l = 0; l < leaves.size(); l++) {
        for (int32_t id : leaves[l]) {
            leaves_of[id].push_back(l);
        }
    }
    size_t hits = 0;
    for (size_t i = 0; i < n; i++) {
        const faiss::idx_t j =
                I[2 * i] == faiss::idx_t(i) ? I[2 * i + 1] : I[2 * i];
        const std::vector<size_t>& a = leaves_of[i];
        const std::vector<size_t>& b = leaves_of[j];
        if (std::find_first_of(a.begin(), a.end(), b.begin(), b.end()) !=
            a.end()) {
            hits++;
        }
    }
    return double(hits) / double(n);
}

} // namespace pipnn_partition_test

TEST(PiPNNStorage, GatherSQRowsMatchesReconstruct) {
    const int d = 12;
    const size_t n = 200;
    const std::vector<float> x = pipnn_partition_test::gaussian_data(n, d, 7);
    faiss::IndexScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.train(n, x.data());
    sq.add(n, x.data());
    const int32_t ids[3] = {5, 0, 199};
    faiss::AlignedTable<float, 64> out(3 * d);
    faiss::pipnn::gather_rows(sq, ids, 3, out.data());
    std::vector<float> expected(d);
    for (size_t i = 0; i < 3; i++) {
        sq.reconstruct(ids[i], expected.data());
        EXPECT_EQ(
                0,
                std::memcmp(
                        out.data() + i * d, expected.data(), sizeof(float) * d))
                << "row " << i;
    }
}

// Pinned values: changing any of them changes every graph.
TEST(PiPNNPartition, SeededHelpersGolden) {
    using faiss::pipnn::mix_seed;
    using faiss::pipnn::detail::sample_positions;
    EXPECT_EQ(mix_seed(1234, 1), 0x626e95467131d717ULL);

    std::vector<size_t> pos;
    sample_positions(100, 5, 42, pos);
    EXPECT_EQ(pos, (std::vector<size_t>{6, 45, 50, 84, 85}));
    sample_positions(10, 10, 7, pos);
    EXPECT_EQ(pos, (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(PiPNNPartition, ParameterHelpers) {
    using faiss::pipnn::Ranking;
    using faiss::pipnn::detail::leader_count;
    using faiss::pipnn::detail::ranking_for;
    faiss::pipnn::PartitionParams p;
    EXPECT_EQ(p.fanout_at(1), 3);
    EXPECT_EQ(p.fanout_at(2), 1);
    EXPECT_EQ(leader_count(1025, p, 0), 40u);       // 4 * fanout floor
    EXPECT_EQ(leader_count(100000, p, 0), 500u);    // ceil(P_samp * |P|)
    EXPECT_EQ(leader_count(10000000, p, 0), 1000u); // leader_cap
    EXPECT_EQ(leader_count(300, p, 1), 12u);
    EXPECT_EQ(leader_count(3, p, 5), 3u); // never more than |P|
    p.leader_cap = 7;
    EXPECT_EQ(leader_count(1025, p, 0), 7u);

    p.ip_partition_by_angle = true;
    EXPECT_EQ(ranking_for(p, false), Ranking::L2);
    p.metric = faiss::METRIC_INNER_PRODUCT;
    EXPECT_EQ(ranking_for(p, false), Ranking::Angle);
    EXPECT_EQ(ranking_for(p, true), Ranking::L2);
    p.ip_partition_by_angle = false;
    EXPECT_EQ(ranking_for(p, false), Ranking::IP);
}

TEST(PiPNNPartition, SizesAroundCMax) {
    namespace ppt = pipnn_partition_test;
    const size_t d = 8;
    const faiss::pipnn::PartitionParams p = ppt::small_params(faiss::METRIC_L2);

    ppt::FlatData empty({}, d, faiss::METRIC_L2);
    EXPECT_TRUE(ppt::visit_all(empty.index, p, 1).empty());

    const size_t n = size_t(p.c_max);
    ppt::FlatData one_leaf(ppt::gaussian_data(n, d, 5), d, faiss::METRIC_L2);
    std::vector<int32_t> all(n);
    for (size_t i = 0; i < n; i++) {
        all[i] = int32_t(i);
    }
    EXPECT_EQ(ppt::visit_all(one_leaf.index, p, 1), ppt::Leaves{all});

    ppt::FlatData above(ppt::gaussian_data(n + 1, d, 11), d, faiss::METRIC_L2);
    // One split whose small children are merged: no leaf is below c_min
    // (without merging, 25 of its 40 leaves are).
    for (const std::vector<int32_t>& leaf : ppt::visit_all(above.index, p, 9)) {
        EXPECT_GE(leaf.size(), size_t(p.c_min));
    }
}

// The root's 70,000 points span two 65,536-point counting-sort blocks, so the
// per-block counts and cursors must combine into valid children.
TEST(PiPNNPartition, RootSpansCountingSortBlocks) {
    namespace ppt = pipnn_partition_test;
    const size_t n = 70000, d = 2;
    ppt::FlatData data(ppt::gaussian_data(n, d, 31), d, faiss::METRIC_L2);
    EXPECT_FALSE(
            ppt::visit_all(
                    data.index, ppt::small_params(faiss::METRIC_L2), ppt::kSeed)
                    .empty());
}

TEST(PiPNNPartition, LeavesGroupNeighbours) {
    namespace ppt = pipnn_partition_test;
    const size_t n = 5000, d = 16;
    const std::vector<float> x = ppt::gaussian_data(n, d, 2024);
    ppt::FlatData l2(x, d, faiss::METRIC_L2);
    ppt::FlatData ip(x, d, faiss::METRIC_INNER_PRODUCT);
    faiss::IndexScalarQuantizer sq(
            d, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    sq.train(n, x.data());
    sq.add(n, x.data());
    struct Case {
        const char* name;
        const faiss::Index* storage;
        faiss::MetricType metric;
        int c_min;
        bool by_angle;
        double min_fraction; // measured: 0.986 for L2, 0.950 for IP
    };
    const Case cases[] = {
            {"L2", &l2.index, faiss::METRIC_L2, 16, false, 0.95},
            {"L2, no merging", &l2.index, faiss::METRIC_L2, 1, false, 0.95},
            {"IP", &ip.index, faiss::METRIC_INNER_PRODUCT, 16, false, 0.9},
            {"IP by angle",
             &ip.index,
             faiss::METRIC_INNER_PRODUCT,
             16,
             true,
             0.9},
            {"SQ8", &sq, faiss::METRIC_L2, 16, false, 0.95}};
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        faiss::pipnn::PartitionParams p = ppt::small_params(c.metric);
        p.c_min = c.c_min;
        p.ip_partition_by_angle = c.by_angle;
        const ppt::Leaves leaves = ppt::visit_all(*c.storage, p, ppt::kSeed);
        EXPECT_GE(
                ppt::fraction_with_neighbour(leaves, x, d, c.metric),
                c.min_fraction);
    }
}

TEST(PiPNNPartition, StatsDescribeLeaves) {
    namespace ppt = pipnn_partition_test;
    const size_t d = 16;
    ppt::FlatData data(ppt::gaussian_data(5000, d, 2024), d, faiss::METRIC_L2);
    const faiss::pipnn::PartitionParams p = ppt::small_params(faiss::METRIC_L2);
    faiss::pipnn::PartitionStats stats;
    const ppt::Leaves leaves =
            ppt::visit_all(data.index, p, ppt::kSeed, &stats);
    size_t instances = 0;
    size_t max_leaf = 0;
    for (const std::vector<int32_t>& leaf : leaves) {
        instances += leaf.size();
        max_leaf = std::max(max_leaf, leaf.size());
    }
    EXPECT_EQ(stats.n_leaves, leaves.size());
    EXPECT_EQ(stats.n_point_instances, instances);
    EXPECT_EQ(stats.max_leaf, max_leaf);
    EXPECT_GE(stats.max_depth_seen, 3);
    size_t histogram_total = 0;
    for (size_t h : stats.leaf_size_histogram) {
        histogram_total += h;
    }
    EXPECT_EQ(histogram_total, stats.n_leaves);
    // Replicas rely on a different seed giving different leaves.
    EXPECT_FALSE(
            leaves ==
            ppt::visit_all(data.index, p, faiss::pipnn::mix_seed(1234, 1)));
}

TEST(PiPNNPartition, IPDegenerateChildUsesL2) {
    namespace ppt = pipnn_partition_test;
    const size_t d = 16;
    ppt::FlatData data(
            ppt::collinear_data(5000, d, 31), d, faiss::METRIC_INNER_PRODUCT);
    const faiss::pipnn::PartitionParams p =
            ppt::small_params(faiss::METRIC_INNER_PRODUCT);
    faiss::pipnn::PartitionStats stats;
    ppt::visit_all(data.index, p, ppt::kSeed, &stats);
    // The children of the 10 largest-norm leaders hold all points and are
    // flagged; their descendants rank by L2 and are not flagged again.
    EXPECT_EQ(stats.n_l2_repartitions, 10u);
}

TEST(PiPNNPartition, AllZeroDataUsesSliceFallback) {
    namespace ppt = pipnn_partition_test;
    // All products are 0, so every point picks leaders 0..9, whose children
    // equal the root and are cut into id slices.
    const size_t n = 3000;
    ppt::FlatData data(std::vector<float>(n, 0.0f), 1, faiss::METRIC_L2);
    faiss::pipnn::PartitionStats stats;
    ppt::visit_all(
            data.index,
            ppt::small_params(faiss::METRIC_L2),
            ppt::kSeed,
            &stats);
    EXPECT_EQ(stats.n_slice_fallbacks, 10u);
    EXPECT_EQ(stats.n_leaves, 10u * ((n + 63) / 64));
}

TEST(PiPNNPartition, DuplicateGroupTerminatesAndCovers) {
    namespace ppt = pipnn_partition_test;
    const size_t n = 3000, d = 16;
    std::vector<float> x = ppt::gaussian_data(n, d, 77);
    std::fill(x.begin() + 1000 * d, x.begin() + 1500 * d, 10.0f);
    ppt::FlatData data(x, d, faiss::METRIC_L2);
    const faiss::pipnn::PartitionParams p = ppt::small_params(faiss::METRIC_L2);
    faiss::pipnn::PartitionStats stats;
    ppt::visit_all(data.index, p, ppt::kSeed, &stats);
    // The group is sliced once a child equals its parent (depth 7 here), well
    // before the depth limit.
    EXPECT_LT(stats.max_depth_seen, p.max_depth);
}

TEST(PiPNNPartition, DepthLimitPrecedesL2Repartition) {
    namespace ppt = pipnn_partition_test;
    // 10 root children hold all points; with max_depth = 1 the depth limit
    // applies before the L2 re-partition rule, so none is flagged.
    const size_t n = 5000, d = 16;
    ppt::FlatData data(
            ppt::collinear_data(n, d, 31), d, faiss::METRIC_INNER_PRODUCT);
    faiss::pipnn::PartitionParams p =
            ppt::small_params(faiss::METRIC_INNER_PRODUCT);
    p.max_depth = 1;
    faiss::pipnn::PartitionStats stats;
    ppt::visit_all(data.index, p, ppt::kSeed, &stats);
    EXPECT_EQ(stats.n_l2_repartitions, 0u);
    EXPECT_EQ(stats.n_slice_fallbacks, 10u);
}

TEST(PiPNNPartition, WavesDoNotChangeLeaves) {
    namespace ppt = pipnn_partition_test;
    // Waves that carry L2 re-partition flags.
    const size_t n = 5000, d = 16;
    ppt::FlatData ip(
            ppt::collinear_data(n, d, 31), d, faiss::METRIC_INNER_PRODUCT);
    faiss::pipnn::PartitionParams p_ip =
            ppt::small_params(faiss::METRIC_INNER_PRODUCT);
    const ppt::Leaves one_wave = ppt::visit_all(ip.index, p_ip, ppt::kSeed);
    p_ip.wave_budget = 3000;
    faiss::pipnn::PartitionStats ip_stats;
    EXPECT_TRUE(
            one_wave == ppt::visit_all(ip.index, p_ip, ppt::kSeed, &ip_stats));
    EXPECT_GT(ip_stats.n_waves, 1u);
}

namespace pipnn_partition_test {

struct AlwaysInterrupt : faiss::InterruptCallback {
    bool want_interrupt() override {
        return true;
    }
};

/// Clears the global interrupt callback when the test scope ends.
struct InterruptGuard {
    ~InterruptGuard() {
        faiss::InterruptCallback::clear_instance();
    }
};

} // namespace pipnn_partition_test

TEST(PiPNNPartition, InterruptStopsWithin64Items) {
    namespace ppt = pipnn_partition_test;
    // All-zero data gives one region of 470 slice leaves. With one thread the
    // master polls after items 0, 64, ...; the interrupt is set while visiting
    // item 9, so items 10..64 still run.
    ppt::FlatData data(std::vector<float>(3000, 0.0f), 1, faiss::METRIC_L2);
    const faiss::pipnn::PartitionParams p = ppt::small_params(faiss::METRIC_L2);
    const size_t install_at = 10;
    std::string what;
    size_t visited = 0;
    {
        ppt::InterruptGuard guard;
        ppt::ScopedOmpThreads one_thread(1);
        try {
            faiss::pipnn::partition_and_visit(
                    data.index,
                    p,
                    ppt::kSeed,
                    [&](const int32_t*, size_t) {
                        if (++visited == install_at) {
                            faiss::InterruptCallback::instance.reset(
                                    new ppt::AlwaysInterrupt());
                        }
                    },
                    nullptr);
        } catch (const faiss::FaissException& e) {
            what = e.what();
        }
    }
    EXPECT_NE(what.find("interrupted"), std::string::npos) << what;
    EXPECT_EQ(visited - install_at, 55u);
}
