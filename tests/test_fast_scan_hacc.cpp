/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/utils/simd_levels.h>

namespace {

using faiss::SIMDLevel;

struct RestoreSIMD {
    SIMDLevel saved = faiss::SIMDConfig::get_level();
    ~RestoreSIMD() {
        faiss::SIMDConfig::set_level(saved);
    }
};

std::vector<SIMDLevel> available_levels() {
    std::vector<SIMDLevel> levels;
    for (auto level :
         {faiss::SIMDConfig::get_level(),
          SIMDLevel::NONE,
          SIMDLevel::AVX2,
          SIMDLevel::AVX512,
          SIMDLevel::AVX512_VPOPCNT,
          SIMDLevel::ARM_NEON}) {
        if (faiss::SIMDConfig::is_simd_level_available(level) &&
            std::find(levels.begin(), levels.end(), level) == levels.end()) {
            levels.push_back(level);
        }
    }
    return levels;
}

/// Compare accumulate_hacc at every available SIMD level with the exact sums
/// computed from the unpacked codes and 16-bit tables.
void check_hacc(int nq, size_t ntotal, int nsq, bool max_entries, int seed) {
    SCOPED_TRACE(
            "nq=" + std::to_string(nq) + " ntotal=" + std::to_string(ntotal) +
            " nsq=" + std::to_string(nsq));
    std::mt19937 rng(seed);
    const size_t ntotal2 = (ntotal + 31) / 32 * 32;

    // flat codes: sub-quantizer 2k in the low nibble of byte k
    std::vector<uint8_t> codes(ntotal * nsq / 2);
    for (auto& c : codes) {
        c = rng();
    }
    std::vector<uint8_t> blocks(ntotal2 * nsq / 2);
    faiss::pq4_pack_codes(
            codes.data(), ntotal, nsq, ntotal2, 32, nsq, blocks.data());

    std::vector<uint16_t> lut(size_t(nq) * nsq * 16);
    for (auto& e : lut) {
        e = max_entries ? 65535 : uint16_t(rng());
    }
    std::vector<uint8_t> packed(size_t(nq) * 2 * nsq * 16);
    for (int q = 0; q < nq; q++) {
        std::vector<uint8_t> lo(nsq * 16), hi(nsq * 16);
        for (int i = 0; i < nsq * 16; i++) {
            lo[i] = lut[size_t(q) * nsq * 16 + i] & 0xff;
            hi[i] = lut[size_t(q) * nsq * 16 + i] >> 8;
        }
        uint8_t* dst = packed.data() + size_t(q) * 2 * nsq * 16;
        faiss::pq4_pack_LUT(1, nsq, lo.data(), dst);
        faiss::pq4_pack_LUT(1, nsq, hi.data(), dst + nsq * 16);
    }

    std::vector<int32_t> ref(nq * ntotal2, 0);
    for (int q = 0; q < nq; q++) {
        for (size_t i = 0; i < ntotal; i++) {
            int32_t sum = 0;
            for (int sq = 0; sq < nsq; sq++) {
                const uint8_t byte = codes[i * nsq / 2 + sq / 2];
                const int c = sq % 2 == 0 ? byte & 15 : byte >> 4;
                sum += lut[(size_t(q) * nsq + sq) * 16 + c];
            }
            ref[q * ntotal2 + i] = sum;
        }
    }

    RestoreSIMD restore;
    for (SIMDLevel level : available_levels()) {
        SCOPED_TRACE("level=" + faiss::to_string(level));
        faiss::SIMDConfig::set_level(level);
        std::vector<int32_t> accu(nq * ntotal2, -1);
        faiss::accumulate_hacc(
                nq, ntotal2, nsq, blocks.data(), packed.data(), accu.data());
        for (int q = 0; q < nq; q++) {
            for (size_t i = 0; i < ntotal; i++) {
                ASSERT_EQ(accu[q * ntotal2 + i], ref[q * ntotal2 + i])
                        << "query " << q << " element " << i;
            }
        }
    }
}

} // namespace

TEST(FastScanHacc, exact_sums) {
    int seed = 0;
    // segment boundaries of the kernels (256 and 512 sub-quantizers) and
    // numbers of sub-quantizers that are not a multiple of 4
    for (int nsq : {2, 4, 6, 30, 254, 256, 258, 510, 512, 514, 1024}) {
        check_hacc(1, 32, nsq, false, seed++);
        check_hacc(3, 70, nsq, false, seed++);
        // every entry at 65535, the largest possible sums
        check_hacc(1, 32, nsq, true, seed++);
    }
}
