/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <sstream>
#include <vector>

TEST(TestCodePacking, NonInterleavedCodes_UnpackPack) {
    using namespace faiss::gpu;

    // We are fine using non-fixed seeds here, the results should be fully
    // deterministic
    auto seed = std::random_device()();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint8_t> dist;

    std::cout << "seed " << seed << "\n";

    for (auto bitsPerCode : {4, 5, 6, 8, 16, 32}) {
        for (auto dims : {1, 7, 8, 31, 32}) {
            for (auto numVecs : {1, 3, 4, 5, 6, 8, 31, 32, 33, 65}) {
                std::cout << bitsPerCode << " " << dims << " " << numVecs
                          << "\n";

                int srcVecSize = utils::divUp(dims * bitsPerCode, 8);
                std::vector<uint8_t> data(numVecs * srcVecSize);

                for (auto& v : data) {
                    v = dist(gen);
                }

                // currently unimplemented
                EXPECT_FALSE(bitsPerCode > 8 && bitsPerCode % 8 != 0);

                // Due to bit packing, mask out bits that should be zero based
                // on dimensions we shouldn't have present
                int vectorSizeBits = dims * bitsPerCode;
                int vectorSizeBytes = utils::divUp(vectorSizeBits, 8);
                int remainder = vectorSizeBits % 8;

                if (remainder > 0) {
                    uint8_t mask = 0xff >> (8 - remainder);

                    for (int i = 0; i < numVecs; ++i) {
                        int lastVecByte = (i + 1) * vectorSizeBytes - 1;
                        data[lastVecByte] &= mask;
                    }
                }

                auto up =
                        unpackNonInterleaved(data, numVecs, dims, bitsPerCode);
                auto p = packNonInterleaved(up, numVecs, dims, bitsPerCode);

                EXPECT_EQ(data, p);
            }
        }
    }
}

TEST(TestCodePacking, NonInterleavedCodes_PackUnpack) {
    using namespace faiss::gpu;

    // We are fine using non-fixed seeds here, the results should be fully
    // deterministic
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist;

    for (auto bitsPerCode : {4, 5, 6, 8, 16, 32}) {
        for (auto dims : {1, 7, 8, 31, 32}) {
            for (auto numVecs : {1, 3, 4, 5, 6, 8, 31, 32, 33, 65}) {
                std::cout << bitsPerCode << " " << dims << " " << numVecs
                          << "\n";

                std::vector<uint8_t> data(
                        numVecs * dims * utils::divUp(bitsPerCode, 8));

                // currently unimplemented
                EXPECT_FALSE(bitsPerCode > 8 && bitsPerCode % 8 != 0);

                // Mask out high bits we shouldn't have based on code size
                uint8_t mask =
                        bitsPerCode < 8 ? (0xff >> (8 - bitsPerCode)) : 0xff;

                for (auto& v : data) {
                    v = dist(gen) & mask;
                }

                auto p = packNonInterleaved(data, numVecs, dims, bitsPerCode);
                auto up = unpackNonInterleaved(p, numVecs, dims, bitsPerCode);

                EXPECT_EQ(data, up);
            }
        }
    }
}

TEST(TestCodePacking, InterleavedCodes_UnpackPack) {
    using namespace faiss::gpu;

    // We are fine using non-fixed seeds here, the results should be fully
    // deterministic
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist;

    for (auto bitsPerCode : {4, 5, 6, 8, 16, 32}) {
        for (auto dims : {1, 7, 8, 31, 32}) {
            for (auto numVecs : {1, 3, 4, 5, 6, 8, 31, 32, 33, 65}) {
                std::cout << bitsPerCode << " " << dims << " " << numVecs
                          << "\n";

                int warpSize = getWarpSizeCurrentDevice();
                int blocks = utils::divUp(numVecs, warpSize);
                int bytesPerDimBlock = warpSize * bitsPerCode / 8;
                int bytesPerBlock = bytesPerDimBlock * dims;
                int size = blocks * bytesPerBlock;

                std::vector<uint8_t> data(size);

                if (bitsPerCode == 8 || bitsPerCode == 16 ||
                    bitsPerCode == 32) {
                    int bytesPerCode = bitsPerCode / 8;

                    for (int i = 0; i < blocks; ++i) {
                        for (int j = 0; j < dims; ++j) {
                            for (int k = 0; k < warpSize; ++k) {
                                for (int l = 0; l < bytesPerCode; ++l) {
                                    int vec = i * warpSize + k;
                                    if (vec < numVecs) {
                                        data[i * bytesPerBlock +
                                             j * bytesPerDimBlock +
                                             k * bytesPerCode + l] = dist(gen);
                                    }
                                }
                            }
                        }
                    }
                } else if (bitsPerCode < 8) {
                    for (int i = 0; i < blocks; ++i) {
                        for (int j = 0; j < dims; ++j) {
                            for (int k = 0; k < bytesPerDimBlock; ++k) {
                                int loVec =
                                        i * warpSize + (k * 8) / bitsPerCode;
                                int hiVec = loVec + 1;
                                int hiVec2 = hiVec + 1;

                                uint8_t lo = loVec < numVecs ? dist(gen) &
                                                (0xff >> (8 - bitsPerCode))
                                                             : 0;
                                uint8_t hi = hiVec < numVecs ? dist(gen) &
                                                (0xff >> (8 - bitsPerCode))
                                                             : 0;
                                uint8_t hi2 = hiVec2 < numVecs ? dist(gen) &
                                                (0xff >> (8 - bitsPerCode))
                                                               : 0;

                                uint8_t v = 0;
                                if (bitsPerCode == 4) {
                                    v = lo | (hi << 4);
                                } else if (bitsPerCode == 5) {
                                    switch (k % 5) {
                                        case 0:
                                            // 5 msbs of lower as vOut lsbs
                                            // 3 lsbs of upper as vOut msbs
                                            v = (lo & 0x1f) | (hi << 5);
                                            break;
                                        case 1:
                                            // 2 msbs of lower as vOut lsbs
                                            // 5 lsbs of upper as vOut msbs
                                            // 1 lsbs of upper2 as vOut msb
                                            v = (lo >> 3) | (hi << 2) |
                                                    (hi2 << 7);
                                            break;
                                        case 2:
                                            // 4 msbs of lower as vOut lsbs
                                            // 4 lsbs of upper as vOut msbs
                                            v = (lo >> 1) | (hi << 4);
                                            break;
                                        case 3:
                                            // 1 msbs of lower as vOut lsbs
                                            // 5 lsbs of upper as vOut msbs
                                            // 2 lsbs of upper2 as vOut msb
                                            v = (lo >> 4) | (hi << 1) |
                                                    (hi2 << 6);
                                            break;
                                        case 4:
                                            // 3 msbs of lower as vOut lsbs
                                            // 5 lsbs of upper as vOut msbs
                                            v = (lo >> 2) | (hi << 3);
                                            break;
                                    }
                                } else if (bitsPerCode == 6) {
                                    switch (k % 3) {
                                        case 0:
                                            // 6 msbs of lower as vOut lsbs
                                            // 2 lsbs of upper as vOut msbs
                                            v = (lo & 0x3f) | (hi << 6);
                                            break;
                                        case 1:
                                            // 4 msbs of lower as vOut lsbs
                                            // 4 lsbs of upper as vOut msbs
                                            v = (lo >> 2) | (hi << 4);
                                            break;
                                        case 2:
                                            // 2 msbs of lower as vOut lsbs
                                            // 6 lsbs of upper as vOut msbs
                                            v = (lo >> 4) | (hi << 2);
                                            break;
                                    }
                                } else {
                                    // unimplemented
                                    EXPECT_TRUE(false);
                                }

                                data[i * bytesPerBlock + j * bytesPerDimBlock +
                                     k] = v;
                            }
                        }
                    }
                } else {
                    // unimplemented
                    EXPECT_TRUE(false);
                }

                auto up = unpackInterleaved(data, numVecs, dims, bitsPerCode);
                auto p = packInterleaved(up, numVecs, dims, bitsPerCode);

                EXPECT_EQ(data, p);
            }
        }
    }
}

TEST(TestCodePacking, InterleavedCodes_PackUnpack) {
    using namespace faiss::gpu;

    // We are fine using non-fixed seeds here, the results should be fully
    // deterministic
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist;

    for (auto bitsPerCode : {4, 5, 6, 8, 16, 32}) {
        for (auto dims : {1, 7, 8, 31, 32}) {
            for (auto numVecs : {1, 3, 4, 5, 6, 8, 31, 32, 33, 65}) {
                std::cout << bitsPerCode << " " << dims << " " << numVecs
                          << "\n";

                std::vector<uint8_t> data(
                        numVecs * dims * utils::divUp(bitsPerCode, 8));

                if (bitsPerCode == 8 || bitsPerCode == 16 ||
                    bitsPerCode == 32) {
                    for (auto& v : data) {
                        v = dist(gen);
                    }
                } else if (bitsPerCode < 8) {
                    uint8_t mask = 0xff >> (8 - bitsPerCode);

                    for (auto& v : data) {
                        v = dist(gen) & mask;
                    }
                } else {
                    // unimplemented
                    EXPECT_TRUE(false);
                }

                auto p = packInterleaved(data, numVecs, dims, bitsPerCode);
                auto up = unpackInterleaved(p, numVecs, dims, bitsPerCode);

                EXPECT_EQ(data, up);
            }
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
