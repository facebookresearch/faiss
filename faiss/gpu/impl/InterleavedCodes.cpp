/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/impl/InterleavedCodes.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace gpu {

inline uint8_t unpack5(int i, uint8_t vLower, uint8_t vUpper) {
    uint8_t v = 0;

    // lsb     ...    msb
    // 0: 0 0 0 0 0 1 1 1
    // 1: 1 1 2 2 2 2 2 3
    // 2: 3 3 3 3 4 4 4 4
    // 3: 4 5 5 5 5 5 6 6
    // 4: 6 6 6 7 7 7 7 7
    switch (i % 8) {
        case 0:
            // 5 lsbs of lower
            v = vLower & 0x1f;
            break;
        case 1:
            // 3 msbs of lower as v lsbs
            // 2 msbs of upper as v msbs
            v = (vLower >> 5) | ((vUpper & 0x3) << 3);
            break;
        case 2:
            // 5 of lower
            v = (vLower >> 2) & 0x1f;
            break;
        case 3:
            // 1 msbs of lower as v lsbs
            // 4 lsbs of upper as v msbs
            v = (vLower >> 7) | ((vUpper & 0xf) << 1);
            break;
        case 4:
            // 4 msbs of lower as v lsbs
            // 1 lsbs of upper as v msbs
            v = (vLower >> 4) | ((vUpper & 0x1) << 4);
            break;
        case 5:
            // 5 of lower
            v = (vLower >> 1) & 0x1f;
            break;
        case 6:
            // 2 msbs of lower as v lsbs
            // 3 lsbs of upper as v msbs
            v = (vLower >> 6) | ((vUpper & 0x7) << 2);
            break;
        case 7:
            // 5 of lower
            v = (vLower >> 3);
            break;
    }

    return v;
}

inline uint8_t unpack6(int i, uint8_t vLower, uint8_t vUpper) {
    uint8_t v = 0;

    switch (i % 4) {
        case 0:
            // 6 lsbs of lower
            v = vLower & 0x3f;
            break;
        case 1:
            // 2 msbs of lower as v lsbs
            // 4 lsbs of upper as v msbs
            v = (vLower >> 6) | ((vUpper & 0xf) << 2);
            break;
        case 2:
            // 4 msbs of lower as v lsbs
            // 2 lsbs of upper as v msbs
            v = (vLower >> 4) | ((vUpper & 0x3) << 4);
            break;
        case 3:
            // 6 msbs of lower
            v = (vLower >> 2);
            break;
    }

    return v;
}

std::vector<uint8_t> unpackNonInterleaved(
        std::vector<uint8_t> data,
        int numVecs,
        int dims,
        int bitsPerCode) {
    int srcVecSize = utils::divUp(dims * bitsPerCode, 8);
    FAISS_ASSERT(data.size() == numVecs * srcVecSize);

    if (bitsPerCode == 8 || bitsPerCode == 16 || bitsPerCode == 32) {
        // nothing to do
        return data;
    }

    // bit codes padded to whole bytes
    std::vector<uint8_t> out(numVecs * dims * utils::divUp(bitsPerCode, 8));

    if (bitsPerCode == 4) {
#pragma omp parallel for
        for (int i = 0; i < numVecs; ++i) {
            for (int j = 0; j < dims; ++j) {
                int srcIdx = i * srcVecSize + (j / 2);
                FAISS_ASSERT(srcIdx < data.size());

                uint8_t v = data[srcIdx];
                v = (j % 2 == 0) ? v & 0xf : v >> 4;

                out[i * dims + j] = v;
            }
        }
    } else if (bitsPerCode == 5) {
#pragma omp parallel for
        for (int i = 0; i < numVecs; ++i) {
            for (int j = 0; j < dims; ++j) {
                int lo = i * srcVecSize + (j * 5) / 8;
                int hi = lo + 1;

                FAISS_ASSERT(lo < data.size());
                FAISS_ASSERT(hi <= data.size());

                auto vLower = data[lo];
                auto vUpper = hi < data.size() ? data[hi] : 0;

                out[i * dims + j] = unpack5(j, vLower, vUpper);
            }
        }
    } else if (bitsPerCode == 6) {
#pragma omp parallel for
        for (int i = 0; i < numVecs; ++i) {
            for (int j = 0; j < dims; ++j) {
                int lo = i * srcVecSize + (j * 6) / 8;
                int hi = lo + 1;

                FAISS_ASSERT(lo < data.size());
                FAISS_ASSERT(hi <= data.size());

                auto vLower = data[lo];
                auto vUpper = hi < data.size() ? data[hi] : 0;

                out[i * dims + j] = unpack6(j, vLower, vUpper);
            }
        }
    } else {
        // unhandled
        FAISS_ASSERT(false);
    }

    return out;
}

template <typename T>
void unpackInterleavedWord(
        const T* in,
        T* out,
        int numVecs,
        int dims,
        int bitsPerCode) {
    int warpSize = getWarpSizeCurrentDevice();
    int wordsPerDimBlock = (size_t)warpSize * bitsPerCode / (8 * sizeof(T));
    int wordsPerBlock = wordsPerDimBlock * dims;
    int numBlocks = utils::divUp(numVecs, warpSize);

#pragma omp parallel for
    for (int i = 0; i < numVecs; ++i) {
        int block = i / warpSize;
        FAISS_ASSERT(block < numBlocks);
        int lane = i % warpSize;

        for (int j = 0; j < dims; ++j) {
            int srcOffset = block * wordsPerBlock + j * wordsPerDimBlock + lane;
            out[i * dims + j] = in[srcOffset];
        }
    }
}

std::vector<uint8_t> unpackInterleaved(
        std::vector<uint8_t> data,
        int numVecs,
        int dims,
        int bitsPerCode) {
    int warpSize = getWarpSizeCurrentDevice();
    int bytesPerDimBlock = warpSize * bitsPerCode / 8;
    int bytesPerBlock = bytesPerDimBlock * dims;
    int numBlocks = utils::divUp(numVecs, warpSize);
    size_t totalSize = (size_t)bytesPerBlock * numBlocks;
    FAISS_ASSERT(data.size() == totalSize);

    // bit codes padded to whole bytes
    std::vector<uint8_t> out(numVecs * dims * utils::divUp(bitsPerCode, 8));

    if (bitsPerCode == 8) {
        unpackInterleavedWord<uint8_t>(
                data.data(), out.data(), numVecs, dims, bitsPerCode);
    } else if (bitsPerCode == 16) {
        unpackInterleavedWord<uint16_t>(
                (uint16_t*)data.data(),
                (uint16_t*)out.data(),
                numVecs,
                dims,
                bitsPerCode);
    } else if (bitsPerCode == 32) {
        unpackInterleavedWord<uint32_t>(
                (uint32_t*)data.data(),
                (uint32_t*)out.data(),
                numVecs,
                dims,
                bitsPerCode);
    } else if (bitsPerCode == 4) {
#pragma omp parallel for
        for (int i = 0; i < numVecs; ++i) {
            int block = i / warpSize;
            int lane = i % warpSize;

            int word = lane / 2;
            int subWord = lane % 2;

            for (int j = 0; j < dims; ++j) {
                auto v =
                        data[block * bytesPerBlock + j * bytesPerDimBlock +
                             word];

                v = (subWord == 0) ? v & 0xf : v >> 4;
                out[i * dims + j] = v;
            }
        }
    } else if (bitsPerCode == 5) {
#pragma omp parallel for
        for (int i = 0; i < numVecs; ++i) {
            int block = i / warpSize;
            int blockVector = i % warpSize;

            for (int j = 0; j < dims; ++j) {
                uint8_t* dimBlock =
                        &data[block * bytesPerBlock + j * bytesPerDimBlock];

                int lo = (blockVector * 5) / 8;
                int hi = lo + 1;

                FAISS_ASSERT(lo < bytesPerDimBlock);
                FAISS_ASSERT(hi <= bytesPerDimBlock);

                auto vLower = dimBlock[lo];
                auto vUpper = hi < bytesPerDimBlock ? dimBlock[hi] : 0;

                out[i * dims + j] = unpack5(blockVector, vLower, vUpper);
            }
        }
    } else if (bitsPerCode == 6) {
#pragma omp parallel for
        for (int i = 0; i < numVecs; ++i) {
            int block = i / warpSize;
            int blockVector = i % warpSize;

            for (int j = 0; j < dims; ++j) {
                uint8_t* dimBlock =
                        &data[block * bytesPerBlock + j * bytesPerDimBlock];

                int lo = (blockVector * 6) / 8;
                int hi = lo + 1;

                FAISS_ASSERT(lo < bytesPerDimBlock);
                FAISS_ASSERT(hi <= bytesPerDimBlock);

                auto vLower = dimBlock[lo];
                auto vUpper = hi < bytesPerDimBlock ? dimBlock[hi] : 0;

                out[i * dims + j] = unpack6(blockVector, vLower, vUpper);
            }
        }
    } else {
        // unimplemented
        FAISS_ASSERT(false);
    }

    return out;
}

inline uint8_t pack5(int i, uint8_t lo, uint8_t hi, uint8_t hi2) {
    FAISS_ASSERT((lo & 0x1f) == lo);
    FAISS_ASSERT((hi & 0x1f) == hi);
    FAISS_ASSERT((hi2 & 0x1f) == hi2);

    uint8_t v = 0;

    // lsb     ...    msb
    // 0: 0 0 0 0 0 1 1 1
    // 1: 1 1 2 2 2 2 2 3
    // 2: 3 3 3 3 4 4 4 4
    // 3: 4 5 5 5 5 5 6 6
    // 4: 6 6 6 7 7 7 7 7
    switch (i % 5) {
        case 0:
            // 5 msbs of lower as vOut lsbs
            // 3 lsbs of upper as vOut msbs
            v = (lo & 0x1f) | (hi << 5);
            break;
        case 1:
            // 2 msbs of lower as vOut lsbs
            // 5 lsbs of upper as vOut msbs
            // 1 lsbs of upper2 as vOut msb
            v = (lo >> 3) | (hi << 2) | (hi2 << 7);
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
            v = (lo >> 4) | (hi << 1) | (hi2 << 6);
            break;
        case 4:
            // 3 msbs of lower as vOut lsbs
            // 5 lsbs of upper as vOut msbs
            v = (lo >> 2) | (hi << 3);
            break;
    }

    return v;
}

inline uint8_t pack6(int i, uint8_t lo, uint8_t hi) {
    FAISS_ASSERT((lo & 0x3f) == lo);
    FAISS_ASSERT((hi & 0x3f) == hi);

    uint8_t v = 0;

    // lsb     ...    msb
    // 0: 0 0 0 0 0 0 1 1
    // 1: 1 1 1 1 2 2 2 2
    // 2: 2 2 3 3 3 3 3 3
    switch (i % 3) {
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

    return v;
}

std::vector<uint8_t> packNonInterleaved(
        std::vector<uint8_t> data,
        int numVecs,
        int dims,
        int bitsPerCode) {
    // bit codes padded to whole bytes
    FAISS_ASSERT(data.size() == numVecs * dims * utils::divUp(bitsPerCode, 8));

    if (bitsPerCode == 8 || bitsPerCode == 16 || bitsPerCode == 32) {
        // nothing to do, whole words are already where they need to be
        return data;
    }

    // bits packed into a whole number of bytes
    int bytesPerVec = utils::divUp(dims * bitsPerCode, 8);

    std::vector<uint8_t> out(numVecs * bytesPerVec);

    if (bitsPerCode == 4) {
#pragma omp parallel for
        for (int i = 0; i < numVecs; ++i) {
            for (int j = 0; j < bytesPerVec; ++j) {
                int dimLo = j * 2;
                int dimHi = dimLo + 1;
                FAISS_ASSERT(dimLo < dims);
                FAISS_ASSERT(dimHi <= dims);

                uint8_t lo = data[i * dims + dimLo];
                uint8_t hi = dimHi < dims ? data[i * dims + dimHi] : 0;

                out[i * bytesPerVec + j] = (hi << 4) | (lo & 0xf);
            }
        }
    } else if (bitsPerCode == 5) {
#pragma omp parallel for
        for (int i = 0; i < numVecs; ++i) {
            for (int j = 0; j < bytesPerVec; ++j) {
                int dimLo = (j * 8) / 5;
                int dimHi = dimLo + 1;
                int dimHi2 = dimHi + 1;
                FAISS_ASSERT(dimLo < dims);
                FAISS_ASSERT(dimHi <= dims);
                FAISS_ASSERT(dimHi <= dims + 1);

                uint8_t lo = data[i * dims + dimLo];
                uint8_t hi = dimHi < dims ? data[i * dims + dimHi] : 0;
                uint8_t hi2 = dimHi2 < dims ? data[i * dims + dimHi2] : 0;

                out[i * bytesPerVec + j] = pack5(j, lo, hi, hi2);
            }
        }
    } else if (bitsPerCode == 6) {
#pragma omp parallel for
        for (int i = 0; i < numVecs; ++i) {
            for (int j = 0; j < bytesPerVec; ++j) {
                int dimLo = (j * 8) / 6;
                int dimHi = dimLo + 1;
                FAISS_ASSERT(dimLo < dims);
                FAISS_ASSERT(dimHi <= dims);

                uint8_t lo = data[i * dims + dimLo];
                uint8_t hi = dimHi < dims ? data[i * dims + dimHi] : 0;

                out[i * bytesPerVec + j] = pack6(j, lo, hi);
            }
        }
    } else {
        // unhandled
        FAISS_ASSERT(false);
    }

    return out;
}

template <typename T>
void packInterleavedWord(
        const T* in,
        T* out,
        int numVecs,
        int dims,
        int bitsPerCode) {
    int warpSize = getWarpSizeCurrentDevice();
    int wordsPerDimBlock = (size_t)warpSize * bitsPerCode / (8 * sizeof(T));
    int wordsPerBlock = wordsPerDimBlock * dims;
    int numBlocks = utils::divUp(numVecs, warpSize);

    // We're guaranteed that all other slots not filled by the vectors present
    // are initialized to zero (from the vector constructor in packInterleaved)
#pragma omp parallel for
    for (int i = 0; i < numVecs; ++i) {
        int block = i / warpSize;
        FAISS_ASSERT(block < numBlocks);
        int lane = i % warpSize;

        for (int j = 0; j < dims; ++j) {
            int dstOffset = block * wordsPerBlock + j * wordsPerDimBlock + lane;
            out[dstOffset] = in[i * dims + j];
        }
    }
}

std::vector<uint8_t> packInterleaved(
        std::vector<uint8_t> data,
        int numVecs,
        int dims,
        int bitsPerCode) {
    int warpSize = getWarpSizeCurrentDevice();
    int bytesPerDimBlock = warpSize * bitsPerCode / 8;
    int bytesPerBlock = bytesPerDimBlock * dims;
    int numBlocks = utils::divUp(numVecs, warpSize);
    size_t totalSize = (size_t)bytesPerBlock * numBlocks;

    // bit codes padded to whole bytes
    FAISS_ASSERT(data.size() == numVecs * dims * utils::divUp(bitsPerCode, 8));

    // packs based on blocks
    std::vector<uint8_t> out(totalSize, 0);

    if (bitsPerCode == 8) {
        packInterleavedWord<uint8_t>(
                data.data(), out.data(), numVecs, dims, bitsPerCode);
    } else if (bitsPerCode == 16) {
        packInterleavedWord<uint16_t>(
                (uint16_t*)data.data(),
                (uint16_t*)out.data(),
                numVecs,
                dims,
                bitsPerCode);
    } else if (bitsPerCode == 32) {
        packInterleavedWord<uint32_t>(
                (uint32_t*)data.data(),
                (uint32_t*)out.data(),
                numVecs,
                dims,
                bitsPerCode);
    } else if (bitsPerCode == 4) {
#pragma omp parallel for
        for (int i = 0; i < numBlocks; ++i) {
            for (int j = 0; j < dims; ++j) {
                for (int k = 0; k < bytesPerDimBlock; ++k) {
                    int loVec = i * warpSize + k * 2;
                    int hiVec = loVec + 1;

                    uint8_t lo = loVec < numVecs ? data[loVec * dims + j] : 0;
                    uint8_t hi = hiVec < numVecs ? data[hiVec * dims + j] : 0;

                    out[i * bytesPerBlock + j * bytesPerDimBlock + k] =
                            (hi << 4) | (lo & 0xf);
                }
            }
        }
    } else if (bitsPerCode == 5) {
#pragma omp parallel for
        for (int i = 0; i < numBlocks; ++i) {
            for (int j = 0; j < dims; ++j) {
                for (int k = 0; k < bytesPerDimBlock; ++k) {
                    // What input vectors we are pulling from
                    int loVec = i * warpSize + (k * 8) / 5;
                    int hiVec = loVec + 1;
                    int hiVec2 = hiVec + 1;

                    uint8_t lo = loVec < numVecs ? data[loVec * dims + j] : 0;
                    uint8_t hi = hiVec < numVecs ? data[hiVec * dims + j] : 0;
                    uint8_t hi2 =
                            hiVec2 < numVecs ? data[hiVec2 * dims + j] : 0;

                    out[i * bytesPerBlock + j * bytesPerDimBlock + k] =
                            pack5(k, lo, hi, hi2);
                }
            }
        }
    } else if (bitsPerCode == 6) {
#pragma omp parallel for
        for (int i = 0; i < numBlocks; ++i) {
            for (int j = 0; j < dims; ++j) {
                for (int k = 0; k < bytesPerDimBlock; ++k) {
                    // What input vectors we are pulling from
                    int loVec = i * warpSize + (k * 8) / 6;
                    int hiVec = loVec + 1;

                    uint8_t lo = loVec < numVecs ? data[loVec * dims + j] : 0;
                    uint8_t hi = hiVec < numVecs ? data[hiVec * dims + j] : 0;

                    out[i * bytesPerBlock + j * bytesPerDimBlock + k] =
                            pack6(k, lo, hi);
                }
            }
        }
    } else {
        // unimplemented
        FAISS_ASSERT(false);
    }

    return out;
}

} // namespace gpu
} // namespace faiss
