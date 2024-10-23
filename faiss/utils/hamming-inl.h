/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace faiss {

// BitstringWriter and BitstringReader functions
inline BitstringWriter::BitstringWriter(uint8_t* code, size_t code_size)
        : code(code), code_size(code_size), i(0) {
    memset(code, 0, code_size);
}

inline void BitstringWriter::write(uint64_t x, int nbit) {
    assert(code_size * 8 >= nbit + i);
    // nb of available bits in i / 8
    int na = 8 - (i & 7);

    if (nbit <= na) {
        code[i >> 3] |= x << (i & 7);
        i += nbit;
        return;
    } else {
        size_t j = i >> 3;
        code[j++] |= x << (i & 7);
        i += nbit;
        x >>= na;
        while (x != 0) {
            code[j++] |= x;
            x >>= 8;
        }
    }
}

inline BitstringReader::BitstringReader(const uint8_t* code, size_t code_size)
        : code(code), code_size(code_size), i(0) {}

inline uint64_t BitstringReader::read(int nbit) {
    assert(code_size * 8 >= nbit + i);
    // nb of available bits in i / 8
    int na = 8 - (i & 7);
    // get available bits in current byte
    uint64_t res = code[i >> 3] >> (i & 7);
    if (nbit <= na) {
        res &= (1 << nbit) - 1;
        i += nbit;
        return res;
    } else {
        int ofs = na;
        size_t j = (i >> 3) + 1;
        i += nbit;
        nbit -= na;
        while (nbit > 8) {
            res |= ((uint64_t)code[j++]) << ofs;
            ofs += 8;
            nbit -= 8; // TODO remove nbit
        }
        uint64_t last_byte = code[j];
        last_byte &= (1 << nbit) - 1;
        res |= last_byte << ofs;
        return res;
    }
}

/** This class maintains a list of best distances seen so far.
 *
 * Since the distances are in a limited range (0 to nbit), the
 * object maintains one list per possible distance, and fills
 * in only the n-first lists, such that the sum of sizes of the
 * n lists is below k.
 */
template <class HammingComputer>
struct HCounterState {
    int* counters;
    int64_t* ids_per_dis;

    HammingComputer hc;
    int thres;
    int count_lt;
    int count_eq;
    int k;

    HCounterState(
            int* counters,
            int64_t* ids_per_dis,
            const uint8_t* x,
            int d,
            int k)
            : counters(counters),
              ids_per_dis(ids_per_dis),
              hc(x, d / 8),
              thres(d + 1),
              count_lt(0),
              count_eq(0),
              k(k) {}

    void update_counter(const uint8_t* y, size_t j) {
        int32_t dis = hc.hamming(y);

        if (dis <= thres) {
            if (dis < thres) {
                ids_per_dis[dis * k + counters[dis]++] = j;
                ++count_lt;
                while (count_lt == k && thres > 0) {
                    --thres;
                    count_eq = counters[thres];
                    count_lt -= count_eq;
                }
            } else if (count_eq < k) {
                ids_per_dis[dis * k + count_eq++] = j;
                counters[dis] = count_eq;
            }
        }
    }
};

} // namespace faiss
