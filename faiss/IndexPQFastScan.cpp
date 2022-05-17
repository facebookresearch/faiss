/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexPQFastScan.h>

#include <limits.h>
#include <cassert>
#include <memory>

#include <omp.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/utils/utils.h>

namespace faiss {

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

IndexPQFastScan::IndexPQFastScan(
        int d,
        size_t M,
        size_t nbits,
        MetricType metric,
        int bbs)
        : pq(d, M, nbits) {
    init_fastscan(d, M, nbits, metric, bbs);
}

IndexPQFastScan::IndexPQFastScan(const IndexPQ& orig, int bbs) : pq(orig.pq) {
    init_fastscan(orig.d, pq.M, pq.nbits, orig.metric_type, bbs);
    ntotal = orig.ntotal;
    ntotal2 = roundup(ntotal, bbs);
    is_trained = orig.is_trained;
    orig_codes = orig.codes.data();

    // pack the codes
    codes.resize(ntotal2 * M2 / 2);
    pq4_pack_codes(orig.codes.data(), ntotal, M, ntotal2, bbs, M2, codes.get());
}

void IndexPQFastScan::train(idx_t n, const float* x) {
    if (is_trained) {
        return;
    }
    pq.train(n, x);
    is_trained = true;
}

void IndexPQFastScan::compute_codes(uint8_t* codes, idx_t n, const float* x)
        const {
    pq.compute_codes(x, codes, n);
}

void IndexPQFastScan::compute_float_LUT(float* lut, idx_t n, const float* x)
        const {
    if (metric_type == METRIC_L2) {
        pq.compute_distance_tables(n, x, lut);
    } else {
        pq.compute_inner_prod_tables(n, x, lut);
    }
}

void IndexPQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    pq.decode(bytes, x, n);
}

} // namespace faiss
