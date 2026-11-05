/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexRowwiseInt8.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace {

std::vector<float> make_data(size_t n, size_t d, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::uniform_real_distribution<float> scale(0.1f, 3.0f);
    std::vector<float> x(n * d);
    for (size_t i = 0; i < n; i++) {
        float s = scale(rng);
        for (size_t j = 0; j < d; j++) {
            x[i * d + j] = normal(rng) * s;
        }
    }
    return x;
}

// exact L2 search on the int8 codes, in double precision; distances get
// - rounding_bias * d / 12 (s_q^2 + s_b^2)
void reference_l2(
        const faiss::IndexRowwiseInt8& index,
        const std::vector<int8_t>& qc,
        const std::vector<float>& qs,
        size_t nq,
        int k,
        std::vector<faiss::idx_t>& labels,
        double rounding_bias = 0) {
    size_t d = index.d;
    size_t nb = index.ntotal;
    std::vector<int8_t> bc(nb * d);
    std::vector<float> bs(nb);
    index.get_int8(0, nb, bc.data(), bs.data());
    labels.resize(nq * k);
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<double, faiss::idx_t>> dis(nb);
        for (size_t b = 0; b < nb; b++) {
            int64_t dot = 0, nqq = 0, nbb = 0;
            for (size_t j = 0; j < d; j++) {
                int64_t a = qc[q * d + j], c = bc[b * d + j];
                dot += a * c;
                nqq += a * a;
                nbb += c * c;
            }
            double sq = qs[q], sb = bs[b];
            double f = rounding_bias * d / 12.0; // zero rows: no error
            dis[b] = {
                    sq * sq * nqq + sb * sb * nbb - 2 * sq * sb * dot -
                            f *
                                    ((nqq > 0 ? sq * sq : 0) +
                                     (nbb > 0 ? sb * sb : 0)),
                    faiss::idx_t(b)};
        }
        std::partial_sort(dis.begin(), dis.begin() + k, dis.end());
        for (int r = 0; r < k; r++) {
            labels[q * k + r] = dis[r].second;
        }
    }
}

} // namespace

TEST(IndexRowwiseInt8, search_matches_exact_reference) {
    for (size_t d : {64, 50}) { // 50: scalar path (d not a multiple of 64)
        size_t nb = 2000, nq = 20;
        int k = 5;
        auto xb = make_data(nb, d, 1);
        auto xq = make_data(nq, d, 2);

        faiss::IndexRowwiseInt8 index(d);
        index.add(nb, xb.data());
        EXPECT_EQ(index.code_size, d + sizeof(float));

        std::vector<int8_t> qc(nq * d);
        std::vector<float> qs(nq);
        faiss::rowwise_int8_encode(d, nq, xq.data(), qc.data(), qs.data());

        std::vector<float> D(nq * k);
        std::vector<faiss::idx_t> I(nq * k), Iref;
        index.search(nq, xq.data(), k, D.data(), I.data());
        reference_l2(index, qc, qs, nq, k, Iref);
        EXPECT_EQ(I, Iref);

        // int8 queries give the same result
        std::vector<float> D8(nq * k);
        std::vector<faiss::idx_t> I8(nq * k);
        index.search_int8(nq, qc.data(), qs.data(), k, D8.data(), I8.data());
        EXPECT_EQ(I8, I);
        EXPECT_EQ(D8, D);
    }
}

TEST(IndexRowwiseInt8, encode_error_bound) {
    size_t d = 96, n = 100;
    auto x = make_data(n, d, 3);
    std::vector<int8_t> c(n * d);
    std::vector<float> s(n);
    faiss::rowwise_int8_encode(d, n, x.data(), c.data(), s.data());
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            float err = std::fabs(c[i * d + j] * s[i] - x[i * d + j]);
            EXPECT_LE(err, s[i] / 2 + 1e-6f);
        }
    }
}

TEST(IndexRowwiseInt8, write_read_round_trip) {
    size_t d = 64, nb = 500, nq = 10;
    int k = 4;
    auto xb = make_data(nb, d, 4);
    auto xq = make_data(nq, d, 5);
    faiss::IndexRowwiseInt8 index(d, faiss::METRIC_INNER_PRODUCT);
    index.add(nb, xb.data());

    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    faiss::VectorIOReader reader;
    reader.data = writer.data;
    std::unique_ptr<faiss::Index> back(faiss::read_index(&reader));
    auto* rw = dynamic_cast<faiss::IndexRowwiseInt8*>(back.get());
    ASSERT_NE(rw, nullptr);
    EXPECT_EQ(rw->metric_type, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(rw->ntotal, index.ntotal);
    EXPECT_EQ(
            0,
            std::memcmp(
                    rw->codes.data(), index.codes.data(), index.codes.size()));

    std::vector<float> Da(nq * k), Db(nq * k);
    std::vector<faiss::idx_t> Ia(nq * k), Ib(nq * k);
    index.search(nq, xq.data(), k, Da.data(), Ia.data());
    rw->search(nq, xq.data(), k, Db.data(), Ib.data());
    EXPECT_EQ(Ia, Ib);
    EXPECT_EQ(Da, Db);
}

TEST(IndexRowwiseInt8, rounding_bias) {
    size_t d = 64, nb = 2000, nq = 20;
    int k = 5;
    auto xb = make_data(nb, d, 6);
    auto xq = make_data(nq, d, 7);
    faiss::IndexRowwiseInt8 index(d);
    index.add(nb, xb.data());
    std::vector<int8_t> qc(nq * d);
    std::vector<float> qs(nq);
    faiss::rowwise_int8_encode(d, nq, xq.data(), qc.data(), qs.data());

    // neighbours chosen on the corrected distances
    faiss::SearchParametersRowwiseInt8 params;
    params.rounding_bias = 1;
    std::vector<float> D(nq * k), D8(nq * k);
    std::vector<faiss::idx_t> I(nq * k), I8(nq * k), Iref;
    index.search(nq, xq.data(), k, D.data(), I.data(), &params);
    reference_l2(index, qc, qs, nq, k, Iref, 1.0);
    EXPECT_EQ(I, Iref);
    index.search_int8(
            nq, qc.data(), qs.data(), k, D8.data(), I8.data(), &params);
    EXPECT_EQ(I8, I);
    EXPECT_EQ(D8, D);

    // rounding_bias = 0 (the default) is identical to no parameters
    std::vector<float> D0(nq * k), Dz(nq * k);
    std::vector<faiss::idx_t> I0(nq * k), Iz(nq * k);
    index.search(nq, xq.data(), k, D0.data(), I0.data());
    params.rounding_bias = 0;
    index.search(nq, xq.data(), k, Dz.data(), Iz.data(), &params);
    EXPECT_EQ(Dz, D0);
    EXPECT_EQ(Iz, I0);

    params.rounding_bias = -1;
    EXPECT_THROW(
            index.search(nq, xq.data(), k, Dz.data(), Iz.data(), &params),
            faiss::FaissException);
}

TEST(IndexRowwiseInt8, rounding_error_model) {
    // the model behind rounding_bias: the squared rounding error of a vector
    // is d / 12 * s^2 on average
    size_t d = 128, nb = 3000;
    auto xb = make_data(nb, d, 8);
    faiss::IndexRowwiseInt8 index(d);
    index.add(nb, xb.data());
    std::vector<float> rec(nb * d);
    index.reconstruct_n(0, nb, rec.data());
    std::vector<int8_t> c(nb * d);
    std::vector<float> s(nb);
    index.get_int8(0, nb, c.data(), s.data());
    double err = 0, model = 0;
    for (size_t i = 0; i < nb; i++) {
        for (size_t j = 0; j < d; j++) {
            double t = double(rec[i * d + j]) - xb[i * d + j];
            err += t * t;
        }
        model += d / 12.0 * double(s[i]) * s[i];
    }
    EXPECT_NEAR(err / model, 1.0, 0.05);
}

TEST(IndexRowwiseInt8, rounding_bias_ignored_for_inner_product) {
    // inner product has no rounding bias: the parameter changes nothing
    size_t d = 128, nb = 3000, nq = 30;
    auto xb = make_data(nb, d, 8);
    auto xq = make_data(nq, d, 9);
    faiss::SearchParametersRowwiseInt8 params;
    params.rounding_bias = 1;
    faiss::IndexRowwiseInt8 ip(d, faiss::METRIC_INNER_PRODUCT);
    ip.add(nb, xb.data());
    std::vector<float> Da(nq * 5), Db(nq * 5);
    std::vector<faiss::idx_t> Ia(nq * 5), Ib(nq * 5);
    ip.search(nq, xq.data(), 5, Da.data(), Ia.data());
    ip.search(nq, xq.data(), 5, Db.data(), Ib.data(), &params);
    EXPECT_EQ(Da, Db);
    EXPECT_EQ(Ia, Ib);
}
