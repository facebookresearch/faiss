/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/utils/random.h>
#include <gtest/gtest.h>

/* This test creates a 3-level RCQ and performs a search on it.
 * Then it crops the RCQ to just the 2 first levels and verifies that
 * the 3-level vectors are in a subtree that was visited in the 2-level RCQ. */
TEST(RCQCropping, test_cropping) {
    size_t nq = 10, nt = 2000, nb = 1000, d = 32;

    using idx_t = faiss::idx_t;

    std::vector<float> buf((nq + nb + nt) * d);
    faiss::rand_smooth_vectors(nq + nb + nt, d, buf.data(), 1234);
    const float* xt = buf.data();
    const float* xb = xt + nt * d;
    const float* xq = xb + nb * d;

    std::vector<size_t> nbits = {5, 4, 4};
    faiss::ResidualCoarseQuantizer rcq(d, nbits);

    rcq.train(nt, xt);

    // the test below works only for beam size == nprobe
    rcq.set_beam_factor(1.0);

    // perform search
    int nprobe = 15;
    std::vector<faiss::idx_t> Iref(nq * nprobe);
    std::vector<float> Dref(nq * nprobe);
    rcq.search(nq, xq, nprobe, Dref.data(), Iref.data());

    // crop to the first 2 quantization levels
    int last_nbits = nbits.back();
    nbits.pop_back();
    faiss::ResidualCoarseQuantizer rcq_cropped(d, nbits);
    rcq_cropped.initialize_from(rcq);

    EXPECT_EQ(rcq_cropped.ntotal, rcq.ntotal >> last_nbits);

    // perform search
    std::vector<faiss::idx_t> Inew(nq * nprobe);
    std::vector<float> Dnew(nq * nprobe);
    rcq_cropped.search(nq, xq, nprobe, Dnew.data(), Inew.data());

    // these bits are in common between the two RCQs
    idx_t mask = ((idx_t)1 << rcq_cropped.rq.tot_bits) - 1;
    for (int q = 0; q < nq; q++) {
        for (int i = 0; i < nprobe; i++) {
            idx_t fine = Iref[q * nprobe + i];
            EXPECT_GE(fine, 0);
            bool found = false;

            // fine should be generated from a path that passes through coarse
            for (int j = 0; j < nprobe; j++) {
                idx_t coarse = Inew[q * nprobe + j];
                if ((fine & mask) == coarse) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found);
        }
    }
}

TEST(RCQCropping, search_params) {
    size_t nq = 10, nt = 2000, nb = 1000, d = 32;

    using idx_t = faiss::idx_t;

    std::vector<float> buf((nq + nb + nt) * d);
    faiss::rand_smooth_vectors(nq + nb + nt, d, buf.data(), 1234);
    const float* xt = buf.data();
    const float* xb = xt + nt * d;
    const float* xq = xb + nb * d;

    std::vector<size_t> nbits = {3, 6, 3};
    faiss::ResidualCoarseQuantizer quantizer(d, nbits);
    size_t ntotal = (size_t)1 << quantizer.rq.tot_bits;
    faiss::IndexIVFScalarQuantizer index(
            &quantizer, d, ntotal, faiss::ScalarQuantizer::QT_8bit);
    index.quantizer_trains_alone = true;

    index.train(nt, xt);
    index.add(nb, xb);

    index.nprobe = 10;

    int k = 4;
    float beam_factor_1 = 8.0;
    quantizer.set_beam_factor(beam_factor_1);
    std::vector<idx_t> I1(nq * k);
    std::vector<float> D1(nq * k);
    index.search(nq, xq, k, D1.data(), I1.data());

    // change from 8 to 1
    quantizer.set_beam_factor(1.0f);
    std::vector<idx_t> I2(nq * k);
    std::vector<float> D2(nq * k);
    index.search(nq, xq, k, D2.data(), I2.data());

    // make sure it changes the result
    EXPECT_NE(I1, I2);
    EXPECT_NE(D1, D2);

    // override the class level beam factor
    faiss::SearchParametersResidualCoarseQuantizer params1;
    params1.beam_factor = beam_factor_1;
    faiss::SearchParametersIVF params;
    params.nprobe = index.nprobe;
    params.quantizer_params = &params1;

    std::vector<idx_t> I3(nq * k);
    std::vector<float> D3(nq * k);
    index.search(nq, xq, k, D3.data(), I3.data(), &params);

    // make sure we find back the original results
    EXPECT_EQ(I1, I3);
    EXPECT_EQ(D1, D3);
}
