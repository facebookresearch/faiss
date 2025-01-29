/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <gtest/gtest.h>

using namespace ::testing;

float eval_codec(faiss::ResidualQuantizer* q, int nb, float* xb) {
    // Compute codes
    uint8_t* codes = new uint8_t[q->code_size * nb];
    std::cout << "code size: " << q->code_size << std::endl;
    q->compute_codes(xb, codes, nb);
    // Decode codes
    float* decoded = new float[nb * q->d];
    q->decode(codes, decoded, nb);
    // Compute reconstruction error
    float err = 0.0f;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < q->d; j++) {
            float diff = xb[i * q->d + j] - decoded[i * q->d + j];
            err = err + (diff * diff);
        }
    }
    delete[] codes;
    delete[] decoded;
    return err;
}

TEST(TestGpuResidualQuantizer, TestNcall) {
    int d = 32;
    int nt = 3000;
    int nb = 1000;
    // Assuming get_dataset_2 is a function that returns xt and xb
    std::vector<float> xt = faiss::gpu::randVecs(nt, d);
    std::vector<float> xb = faiss::gpu::randVecs(nb, d);
    faiss::ResidualQuantizer rq0(d, 4, 6);
    rq0.train(nt, xt.data());
    float err_rq0 = eval_codec(&rq0, nb, xb.data());
    faiss::ResidualQuantizer rq1(d, 4, 6);
    faiss::gpu::GpuProgressiveDimIndexFactory fac(1);
    rq1.assign_index_factory = &fac;
    rq1.train(nt, xt.data());
    ASSERT_GT(fac.ncall, 0);
    int ncall_train = fac.ncall;
    float err_rq1 = eval_codec(&rq1, nb, xb.data());
    ASSERT_GT(fac.ncall, ncall_train);
    std::cout << "Error RQ0: " << err_rq0 << ", Error RQ1: " << err_rq1
              << std::endl;
    ASSERT_TRUE(0.9 * err_rq0 < err_rq1);
    ASSERT_TRUE(err_rq1 < 1.1 * err_rq0);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
