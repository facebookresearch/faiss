/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissException.h>
#include <faiss/svs/IndexSVSFaissUtils.h>
#include <faiss/svs/IndexSVSFlat.h>
#include <gtest/gtest.h>

#include <vector>

// Verify that to_svs_metric throws FaissException (not abort) for unsupported
// metric types.

// Direct call: exercises the deserialization path where metric_type is read
// from a serialized stream and passed to to_svs_metric().
TEST(SVSMetric, UnsupportedMetricThrowsDirect) {
    EXPECT_THROW(faiss::to_svs_metric(faiss::METRIC_L1), faiss::FaissException);
    EXPECT_THROW(
            faiss::to_svs_metric(faiss::METRIC_Linf), faiss::FaissException);
    EXPECT_THROW(
            faiss::to_svs_metric(faiss::METRIC_Jaccard), faiss::FaissException);
}

// Supported metrics should not throw.
TEST(SVSMetric, SupportedMetricsSucceed) {
    EXPECT_NO_THROW(faiss::to_svs_metric(faiss::METRIC_L2));
    EXPECT_NO_THROW(faiss::to_svs_metric(faiss::METRIC_INNER_PRODUCT));
}

// Construction path: creating an IndexSVSFlat with an unsupported metric and
// calling add() triggers create_impl() -> to_svs_metric(). Verifies the throw
// propagates through the index construction API.
TEST(SVSMetric, IndexSVSFlatCreateImplUnsupportedMetricThrows) {
    faiss::IndexSVSFlat idx(4, faiss::METRIC_L1);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW(idx.add(1, data.data()), faiss::FaissException);
}
