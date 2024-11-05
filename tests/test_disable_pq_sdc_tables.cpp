/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <random>

#include "faiss/Index.h"
#include "faiss/IndexHNSW.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "test_util.h"

pthread_mutex_t temp_file_mutex = PTHREAD_MUTEX_INITIALIZER;

TEST(IO, TestReadHNSWPQ_whenSDCDisabledFlagPassed_thenDisableSDCTable) {
    Tempfilename index_filename(&temp_file_mutex, "/tmp/faiss_TestReadHNSWPQ");
    int d = 32, n = 256;
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 100);
    std::vector<float> vectors(n * d);
    for (size_t i = 0; i < n * d; i++) {
        vectors[i] = u(rng);
    }

    // Build the index and write it to the temp file
    {
        std::unique_ptr<faiss::Index> index_writer(
                faiss::index_factory(d, "HNSW8,PQ4np", faiss::METRIC_L2));
        index_writer->train(n, vectors.data());
        index_writer->add(n, vectors.data());

        faiss::write_index(index_writer.get(), index_filename.c_str());
    }

    // Load index from disk. Confirm that the sdc table is equal to 0 when
    // disable sdc is set
    {
        std::unique_ptr<faiss::IndexHNSWPQ> index_reader_read_write(
                dynamic_cast<faiss::IndexHNSWPQ*>(
                        faiss::read_index(index_filename.c_str())));
        std::unique_ptr<faiss::IndexHNSWPQ> index_reader_sdc_disabled(
                dynamic_cast<faiss::IndexHNSWPQ*>(faiss::read_index(
                        index_filename.c_str(),
                        faiss::IO_FLAG_PQ_SKIP_SDC_TABLE)));

        ASSERT_NE(
                dynamic_cast<faiss::IndexPQ*>(index_reader_read_write->storage)
                        ->pq.sdc_table.size(),
                0);
        ASSERT_EQ(
                dynamic_cast<faiss::IndexPQ*>(
                        index_reader_sdc_disabled->storage)
                        ->pq.sdc_table.size(),
                0);
    }
}
