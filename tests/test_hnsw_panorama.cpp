/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSW.h>

#include <cstdio>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/clone_index.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/utils/random.h>

namespace {

// Generate random vectors for testing
std::vector<float> generate_random_vectors(int n, int d, int seed = 1234) {
    std::vector<float> data(n * d);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib(-1.0, 1.0);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = distrib(rng);
    }
    return data;
}

// Compute ground truth using brute force search
void compute_ground_truth(
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        int d,
        int k,
        faiss::idx_t* gt_I,
        float* gt_D) {
    faiss::IndexFlatL2 index(d);
    index.add(nb, xb);
    index.search(nq, xq, k, gt_D, gt_I);
}

// Compute recall@k
float compute_recall(
        const faiss::idx_t* gt_I,
        const faiss::idx_t* test_I,
        int nq,
        int k) {
    int correct = 0;
    int total = nq * k;

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            faiss::idx_t gt_id = gt_I[i * k + j];
            for (int l = 0; l < k; l++) {
                if (test_I[i * k + l] == gt_id) {
                    correct++;
                    break;
                }
            }
        }
    }
    return float(correct) / float(total);
}

} // anonymous namespace

TEST(IndexHNSWFlatPanorama, BasicConstruction) {
    int d = 128;
    int M = 16;
    int num_levels = 8;

    faiss::IndexHNSWFlatPanorama index(d, M, num_levels);

    EXPECT_EQ(index.d, d);
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.hnsw.nb_neighbors(0), 2 * M); // Level 0 has 2*M neighbors
    EXPECT_EQ(index.num_panorama_levels, num_levels);
    EXPECT_EQ(index.metric_type, faiss::METRIC_L2);
}

TEST(IndexHNSWFlatPanorama, AddAndSearch) {
    int d = 128;
    int nb = 1000;
    int nq = 10;
    int k = 10;

    faiss::IndexHNSWFlatPanorama index(d, 16, 8);

    // Generate and add data
    auto xb = generate_random_vectors(nb, d);
    index.add(nb, xb.data());

    EXPECT_EQ(index.ntotal, nb);
    EXPECT_EQ(index.cum_sums.size(), nb * (8 + 1));

    // Search
    auto xq = generate_random_vectors(nq, d, 5678);
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    index.search(nq, xq.data(), k, D.data(), I.data());

    // Check that results are valid
    for (int i = 0; i < nq * k; i++) {
        EXPECT_GE(I[i], -1);
        EXPECT_LT(I[i], nb);
        EXPECT_GE(D[i], 0.0f);
    }
}

TEST(IndexHNSWFlatPanorama, Recall) {
    int d = 128;
    int nb = 1000;
    int nq = 100;
    int k = 10;

    // Create index
    faiss::IndexHNSWFlatPanorama index(d, 32, 8);
    index.hnsw.efConstruction = 40;
    index.hnsw.efSearch = 64;

    // Generate data
    auto xb = generate_random_vectors(nb, d);
    auto xq = generate_random_vectors(nq, d, 5678);

    // Add to index
    index.add(nb, xb.data());

    // Search
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);
    index.search(nq, xq.data(), k, D.data(), I.data());

    // Compute ground truth
    std::vector<faiss::idx_t> gt_I(nq * k);
    std::vector<float> gt_D(nq * k);
    compute_ground_truth(
            xb.data(), nb, xq.data(), nq, d, k, gt_I.data(), gt_D.data());

    // Check recall
    float recall = compute_recall(gt_I.data(), I.data(), nq, k);
    std::cout << "Recall@" << k << ": " << recall << std::endl;

    // With efSearch=64, we should get reasonably good recall
    // The threshold is lower than vanilla HNSW because of approximate distances
    EXPECT_GE(recall, 0.85f);
}

TEST(IndexHNSWFlatPanorama, DifferentPanoramaLevels) {
    int d = 256;
    int nb = 500;
    int nq = 50;
    int k = 5;

    auto xb = generate_random_vectors(nb, d);
    auto xq = generate_random_vectors(nq, d, 5678);

    // Compute ground truth
    std::vector<faiss::idx_t> gt_I(nq * k);
    std::vector<float> gt_D(nq * k);
    compute_ground_truth(
            xb.data(), nb, xq.data(), nq, d, k, gt_I.data(), gt_D.data());

    // Test different number of panorama levels
    for (int nlevels : {4, 8, 16}) {
        faiss::IndexHNSWFlatPanorama index(d, 32, nlevels);
        index.hnsw.efSearch = 64;

        index.add(nb, xb.data());

        std::vector<faiss::idx_t> I(nq * k);
        std::vector<float> D(nq * k);
        index.search(nq, xq.data(), k, D.data(), I.data());

        float recall = compute_recall(gt_I.data(), I.data(), nq, k);
        std::cout << "Recall@" << k << " with " << nlevels
                  << " levels: " << recall << std::endl;

        // More levels should still maintain reasonable recall
        EXPECT_GE(recall, 0.80f);
    }
}

TEST(IndexHNSWFlatPanorama, Consistency) {
    int d = 64;
    int nb = 500;
    int nq = 10;
    int k = 5;

    faiss::IndexHNSWFlatPanorama index(d, 16, 8);

    auto xb = generate_random_vectors(nb, d);
    auto xq = generate_random_vectors(nq, d, 5678);

    index.add(nb, xb.data());

    // Search twice and compare results
    std::vector<faiss::idx_t> I1(nq * k);
    std::vector<float> D1(nq * k);
    index.search(nq, xq.data(), k, D1.data(), I1.data());

    std::vector<faiss::idx_t> I2(nq * k);
    std::vector<float> D2(nq * k);
    index.search(nq, xq.data(), k, D2.data(), I2.data());

    // Results should be identical
    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(I1[i], I2[i]);
        EXPECT_FLOAT_EQ(D1[i], D2[i]);
    }
}

TEST(IndexHNSWFlatPanorama, IOTest) {
    int d = 64;
    int nb = 500;
    int nq = 10;
    int k = 5;

    faiss::IndexHNSWFlatPanorama index(d, 16, 8);

    auto xb = generate_random_vectors(nb, d);
    auto xq = generate_random_vectors(nq, d, 5678);

    index.add(nb, xb.data());

    // Get search results before saving
    std::vector<faiss::idx_t> I_before(nq * k);
    std::vector<float> D_before(nq * k);
    index.search(nq, xq.data(), k, D_before.data(), I_before.data());

    // Save and load
    const char* fname = "/tmp/test_hnsw_panorama.index";
    faiss::write_index(&index, fname);

    auto loaded_index = faiss::read_index(fname);
    auto* panorama_loaded =
            dynamic_cast<faiss::IndexHNSWFlatPanorama*>(loaded_index);

    ASSERT_NE(panorama_loaded, nullptr);
    EXPECT_EQ(panorama_loaded->d, d);
    EXPECT_EQ(panorama_loaded->ntotal, nb);
    EXPECT_EQ(panorama_loaded->num_panorama_levels, 8);

    // Search after loading
    std::vector<faiss::idx_t> I_after(nq * k);
    std::vector<float> D_after(nq * k);
    panorama_loaded->search(nq, xq.data(), k, D_after.data(), I_after.data());

    // Results should be identical after serialization
    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(I_before[i], I_after[i]);
        EXPECT_EQ(D_before[i], D_after[i]);
    }

    delete loaded_index;
    std::remove(fname);
}

TEST(IndexHNSWFlatPanorama, CloneTest) {
    int d = 64;
    int nb = 500;
    int nq = 10;
    int k = 5;

    faiss::IndexHNSWFlatPanorama index(d, 16, 8);

    auto xb = generate_random_vectors(nb, d);
    auto xq = generate_random_vectors(nq, d, 5678);

    index.add(nb, xb.data());

    // Get search results before cloning
    std::vector<faiss::idx_t> I_before(nq * k);
    std::vector<float> D_before(nq * k);
    index.search(nq, xq.data(), k, D_before.data(), I_before.data());

    // Clone
    auto cloned_index = faiss::clone_index(&index);
    auto* panorama_cloned =
            dynamic_cast<faiss::IndexHNSWFlatPanorama*>(cloned_index);

    ASSERT_NE(panorama_cloned, nullptr);
    EXPECT_EQ(panorama_cloned->d, d);
    EXPECT_EQ(panorama_cloned->ntotal, nb);
    EXPECT_EQ(panorama_cloned->num_panorama_levels, 8);

    // Search after cloning
    std::vector<faiss::idx_t> I_after(nq * k);
    std::vector<float> D_after(nq * k);
    panorama_cloned->search(nq, xq.data(), k, D_after.data(), I_after.data());

    // Results should be identical after cloning
    for (int i = 0; i < nq * k; i++) {
        EXPECT_EQ(I_before[i], I_after[i]);
        EXPECT_EQ(D_before[i], D_after[i]);
    }

    delete cloned_index;
}

TEST(IndexHNSWFlatPanorama, FactoryTest) {
    int d = 64;
    int nb = 500;

    // Test factory creation with default levels
    auto index1 = faiss::index_factory(d, "HNSW32_FlatPanorama");
    auto* panorama1 = dynamic_cast<faiss::IndexHNSWFlatPanorama*>(index1);
    ASSERT_NE(panorama1, nullptr);
    EXPECT_EQ(panorama1->d, d);
    EXPECT_EQ(
            panorama1->hnsw.nb_neighbors(0),
            2 * 32);                              // Level 0 has 2*M neighbors
    EXPECT_EQ(panorama1->num_panorama_levels, 8); // default

    // Test factory creation with explicit levels
    auto index2 = faiss::index_factory(d, "HNSW16_FlatPanorama12");
    auto* panorama2 = dynamic_cast<faiss::IndexHNSWFlatPanorama*>(index2);
    ASSERT_NE(panorama2, nullptr);
    EXPECT_EQ(panorama2->d, d);
    EXPECT_EQ(
            panorama2->hnsw.nb_neighbors(0),
            2 * 16); // Level 0 has 2*M neighbors
    EXPECT_EQ(panorama2->num_panorama_levels, 12);

    // Test that it works
    auto xb = generate_random_vectors(nb, d);
    panorama2->add(nb, xb.data());
    EXPECT_EQ(panorama2->ntotal, nb);

    delete index1;
    delete index2;
}

TEST(IndexHNSWFlatPanorama, ResetTest) {
    int d = 64;
    int nb = 500;

    faiss::IndexHNSWFlatPanorama index(d, 16, 8);

    auto xb = generate_random_vectors(nb, d);
    index.add(nb, xb.data());

    EXPECT_EQ(index.ntotal, nb);
    EXPECT_GT(index.cum_sums.size(), 0);

    // Reset
    index.reset();

    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(index.cum_sums.size(), 0);
}

TEST(IndexHNSWFlatPanorama, HighDimensional) {
    // Panorama is designed for high-dimensional data
    int d = 512;
    int nb = 500;
    int nq = 50;
    int k = 10;

    faiss::IndexHNSWFlatPanorama index(d, 32, 16);
    index.hnsw.efSearch = 128;

    auto xb = generate_random_vectors(nb, d);
    auto xq = generate_random_vectors(nq, d, 5678);

    index.add(nb, xb.data());

    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);
    index.search(nq, xq.data(), k, D.data(), I.data());

    // Compute ground truth
    std::vector<faiss::idx_t> gt_I(nq * k);
    std::vector<float> gt_D(nq * k);
    compute_ground_truth(
            xb.data(), nb, xq.data(), nq, d, k, gt_I.data(), gt_D.data());

    float recall = compute_recall(gt_I.data(), I.data(), nq, k);
    std::cout << "High-dimensional (d=" << d << ") Recall@" << k << ": "
              << recall << std::endl;

    // Should work well for high-dimensional data
    EXPECT_GE(recall, 0.85f);
}

TEST(IndexHNSWFlatPanorama, AddAfterSearch) {
    int d = 64;
    int nb_initial = 300;
    int nb_add = 200;
    int nq = 10;
    int k = 5;

    faiss::IndexHNSWFlatPanorama index(d, 16, 8);

    // Add initial vectors
    auto xb_initial = generate_random_vectors(nb_initial, d);
    index.add(nb_initial, xb_initial.data());

    // Search on initial index
    auto xq = generate_random_vectors(nq, d, 5678);
    std::vector<faiss::idx_t> I_before(nq * k);
    std::vector<float> D_before(nq * k);
    index.search(nq, xq.data(), k, D_before.data(), I_before.data());

    // Add more vectors after search
    auto xb_add = generate_random_vectors(nb_add, d, 9999);
    index.add(nb_add, xb_add.data());

    EXPECT_EQ(index.ntotal, nb_initial + nb_add);
    EXPECT_EQ(
            index.cum_sums.size(),
            (nb_initial + nb_add) * (index.num_panorama_levels + 1));

    // Search again - should work correctly with all vectors
    std::vector<faiss::idx_t> I_after(nq * k);
    std::vector<float> D_after(nq * k);
    index.search(nq, xq.data(), k, D_after.data(), I_after.data());

    // Compute ground truth on all vectors
    std::vector<float> xb_all(d * (nb_initial + nb_add));
    std::copy(xb_initial.begin(), xb_initial.end(), xb_all.begin());
    std::copy(xb_add.begin(), xb_add.end(), xb_all.begin() + d * nb_initial);

    std::vector<faiss::idx_t> gt_I(nq * k);
    std::vector<float> gt_D(nq * k);
    compute_ground_truth(
            xb_all.data(),
            nb_initial + nb_add,
            xq.data(),
            nq,
            d,
            k,
            gt_I.data(),
            gt_D.data());

    float recall = compute_recall(gt_I.data(), I_after.data(), nq, k);
    std::cout << "Recall after adding more vectors: " << recall << std::endl;

    // Recall might be slightly lower than single-batch due to HNSW graph
    // structure
    EXPECT_GE(recall, 0.80f);

    // Verify that previously found neighbors can still be found
    // (they might not be in exact same positions due to new neighbors)
    int found_count = 0;
    for (int i = 0; i < nq * k; i++) {
        if (I_before[i] >= 0) {
            // Check if this neighbor is still in the result set
            for (int j = 0; j < k; j++) {
                if (I_after[i / k * k + j] == I_before[i]) {
                    found_count++;
                    break;
                }
            }
        }
    }
    float retention = static_cast<float>(found_count) / (nq * k);
    std::cout << "Retention of previous neighbors: " << retention << std::endl;
    // Should retain most previous neighbors (new ones might push some out)
    EXPECT_GE(retention, 0.5f);
}

TEST(IndexHNSWFlatPanorama, PermuteEntries) {
    int d = 64;
    int nb = 500;
    int nq = 10;
    int k = 5;

    faiss::IndexHNSWFlatPanorama index(d, 16, 8);

    auto xb = generate_random_vectors(nb, d);
    auto xq = generate_random_vectors(nq, d, 5678);

    index.add(nb, xb.data());

    // Search before permutation
    std::vector<faiss::idx_t> I_before(nq * k);
    std::vector<float> D_before(nq * k);
    index.search(nq, xq.data(), k, D_before.data(), I_before.data());

    // Create a permutation (reverse order for simplicity)
    std::vector<faiss::idx_t> perm(nb);
    for (int i = 0; i < nb; i++) {
        perm[i] = nb - 1 - i;
    }

    // Store cum_sums before permutation for verification
    std::vector<float> cum_sums_before = index.cum_sums;

    // Apply permutation
    index.permute_entries(perm.data());

    // Verify cum_sums were permuted correctly
    for (int i = 0; i < nb; i++) {
        int src = perm[i];
        for (int j = 0; j < index.num_panorama_levels + 1; j++) {
            EXPECT_EQ(
                    index.cum_sums[i * (index.num_panorama_levels + 1) + j],
                    cum_sums_before[src * (index.num_panorama_levels + 1) + j])
                    << "cum_sums not permuted correctly at i=" << i
                    << ", j=" << j;
        }
    }

    // Search after permutation
    std::vector<faiss::idx_t> I_after(nq * k);
    std::vector<float> D_after(nq * k);
    index.search(nq, xq.data(), k, D_after.data(), I_after.data());

    // Results should be identical (with permuted IDs)
    for (int i = 0; i < nq * k; i++) {
        EXPECT_FLOAT_EQ(D_before[i], D_after[i])
                << "Distance changed after permutation at position " << i;

        // The ID should be the permuted version
        // If before we found vector j, after permutation we should find
        // the new position of vector j
        if (I_before[i] >= 0) {
            // Find where I_before[i] moved to
            int old_id = I_before[i];
            int new_id = -1;
            for (int p = 0; p < nb; p++) {
                if (perm[p] == old_id) {
                    new_id = p;
                    break;
                }
            }
            EXPECT_EQ(I_after[i], new_id)
                    << "Permuted ID mismatch at position " << i;
        }
    }

    // Verify overall recall is maintained
    std::vector<faiss::idx_t> gt_I(nq * k);
    std::vector<float> gt_D(nq * k);
    compute_ground_truth(
            xb.data(), nb, xq.data(), nq, d, k, gt_I.data(), gt_D.data());

    // Map ground truth IDs through the permutation
    std::vector<faiss::idx_t> gt_I_permuted(nq * k);
    for (int i = 0; i < nq * k; i++) {
        if (gt_I[i] >= 0) {
            // Find where gt_I[i] moved to
            for (int p = 0; p < nb; p++) {
                if (perm[p] == gt_I[i]) {
                    gt_I_permuted[i] = p;
                    break;
                }
            }
        } else {
            gt_I_permuted[i] = -1;
        }
    }

    float recall = compute_recall(gt_I_permuted.data(), I_after.data(), nq, k);
    std::cout << "Recall after permutation: " << recall << std::endl;
    EXPECT_GE(recall, 0.85f);
}
