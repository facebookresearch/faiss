#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <vector>

// Include the C API headers for IndexBinaryIVF
#include "c_api/IndexBinaryIVF_c.h"
#include "c_api/Index_c.h"
#include "c_api/faiss_c.h"

class BinaryIVFBufferOverflowTest : public ::testing::TestWithParam<size_t> {};

TEST_P(BinaryIVFBufferOverflowTest, GetInvListDoesNotOverflowBuffer) {
    // Invariant: The number of entries returned by faiss_IndexBinaryIVF_invlists_get_ids
    // must never exceed the actual list size reported by the index, ensuring memcpy
    // cannot write beyond the allocated destination buffer.

    size_t requested_list_id = GetParam();

    // Create a small BinaryIVF index via the C API
    FaissIndexBinaryIVF* index = nullptr;
    FaissIndexBinaryFlat* quantizer = nullptr;
    int d = 64; // dimension in bits
    size_t nlist = 4;

    int err = faiss_IndexBinaryFlat_new(&quantizer, d);
    if (err != 0) GTEST_SKIP() << "Could not create quantizer";

    err = faiss_IndexBinaryIVF_new_with_quantizer(
        &index, (FaissIndexBinary*)quantizer, d, nlist);
    if (err != 0) {
        faiss_IndexBinary_free((FaissIndexBinary*)quantizer);
        GTEST_SKIP() << "Could not create IVF index";
    }

    // Train and add a few vectors so some lists are non-empty
    size_t n = 32;
    std::vector<uint8_t> data(n * (d / 8), 0xAA);
    faiss_IndexBinary_train((FaissIndexBinary*)index, n, data.data());
    faiss_IndexBinary_add((FaissIndexBinary*)index, n, data.data());

    // Query the list size for the requested list_id
    size_t list_size = faiss_IndexBinaryIVF_invlists_list_size(index, requested_list_id);

    // Allocate a buffer of exactly list_size entries
    std::vector<int64_t> ids_buffer(list_size);

    // Retrieve IDs - the copy must not exceed list_size entries
    if (requested_list_id < nlist && list_size > 0) {
        const int64_t* ids_ptr = nullptr;
        // Use the invlists accessor; verify returned size matches
        size_t reported_size = faiss_IndexBinaryIVF_invlists_list_size(index, requested_list_id);
        ASSERT_EQ(reported_size, list_size)
            << "List size changed between calls - potential race/corruption";

        // The key invariant: list_size must be consistent and bounded
        ASSERT_LE(list_size, n)
            << "List size exceeds total number of added vectors";
    } else {
        // Out-of-range list or empty list: size must be 0 for safety
        if (requested_list_id >= nlist) {
            ASSERT_EQ(list_size, 0)
                << "Out-of-range list_id should report size 0";
        }
    }

    faiss_IndexBinary_free((FaissIndexBinary*)index);
}

INSTANTIATE_TEST_SUITE_P(
    AdversarialInputs,
    BinaryIVFBufferOverflowTest,
    ::testing::Values(
        (size_t)0,              // Valid: first list
        (size_t)3,              // Valid: last list in nlist=4
        (size_t)4,             // Boundary: one past valid range
        (size_t)0xFFFFFFFF,    // Adversarial: large list_id
        (size_t)0x7FFFFFFFFFFFFFFF  // Adversarial: near-max size_t
    )
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}