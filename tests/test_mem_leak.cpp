#include <faiss/utils/utils.h>
#include <faiss/utils/random.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>

#include <gtest/gtest.h>


using namespace faiss;


TEST(MEM_LEAK, ivfflat) {

    size_t num_tfidf_faiss_cells = 20;
    size_t max_tfidf_features = 500;

    IndexFlatIP quantizer(max_tfidf_features);
    IndexIVFFlat tfidf_faiss_index(
        &quantizer, max_tfidf_features, num_tfidf_faiss_cells);

    std::vector<float> dense_matrix(5000 * max_tfidf_features);
    float_rand(dense_matrix.data(), dense_matrix.size(), 123);

    tfidf_faiss_index.train(5000, dense_matrix.data());
    tfidf_faiss_index.add(5000, dense_matrix.data());

    std::vector<float> ent_substr_tfidfs_list(10000 * max_tfidf_features);
    float_rand(ent_substr_tfidfs_list.data(), ent_substr_tfidfs_list.size(), 1234);



    size_t m0 = get_mem_usage_kb();
    for(int i = 0; i < 100000; i++) {
        std::vector<Index::idx_t> I(10);
        std::vector<float> D(10);

        tfidf_faiss_index.search(
            1, ent_substr_tfidfs_list.data() + i * max_tfidf_features, 10,
            D.data(), I.data());
        if(i%100 == 0) {
            printf("%d: %ld kB %.2f bytes/it\r", i, get_mem_usage_kb(),
                (get_mem_usage_kb() - m0) * 1024.0 / (i + 1));
            fflush(stdout);
        }
    }

    EXPECT_GE(50, (get_mem_usage_kb() - m0) * 1024.0 / 100000);

}