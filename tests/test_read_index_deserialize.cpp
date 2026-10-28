/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

using namespace faiss;

/// Helper: append a scalar value to the buffer in little-endian format,
/// matching WRITE1.
template <typename T>
static void push_val(std::vector<uint8_t>& buf, T val) {
    const auto* p = reinterpret_cast<const uint8_t*>(&val);
    buf.insert(buf.end(), p, p + sizeof(T));
}

/// Helper: append a WRITEVECTOR-formatted vector (size_t length prefix
/// followed by raw element data).
template <typename T>
static void push_vector(std::vector<uint8_t>& buf, const std::vector<T>& vec) {
    push_val<size_t>(buf, vec.size());
    const auto* p = reinterpret_cast<const uint8_t*>(vec.data());
    buf.insert(buf.end(), p, p + vec.size() * sizeof(T));
}

/// Helper: append a fourcc string as a uint32_t.
static void push_fourcc(std::vector<uint8_t>& buf, const char s[4]) {
    const auto* x = reinterpret_cast<const unsigned char*>(s);
    uint32_t h = x[0] | (x[1] << 8) | (x[2] << 16) | (x[3] << 24);
    push_val<uint32_t>(buf, h);
}

/// Helper: append write_index_header fields.
static void push_index_header(
        std::vector<uint8_t>& buf,
        int d,
        int64_t ntotal,
        bool is_trained = true,
        int metric_type = 1 /* L2 */) {
    push_val<int>(buf, d);
    push_val<int64_t>(buf, ntotal);
    int64_t dummy = 1 << 20;
    push_val<int64_t>(buf, dummy);
    push_val<int64_t>(buf, dummy);
    push_val<bool>(buf, is_trained);
    push_val<int>(buf, metric_type);
}

// -----------------------------------------------------------------------
// Test: Index2Layer deserialization sets own_fields=true so the
// sub-quantizer is freed when the deserialized index is destroyed.
// C++ only: Python cannot subclass Index to observe the destructor.
// -----------------------------------------------------------------------

struct CountedDestructorIndex : IndexFlatL2 {
    static int destructor_count;
    using IndexFlatL2::IndexFlatL2;
    ~CountedDestructorIndex() override {
        destructor_count++;
    }
};
int CountedDestructorIndex::destructor_count = 0;

TEST(ReadIndexDeserialize, Index2LayerQuantizerOwnership) {
    int d = 4, nb = 512, nlist = 2;

    auto original_q = std::make_unique<IndexFlatL2>(d);
    Index2Layer original(original_q.release(), nlist, 2);
    original.q1.own_fields = true;
    std::vector<float> xb(d * nb);
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist;
        for (auto& v : xb) {
            v = dist(rng);
        }
    }
    original.train(nb, xb.data());
    original.add(nb, xb.data());

    VectorIOWriter writer;
    write_index(&original, &writer);

    // Swap in a CountedDestructorIndex to observe whether own_fields frees it.
    CountedDestructorIndex::destructor_count = 0;
    {
        VectorIOReader reader;
        reader.data = writer.data;
        auto idx = std::unique_ptr<Index>(read_index(&reader));
        auto* idx2l = dynamic_cast<Index2Layer*>(idx.get());
        ASSERT_NE(idx2l, nullptr);

        delete idx2l->q1.quantizer;
        idx2l->q1.quantizer = new CountedDestructorIndex(d);
        // idx goes out of scope here. If own_fields is true,
        // ~Level1Quantizer deletes the CountedDestructorIndex.
    }
    EXPECT_EQ(CountedDestructorIndex::destructor_count, 1);
}

namespace {

constexpr size_t kTurboQFullDimension = 16;
constexpr uint64_t kTurboQFullSeed = 0x0123456789abcdefULL;
constexpr auto kTurboQFullQtype = ScalarQuantizer::QT_3bit_tq;
// QT_3bit_tq: (3 - 1) MSE bits + 1 QJL bit, so 1 << (3 - 1) centroids.
constexpr size_t kTurboQFullCentroids = 4;

std::vector<uint8_t> make_turboq_full_sq_payload(uint8_t qjl_type) {
    float seed[2];
    ScalarQuantizer::TurboQuantRefine::pack_seed(kTurboQFullSeed, seed);
    std::vector<float> trained(
            kTurboQFullCentroids + (kTurboQFullCentroids - 1) + 3, 0.0f);
    trained[trained.size() - 3] = seed[0];
    trained[trained.size() - 2] = seed[1];
    trained[trained.size() - 1] = static_cast<float>(qjl_type);

    std::vector<uint8_t> buf;
    push_fourcc(buf, "IxSQ");
    push_index_header(
            buf, static_cast<int>(kTurboQFullDimension), /*ntotal=*/0);
    push_val<int>(buf, kTurboQFullQtype);
    push_val<int>(buf, 0);      // rangestat
    push_val<float>(buf, 0.0f); // rangestat_arg
    push_val<size_t>(buf, kTurboQFullDimension);
    push_val<size_t>(buf, 0); // code_size is recomputed on read
    push_vector<float>(buf, trained);
    push_vector<uint8_t>(buf, {}); // codes
    return buf;
}

} // namespace

// C++ only: swig does not wrap ScalarQuantizer::TurboQuantRefine.
TEST(ReadIndexDeserialize, SQTurboQuantFullRecoversConfigFromTrained) {
    // qjl_type 2, not the default 0, so a missing extraction is caught.
    const uint8_t qjl_type = 2;
    VectorIOReader reader;
    reader.data = make_turboq_full_sq_payload(qjl_type);
    std::unique_ptr<Index> index;
    ASSERT_NO_THROW(index = read_index_up(&reader));

    const auto* sq_index = dynamic_cast<IndexScalarQuantizer*>(index.get());
    ASSERT_NE(sq_index, nullptr);
    EXPECT_EQ(sq_index->sq.turboq_refine.qjl_type, qjl_type);
    EXPECT_EQ(sq_index->sq.turboq_refine.seed, kTurboQFullSeed);
}

#ifdef FAISS_ENABLE_SVS

// -----------------------------------------------------------------------
// Tests: SVS DirectoryArchiver entries stay inside the extraction root.
// C++ only: they check the path check in the fbsource copy of SVS
// (third-party/svs/v0.4.0/src/include/svs/lib/file.h). The SVS runtime that
// CMake builds use does not have this check.
// -----------------------------------------------------------------------

/// Try to read a float index and expect a FaissException whose message
/// contains the given substring.
static void expect_read_throws_with(
        const std::vector<uint8_t>& data,
        const std::string& expected_substr) {
    VectorIOReader reader;
    reader.data = data;
    try {
        read_index_up(&reader);
        FAIL() << "expected FaissException";
    } catch (const FaissException& e) {
        EXPECT_NE(
                std::string(e.what()).find(expected_substr), std::string::npos)
                << "expected '" << expected_substr << "' in: " << e.what();
    }
}

/// DirectoryArchiver::magic_number, from
/// third-party/svs/v0.4.0/src/include/svs/lib/file.h.
static constexpr uint64_t kSvsDirectoryArchiverMagic = 0x5e2d58d9f3b4a6c1ULL;

/// Append an "ISVF" index that routes into DirectoryArchiver::unpack carrying
/// a single archive entry. `declared_name_len` is written to the stream as the
/// entry's name length and is deliberately separate from `filename` so that a
/// test can claim a length it does not supply.
static void push_svs_flat_archive_entry(
        std::vector<uint8_t>& buf,
        const std::string& filename,
        uint64_t declared_name_len,
        const std::vector<uint8_t>& contents = {}) {
    push_fourcc(buf, "ISVF");
    push_index_header(buf, 8, 0);
    push_val<bool>(buf, true); // initialized -> enters deserialize_impl
    push_val<uint64_t>(buf, kSvsDirectoryArchiverMagic);
    push_val<uint64_t>(buf, uint64_t(1)); // num_files
    push_val<uint64_t>(buf, declared_name_len);
    buf.insert(buf.end(), filename.begin(), filename.end());
    push_val<uint64_t>(buf, uint64_t(contents.size()));
    buf.insert(buf.end(), contents.begin(), contents.end());
}

// An absolute entry name must not escape the archiver's temp root.
// std::filesystem::operator/ discards its left operand entirely when the right
// operand is absolute, so `root / filename` collapses to `filename`.
TEST(ReadIndexDeserialize, SVSArchiveAbsolutePathEntryRejected) {
    const auto probe = std::filesystem::temp_directory_path() /
            "faiss_svs_archive_absolute_probe";
    std::filesystem::remove(probe);

    const std::string filename = probe.string();
    std::vector<uint8_t> buf;
    push_svs_flat_archive_entry(buf, filename, filename.size(), {'x'});

    expect_read_throws_with(buf, "escapes the extraction root");
    EXPECT_FALSE(std::filesystem::exists(probe));
}

// Same, for a relative entry name that climbs out of the temp root. The
// archiver unpacks into temp_directory_path()/svs_flat_load-<uuid>, so a
// single ".." lands back in temp_directory_path().
TEST(ReadIndexDeserialize, SVSArchiveParentTraversalEntryRejected) {
    const auto probe = std::filesystem::temp_directory_path() /
            "faiss_svs_archive_traversal_probe";
    std::filesystem::remove(probe);

    const std::string filename = "../faiss_svs_archive_traversal_probe";
    std::vector<uint8_t> buf;
    push_svs_flat_archive_entry(buf, filename, filename.size(), {'x'});

    expect_read_throws_with(buf, "escapes the extraction root");
    EXPECT_FALSE(std::filesystem::exists(probe));
}

#endif // FAISS_ENABLE_SVS
