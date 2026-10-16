/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// HadamardRotation with pad_to_power_of_2 = false: d_out is the smallest
// multiple of 64 >= d_in, and a d_out that is not a power of 2 uses the
// FHT-Kac rotation.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <faiss/VectorTransform.h>
#include <faiss/clone_index.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace {

std::vector<float> random_vectors(size_t n, size_t d, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> x(n * d);
    for (auto& v : x) {
        v = normal(rng);
    }
    return x;
}

double dot(const float* a, const float* b, size_t d) {
    double s = 0;
    for (size_t i = 0; i < d; i++) {
        s += double(a[i]) * b[i];
    }
    return s;
}

bool is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

uint32_t leading_fourcc(const faiss::VectorIOWriter& writer) {
    uint32_t h = 0;
    std::memcpy(&h, writer.data.data(), sizeof(h));
    return h;
}

} // namespace

TEST(HadamardRotation, OutputDimensions) {
    struct Case {
        int d, power_of_2, multiple_of_64;
    };
    for (const Case& c :
         {Case{1, 1, 64},
          Case{64, 64, 64},
          Case{65, 128, 128},
          Case{100, 128, 128},
          Case{192, 256, 192},
          Case{768, 1024, 768},
          Case{1000, 1024, 1024},
          Case{1280, 2048, 1280},
          Case{1536, 2048, 1536}}) {
        faiss::HadamardRotation pow2(c.d);
        EXPECT_TRUE(pow2.pad_to_power_of_2);
        EXPECT_EQ(pow2.d_out, c.power_of_2) << "d=" << c.d;
        EXPECT_TRUE(pow2.signs4.empty());

        faiss::HadamardRotation mult64(c.d, 12345, false);
        EXPECT_FALSE(mult64.pad_to_power_of_2);
        EXPECT_EQ(mult64.d_out, c.multiple_of_64) << "d=" << c.d;
        // The fourth round only exists when d_out is not a power of 2.
        EXPECT_EQ(
                mult64.signs4.size(),
                is_power_of_2(c.multiple_of_64) ? 0 : size_t(c.multiple_of_64))
                << "d=" << c.d;
    }
}

TEST(HadamardRotation, PaddedToMultipleOf64IsOrthogonal) {
    const size_t n = 40;
    for (int d : {100, 192, 700, 768, 1000, 1280, 1536}) {
        faiss::HadamardRotation hr(d, 42, false);
        const size_t p = hr.d_out;
        const auto x = random_vectors(n, d, d);
        std::vector<float> y(n * p);
        hr.apply_noalloc(n, x.data(), y.data());

        double max_change = 0;
        for (size_t i = 0; i < n; i++) {
            const float* xi = x.data() + i * d;
            const float* yi = y.data() + i * p;
            const double norm2 = dot(xi, xi, d);
            EXPECT_NEAR(dot(yi, yi, p), norm2, 1e-4 * norm2)
                    << "d=" << d << " i=" << i;
            if (i + 1 < n) {
                const float* xj = xi + d;
                const float* yj = yi + p;
                const double scale = std::sqrt(norm2 * dot(xj, xj, d));
                EXPECT_NEAR(dot(yi, yj, p), dot(xi, xj, d), 1e-4 * scale)
                        << "d=" << d << " i=" << i;
            }
            for (size_t k = 0; k < p; k++) {
                const float xk = k < size_t(d) ? xi[k] : 0.0f;
                max_change = std::max(max_change, double(std::abs(yi[k] - xk)));
            }
        }
        // Not the identity.
        EXPECT_GT(max_change, 0.1) << "d=" << d;
    }
}

TEST(HadamardRotation, PaddedToMultipleOf64ReverseTransform) {
    const size_t n = 10;
    // d_in == d_out, both for power-of-2 and FHT-Kac outputs.
    for (int d : {128, 768, 1280, 1536}) {
        faiss::HadamardRotation hr(d, 7, false);
        ASSERT_EQ(hr.d_out, d);
        const auto x = random_vectors(n, d, 99);
        std::vector<float> y(n * d), x2(n * d);
        hr.apply_noalloc(n, x.data(), y.data());
        hr.reverse_transform(n, y.data(), x2.data());
        for (size_t i = 0; i < n * d; i++) {
            ASSERT_NEAR(x2[i], x[i], 1e-4) << "d=" << d << " i=" << i;
        }
    }
}

TEST(HadamardRotation, PaddedToMultipleOf64SerializationRoundTrip) {
    const size_t n = 10;
    for (int d : {100, 1280}) {
        faiss::HadamardRotation hr(d, 2024, false);
        faiss::VectorIOWriter writer;
        faiss::write_VectorTransform(&hr, &writer);
        ASSERT_GE(writer.data.size(), sizeof(uint32_t));
        EXPECT_EQ(leading_fourcc(writer), faiss::fourcc("HR64"));

        faiss::VectorIOReader reader;
        reader.data = writer.data;
        auto vt = faiss::read_VectorTransform_up(&reader);
        auto* hr2 = dynamic_cast<faiss::HadamardRotation*>(vt.get());
        ASSERT_NE(hr2, nullptr);
        EXPECT_FALSE(hr2->pad_to_power_of_2);
        EXPECT_EQ(hr2->d_out, hr.d_out);
        hr.check_identical(*hr2);

        const auto x = random_vectors(n, d, 5);
        std::vector<float> y1(n * hr.d_out), y2(n * hr.d_out);
        hr.apply_noalloc(n, x.data(), y1.data());
        hr2->apply_noalloc(n, x.data(), y2.data());
        EXPECT_EQ(y1, y2) << "d=" << d;
    }
}

TEST(HadamardRotation, DefaultModeKeepsItsFormat) {
    faiss::HadamardRotation hr(1280, 2024);
    faiss::VectorIOWriter writer;
    faiss::write_VectorTransform(&hr, &writer);
    ASSERT_GE(writer.data.size(), sizeof(uint32_t));
    EXPECT_EQ(leading_fourcc(writer), faiss::fourcc("HRot"));

    faiss::VectorIOReader reader;
    reader.data = writer.data;
    auto vt = faiss::read_VectorTransform_up(&reader);
    auto* hr2 = dynamic_cast<faiss::HadamardRotation*>(vt.get());
    ASSERT_NE(hr2, nullptr);
    EXPECT_TRUE(hr2->pad_to_power_of_2);
    EXPECT_EQ(hr2->d_out, 2048);
    EXPECT_TRUE(hr2->signs4.empty());
    hr.check_identical(*hr2);
}

TEST(HadamardRotation, ClonePreservesPadding) {
    faiss::HadamardRotation hr(1280, 3, false);
    faiss::Cloner cloner;
    std::unique_ptr<faiss::VectorTransform> vt(
            cloner.clone_VectorTransform(&hr));
    auto* hr2 = dynamic_cast<faiss::HadamardRotation*>(vt.get());
    ASSERT_NE(hr2, nullptr);
    hr.check_identical(*hr2);
    EXPECT_EQ(hr2->signs4, hr.signs4);
}

TEST(HadamardRotation, CheckIdenticalComparesPadding) {
    // Same d_out, different padding modes.
    faiss::HadamardRotation pow2(64, 1, true);
    faiss::HadamardRotation mult64(64, 1, false);
    ASSERT_EQ(pow2.d_out, mult64.d_out);
    EXPECT_THROW(pow2.check_identical(mult64), faiss::FaissException);
}
