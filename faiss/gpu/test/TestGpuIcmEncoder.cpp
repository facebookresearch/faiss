#include <faiss/gpu/GpuIcmEncoder.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/impl/LocalSearchQuantizer.h>

#include <gtest/gtest.h>
#include <tuple>
#include <vector>

using faiss::LocalSearchQuantizer;
using faiss::gpu::GpuIcmEncoder;
using faiss::gpu::GpuResourcesProvider;
using faiss::gpu::StandardGpuResources;

struct ShardingTestParams {
    size_t n;
    size_t nshards;
};

class GpuIcmEncoderShardingTest
        : public ::testing::TestWithParam<ShardingTestParams> {
   protected:
    void SetUp() override {
        params = GetParam();

        lsq.M = 4;
        lsq.K = 16;
        lsq.d = 32;

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        lsq.codebooks.resize(lsq.M * lsq.K * lsq.d);
        for (auto& v : lsq.codebooks) {
            v = dist(gen);
        }

        x.resize(params.n * lsq.d);
        codes.resize(params.n * lsq.M);

        for (auto& v : x) {
            v = dist(gen);
        }

        std::uniform_int_distribution<int32_t> codeDist(0, lsq.K - 1);
        for (auto& c : codes) {
            c = codeDist(gen);
        }
    }

    LocalSearchQuantizer lsq;
    std::vector<float> x;
    std::vector<int32_t> codes;
    std::mt19937 gen;
    ShardingTestParams params;
    static constexpr size_t ils_iters = 4;
};

TEST_P(GpuIcmEncoderShardingTest, DataShardingCorrectness) {
    std::vector<StandardGpuResources> resources(params.nshards);
    std::vector<GpuResourcesProvider*> provs;
    std::vector<int> devices;

    for (size_t i = 0; i < params.nshards; ++i) {
        resources[i].noTempMemory();
        provs.push_back(&resources[i]);
        devices.push_back(0); // use GPU 0 for testing all shards
    }

    GpuIcmEncoder encoder(&lsq, provs, devices);
    encoder.set_binary_term();

    gen.seed(42);
    EXPECT_NO_THROW(
            encoder.encode(codes.data(), x.data(), gen, params.n, ils_iters));

    for (auto c : codes) {
        EXPECT_GE(c, 0);
        EXPECT_LT(c, lsq.K);
    }
}

std::vector<ShardingTestParams> GetShardingTestCases() {
    return {
            {1, 8},

            {5, 4},

            {10, 2},
            {10, 3},
            {10, 5},
            {10, 8},

            {20, 8},
    };
}

INSTANTIATE_TEST_SUITE_P(
        MultiGpuShardingTests,
        GpuIcmEncoderShardingTest,
        ::testing::ValuesIn(GetShardingTestCases()),
        [](const ::testing::TestParamInfo<ShardingTestParams>& info) {
            return "n" + std::to_string(info.param.n) + "_shards" +
                    std::to_string(info.param.nshards);
        });

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
