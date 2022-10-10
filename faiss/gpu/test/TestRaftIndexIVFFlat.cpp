/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/raft/RaftIndexIVFFlat.h>
#include <faiss/gpu/raft/RmmGpuResources.hpp>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>

#include <faiss/gpu/GpuDistance.h>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/nvtx.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <vector>

// FIXME: figure out a better way to test fp16
constexpr float kF16MaxRelErr = 0.3f;
constexpr float kF32MaxRelErr = 0.03f;

struct Options {
    Options() {
        numAdd = 2 * faiss::gpu::randVal(50000, 70000);
        dim = faiss::gpu::randVal(64, 200);

        numCentroids = std::sqrt((float)numAdd / 2);
        numTrain = numCentroids * 50;
        nprobe = faiss::gpu::randVal(std::min(10, numCentroids), numCentroids);
        numQuery = numAdd / 10;//faiss::gpu::randVal(32, 100);

        // Due to the approximate nature of the query and of floating point
        // differences between GPU and CPU, to stay within our error bounds,
        // only use a small k
        k = std::min(faiss::gpu::randVal(10, 30), numAdd / 40);
        indicesOpt = faiss::gpu::randSelect(
                {faiss::gpu::INDICES_CPU,
                 faiss::gpu::INDICES_32_BIT,
                 faiss::gpu::INDICES_64_BIT});

        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
    }

    std::string toString() const {
        std::stringstream str;
        str << "IVFFlat device " << device << " numVecs " << numAdd << " dim "
            << dim << " numCentroids " << numCentroids << " nprobe " << nprobe
            << " numQuery " << numQuery << " k " << k << " indicesOpt "
            << indicesOpt;

        return str.str();
    }

    int numAdd;
    int dim;
    int numCentroids;
    int numTrain;
    int nprobe;
    int numQuery;
    int k;
    int device;
    faiss::gpu::IndicesOptions indicesOpt;
};

template<typename idx_type>
void train_index(const raft::handle_t &raft_handle, Options &opt, idx_type &index, std::vector<float> &trainVecs, std::vector<float> &addVecs) {

    uint32_t train_start = raft::curTimeMillis();
    index.train(opt.numTrain, trainVecs.data());
    raft_handle.sync_stream();
    uint32_t train_stop = raft::curTimeMillis();

    uint32_t add_start = raft::curTimeMillis();
    index.add(opt.numAdd, addVecs.data());
    raft_handle.sync_stream();
    uint32_t add_stop = raft::curTimeMillis();
//    index.train(opt.numTrain, trainVecs.data());
    index.setNumProbes(opt.nprobe);

    std::cout << "train=" << (train_stop - train_start) << ", add=" << (add_stop - add_start) << std::endl;
}


void invoke_bfknn(const raft::handle_t &raft_handle, Options &opt, float *dists, faiss::Index::idx_t *inds, faiss::MetricType m,
                  std::vector<float> &addVecs, std::vector<float> &queryVecs) {



    faiss::gpu::RmmGpuResources gpu_res;
    gpu_res.setDefaultStream(opt.device, raft_handle.get_stream());

    rmm::device_uvector<float> addVecsDev(addVecs.size(), raft_handle.get_stream());
    raft::copy(addVecsDev.data(), addVecs.data(), addVecs.size(), raft_handle.get_stream());

    rmm::device_uvector<float> queryVecsDev(queryVecs.size(), raft_handle.get_stream());
    raft::copy(queryVecsDev.data(), queryVecs.data(), queryVecs.size(), raft_handle.get_stream());

    faiss::gpu::GpuDistanceParams args;
    args.metric          = m;
    args.k               = opt.k;
    args.dims            = opt.dim;
    args.vectors         = addVecs.data();
    args.vectorsRowMajor = true;
    args.numVectors      = opt.numAdd;
    args.queries         = queryVecs.data();
    args.queriesRowMajor = true;
    args.numQueries      = opt.numQuery;
    args.outDistances    = dists;
    args.outIndices      = inds;
    args.outIndicesType  = faiss::gpu::IndicesDataType::I64;

    /**
     * @todo: Until FAISS supports pluggable allocation strategies,
     * we will not reap the benefits of the pool allocator for
     * avoiding device-wide synchronizations from cudaMalloc/cudaFree
     */
    bfKnn(&gpu_res, args);
}

void queryTest(
        faiss::MetricType metricType,
        bool useFloat16CoarseQuantizer,
        int dimOverride = -1) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;
        opt.dim = dimOverride != -1 ? dimOverride : opt.dim;

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

        std::vector<float> queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);

        std::cout << "numTrain: " << opt.numTrain << "numCentroids: " << opt.numCentroids << std::endl;

        printf("Creating rmm resources\n");
        faiss::gpu::RmmGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = opt.device;
        config.indicesOptions = opt.indicesOpt;
        config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;

        // TODO: Since we are modifying the centroids when adding new vectors,
        // the neighbors are no longer going to match completely between CPU
        // and the RAFT indexes. We will probably want to perform a bfknn as
        // ground truth and then compare the recall for both the RAFT and FAISS
        // indices.

        printf("Building raft index\n");
        faiss::gpu::RaftIndexIVFFlat raftIndex(
                &res, opt.dim, opt.numCentroids, metricType, config);

        printf("Done.\n");

        faiss::gpu::GpuIndexIVFFlat gpuIndex(
                &res, opt.dim, opt.numCentroids, metricType, config);


        printf("Creating raft handle\n");
        raft::handle_t raft_handle;
        printf("Done\n");

        std::cout << "Training raft index" << std::endl;
        uint32_t r_train_start = raft::curTimeMillis();
        train_index(raft_handle, opt, raftIndex, trainVecs, addVecs);
        raft_handle.sync_stream();
        uint32_t r_train_stop = raft::curTimeMillis();
        std::cout << "Raft train time " << r_train_start << " " << r_train_stop << " " << (r_train_stop - r_train_start) << std::endl;

        std::cout << "Training gpu index" << std::endl;
        uint32_t g_train_start = raft::curTimeMillis();
        train_index(raft_handle, opt, gpuIndex, trainVecs, addVecs);
        raft_handle.sync_stream();
        uint32_t g_train_stop = raft::curTimeMillis();
        std::cout << "FAISS train time " << g_train_start << " " << g_train_stop << " " << (g_train_stop - g_train_start) << std::endl;

        std::cout << "Computing ground truth" << std::endl;
        rmm::device_uvector<faiss::Index::idx_t> ref_inds(opt.numQuery * opt.k, raft_handle.get_stream());
        rmm::device_uvector<float> ref_dists(opt.numQuery * opt.k, raft_handle.get_stream());

        invoke_bfknn(raft_handle, opt, ref_dists.data(), ref_inds.data(), metricType, addVecs, queryVecs);

        std::cout << "Done." << std::endl;
        raft::print_device_vector("ref_dists", ref_dists.data(), opt.k, std::cout);
        raft::print_device_vector("ref_inds", ref_inds.data(), opt.k, std::cout);

        rmm::device_uvector<faiss::Index::idx_t> raft_inds(opt.numQuery * opt.k, raft_handle.get_stream());
        rmm::device_uvector<float> raft_dists(opt.numQuery * opt.k, raft_handle.get_stream());

        uint32_t rstart = raft::curTimeMillis();
        raftIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                raft_dists.data(),
                raft_inds.data());

        raft_handle.sync_stream();
        uint32_t rstop = raft::curTimeMillis();
        std::cout << "Raft query time " << rstart << " " << rstop << " " << (rstop - rstart) << std::endl;

        rmm::device_uvector<faiss::Index::idx_t> gpu_inds(opt.numQuery * opt.k, raft_handle.get_stream());
        rmm::device_uvector<float> gpu_dists(opt.numQuery * opt.k, raft_handle.get_stream());

        uint32_t gstart = raft::curTimeMillis();
        gpuIndex.search(
                opt.numQuery,
                queryVecs.data(),
                opt.k,
                gpu_dists.data(),
                gpu_inds.data());

        raft_handle.sync_stream();
        uint32_t gstop = raft::curTimeMillis();

        std::cout << "FAISS query time " << gstart << " " << gstop << " " << (gstop - gstart) << std::endl;

        // TODO: Compare recall, perhaps by adding the indices/distances to a hashmap.

        raft::print_device_vector("raft_dists", raft_dists.data(), opt.k, std::cout);
        raft::print_device_vector("raft_inds", raft_inds.data(), opt.k, std::cout);

//        raft::print_device_vector("gpu_dists", gpu_dists.data(), opt.k, std::cout);
//        raft::print_device_vector("gpu_inds", gpu_inds.data(), opt.k, std::cout);

//
//        bool compFloat16 = useFloat16CoarseQuantizer;
//        faiss::gpu::compareIndices(
//                cpuIndex,
//                gpuIndex,
//                opt.numQuery,
//                opt.dim,
//                opt.k,
//                opt.toString(),
//                compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
//                // FIXME: the fp16 bounds are
//                // useless when math (the accumulator) is
//                // in fp16. Figure out another way to test
//                compFloat16 ? 0.70f : 0.1f,
//                compFloat16 ? 0.65f : 0.015f);
    }
}

void addTest(faiss::MetricType metricType, bool useFloat16CoarseQuantizer) {
    for (int tries = 0; tries < 2; ++tries) {
        Options opt;

        std::vector<float> trainVecs =
                faiss::gpu::randVecs(opt.numTrain, opt.dim);
        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

        faiss::IndexFlatL2 quantizerL2(opt.dim);
        faiss::IndexFlatIP quantizerIP(opt.dim);
        faiss::Index* quantizer = metricType == faiss::METRIC_L2
                ? (faiss::Index*)&quantizerL2
                : (faiss::Index*)&quantizerIP;

        faiss::IndexIVFFlat cpuIndex(
                quantizer, opt.dim, opt.numCentroids, metricType);
        cpuIndex.train(opt.numTrain, trainVecs.data());
        cpuIndex.nprobe = opt.nprobe;

        faiss::gpu::RmmGpuResources res;
        res.noTempMemory();

        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = opt.device;
        config.indicesOptions = opt.indicesOpt;
        config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;

        faiss::gpu::RaftIndexIVFFlat gpuIndex(
                &res, cpuIndex.d, cpuIndex.nlist, cpuIndex.metric_type, config);
        gpuIndex.copyFrom(&cpuIndex);
        gpuIndex.setNumProbes(opt.nprobe);

        cpuIndex.add(opt.numAdd, addVecs.data());
        gpuIndex.add(opt.numAdd, addVecs.data());

        bool compFloat16 = useFloat16CoarseQuantizer;
        faiss::gpu::compareIndices(
                cpuIndex,
                gpuIndex,
                opt.numQuery,
                opt.dim,
                opt.k,
                opt.toString(),
                compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
                compFloat16 ? 0.70f : 0.1f,
                compFloat16 ? 0.30f : 0.015f);
    }
}

void copyToTest(bool useFloat16CoarseQuantizer) {
    Options opt;
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::gpu::RmmGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = opt.device;
    config.indicesOptions = opt.indicesOpt;
    config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;

    faiss::gpu::RaftIndexIVFFlat gpuIndex(
            &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
    gpuIndex.train(opt.numTrain, trainVecs.data());
    gpuIndex.add(opt.numAdd, addVecs.data());
    gpuIndex.setNumProbes(opt.nprobe);

    // use garbage values to see if we overwrite then
    faiss::IndexFlatL2 cpuQuantizer(1);
    faiss::IndexIVFFlat cpuIndex(&cpuQuantizer, 1, 1, faiss::METRIC_L2);
    cpuIndex.nprobe = 1;

    gpuIndex.copyTo(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);

    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
    EXPECT_EQ(cpuIndex.quantizer->d, gpuIndex.quantizer->d);
    EXPECT_EQ(cpuIndex.d, opt.dim);
    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.getNumProbes());

    testIVFEquality(cpuIndex, gpuIndex);

    // Query both objects; results should be equivalent
    bool compFloat16 = useFloat16CoarseQuantizer;
    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
            compFloat16 ? 0.70f : 0.1f,
            compFloat16 ? 0.30f : 0.015f);
}

void copyFromTest(bool useFloat16CoarseQuantizer) {
    Options opt;
    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);

    faiss::IndexFlatL2 cpuQuantizer(opt.dim);
    faiss::IndexIVFFlat cpuIndex(
            &cpuQuantizer, opt.dim, opt.numCentroids, faiss::METRIC_L2);
    cpuIndex.nprobe = opt.nprobe;
    cpuIndex.train(opt.numTrain, trainVecs.data());
    cpuIndex.add(opt.numAdd, addVecs.data());

    // use garbage values to see if we overwrite then
    faiss::gpu::RmmGpuResources res;
    res.noTempMemory();

    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = opt.device;
    config.indicesOptions = opt.indicesOpt;
    config.flatConfig.useFloat16 = useFloat16CoarseQuantizer;

    faiss::gpu::RaftIndexIVFFlat gpuIndex(&res, 1, 1, faiss::METRIC_L2, config);
    gpuIndex.setNumProbes(1);

    gpuIndex.copyFrom(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);

    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
    EXPECT_EQ(cpuIndex.d, opt.dim);
    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.getNumProbes());

    testIVFEquality(cpuIndex, gpuIndex);

    // Query both objects; results should be equivalent
    bool compFloat16 = useFloat16CoarseQuantizer;
    faiss::gpu::compareIndices(
            cpuIndex,
            gpuIndex,
            opt.numQuery,
            opt.dim,
            opt.k,
            opt.toString(),
            compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
            compFloat16 ? 0.70f : 0.1f,
            compFloat16 ? 0.30f : 0.015f);
}

//TEST(TestRaftIndexIVFFlat, Float32_32_Add_L2) {
//    addTest(faiss::METRIC_L2, false);
//    printf("Finished addTest(faiss::METRIC_L2, false)\n");
//}
//
//TEST(TestRaftIndexIVFFlat, Float32_32_Add_IP) {
//    addTest(faiss::METRIC_INNER_PRODUCT, false);
//    printf("Finished addTest(faiss::METRIC_INNER_PRODUCT, false)\n");
//}
//
//TEST(TestRaftIndexIVFFlat, Float16_32_Add_L2) {
//    addTest(faiss::METRIC_L2, true);
//    printf("Finished addTest(faiss::METRIC_L2, true)\n");
//}
//
//TEST(TestRaftIndexIVFFlat, Float16_32_Add_IP) {
//    addTest(faiss::METRIC_INNER_PRODUCT, true);
//    printf("Finished addTest(faiss::METRIC_INNER_PRODUCT, true)\n");
//}

//
// General query tests
//

TEST(TestRaftIndexIVFFlat, Float32_Query_L2) {
    queryTest(faiss::METRIC_L2, false);
    printf("Finished queryTest(faiss::METRIC_L2, false);\n");
}

//TEST(TestRaftIndexIVFFlat, Float32_Query_IP) {
//    queryTest(faiss::METRIC_INNER_PRODUCT, false);
//    printf("Finished queryTest(faiss::METRIC_INNER_PRODUCT, false)\n");
//}

// float16 coarse quantizer

TEST(TestRaftIndexIVFFlat, Float16_32_Query_L2) {
    queryTest(faiss::METRIC_L2, true);
    printf("Finished queryTest(faiss::METRIC_L2, true)\n");
}

//TEST(TestRaftIndexIVFFlat, Float16_32_Query_IP) {
//    queryTest(faiss::METRIC_INNER_PRODUCT, true);
//    printf("Finished queryTest(faiss::METRIC_INNER_PRODUCT, true)\n");
//}

//
// There are IVF list scanning specializations for 64-d and 128-d that we
// make sure we explicitly test here
//

TEST(TestRaftIndexIVFFlat, Float32_Query_L2_64) {
    queryTest(faiss::METRIC_L2, false, 64);
    printf("Finished queryTest(faiss::METRIC_L2, false, 64)\n");
}

//TEST(TestRaftIndexIVFFlat, Float32_Query_IP_64) {
//    queryTest(faiss::METRIC_INNER_PRODUCT, false, 64);
//    printf("Finished queryTest(faiss::METRIC_INNER_PRODUCT, false, 64)\n");
//}

TEST(TestRaftIndexIVFFlat, Float32_Query_L2_128) {
    queryTest(faiss::METRIC_L2, false, 128);
    printf("Finished queryTest(faiss::METRIC_L2, false, 128)\n");
}

//TEST(TestRaftIndexIVFFlat, Float32_Query_IP_128) {
//    queryTest(faiss::METRIC_INNER_PRODUCT, false, 128);
//    printf("Finished queryTest(faiss::METRIC_INNER_PRODUCT, false, 128)\n");
//}

//
// Copy tests
//

/** TODO: test crashes */
// TEST(TestRaftIndexIVFFlat, Float32_32_CopyTo) {
//     copyToTest(false);
//     printf("Finished copyToTest(false)\n");
// }

//TEST(TestRaftIndexIVFFlat, Float32_32_CopyFrom) {
//    copyFromTest(false);
//    printf("Finished copyFromTest(false)\n");
//}

//TEST(TestRaftIndexIVFFlat, Float32_negative) {
//    Options opt;
//
//    auto trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
//    auto addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
//
//    // Put all vecs on negative side
//    for (auto& f : trainVecs) {
//        f = std::abs(f) * -1.0f;
//    }
//
//    for (auto& f : addVecs) {
//        f *= std::abs(f) * -1.0f;
//    }
//
//    faiss::IndexFlatIP quantizerIP(opt.dim);
//    faiss::Index* quantizer = (faiss::Index*)&quantizerIP;
//
//    faiss::IndexIVFFlat cpuIndex(
//            quantizer, opt.dim, opt.numCentroids, faiss::METRIC_INNER_PRODUCT);
//    cpuIndex.train(opt.numTrain, trainVecs.data());
//    cpuIndex.add(opt.numAdd, addVecs.data());
//    cpuIndex.nprobe = opt.nprobe;
//
//    faiss::gpu::RmmGpuResources res;
//    res.noTempMemory();
//
//    faiss::gpu::GpuIndexIVFFlatConfig config;
//    config.device = opt.device;
//    config.indicesOptions = opt.indicesOpt;
//
//    faiss::gpu::RaftIndexIVFFlat gpuIndex(
//            &res, cpuIndex.d, cpuIndex.nlist, cpuIndex.metric_type, config);
//    gpuIndex.copyFrom(&cpuIndex);
//    gpuIndex.setNumProbes(opt.nprobe);
//
//    // Construct a positive test set
//    auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
//
//    // Put all vecs on positive size
//    for (auto& f : queryVecs) {
//        f = std::abs(f);
//    }
//
//    bool compFloat16 = false;
//    faiss::gpu::compareIndices(
//            queryVecs,
//            cpuIndex,
//            gpuIndex,
//            opt.numQuery,
//            opt.dim,
//            opt.k,
//            opt.toString(),
//            compFloat16 ? kF16MaxRelErr : kF32MaxRelErr,
//            // FIXME: the fp16 bounds are
//            // useless when math (the accumulator) is
//            // in fp16. Figure out another way to test
//            compFloat16 ? 0.99f : 0.1f,
//            compFloat16 ? 0.65f : 0.015f);
//}

//
// NaN tests
//

/** TODO: test crashes */
// TEST(TestRaftIndexIVFFlat, QueryNaN) {
//     Options opt;

//     std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain,
//     opt.dim); std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd,
//     opt.dim);

//     faiss::gpu::RmmGpuResources res;
//     res.noTempMemory();

//     faiss::gpu::GpuIndexIVFFlatConfig config;
//     config.device = opt.device;
//     config.indicesOptions = opt.indicesOpt;
//     config.flatConfig.useFloat16 = faiss::gpu::randBool();

//     faiss::gpu::RaftIndexIVFFlat gpuIndex(
//             &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
//     gpuIndex.setNumProbes(opt.nprobe);

//     gpuIndex.train(opt.numTrain, trainVecs.data());
//     gpuIndex.add(opt.numAdd, addVecs.data());

//     int numQuery = 10;
//     std::vector<float> nans(
//             numQuery * opt.dim, std::numeric_limits<float>::quiet_NaN());

//     std::vector<float> distances(numQuery * opt.k, 0);
//     std::vector<faiss::Index::idx_t> indices(numQuery * opt.k, 0);

//     gpuIndex.search(
//             numQuery, nans.data(), opt.k, distances.data(), indices.data());

//     for (int q = 0; q < numQuery; ++q) {
//         for (int k = 0; k < opt.k; ++k) {
//             EXPECT_EQ(indices[q * opt.k + k], -1);
//             EXPECT_EQ(
//                     distances[q * opt.k + k],
//                     std::numeric_limits<float>::max());
//         }
//     }
// }

/** TODO: test crashes */
// TEST(TestRaftIndexIVFFlat, AddNaN) {
//     Options opt;

//     faiss::gpu::RmmGpuResources res;
//     res.noTempMemory();

//     faiss::gpu::GpuIndexIVFFlatConfig config;
//     config.device = opt.device;
//     config.indicesOptions = opt.indicesOpt;
//     config.flatConfig.useFloat16 = faiss::gpu::randBool();

//     faiss::gpu::RaftIndexIVFFlat gpuIndex(
//             &res, opt.dim, opt.numCentroids, faiss::METRIC_L2, config);
//     gpuIndex.setNumProbes(opt.nprobe);

//     int numNans = 10;
//     std::vector<float> nans(
//             numNans * opt.dim, std::numeric_limits<float>::quiet_NaN());

//     // Make one vector valid (not the first vector, in order to test offset
//     // issues), which should actually add
//     for (int i = 0; i < opt.dim; ++i) {
//         nans[opt.dim + i] = i;
//     }

//     std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain,
//     opt.dim); gpuIndex.train(opt.numTrain, trainVecs.data());

//     // should not crash
//     EXPECT_EQ(gpuIndex.ntotal, 0);
//     gpuIndex.add(numNans, nans.data());

//     std::vector<float> queryVecs = faiss::gpu::randVecs(opt.numQuery,
//     opt.dim); std::vector<float> distance(opt.numQuery * opt.k, 0);
//     std::vector<faiss::Index::idx_t> indices(opt.numQuery * opt.k, 0);

//     // should not crash
//     gpuIndex.search(
//             opt.numQuery,
//             queryVecs.data(),
//             opt.k,
//             distance.data(),
//             indices.data());
// }

//TEST(TestRaftIndexIVFFlat, UnifiedMemory) {
//    // Construct on a random device to test multi-device, if we have
//    // multiple devices
//    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
//
//    if (!faiss::gpu::getFullUnifiedMemSupport(device)) {
//        return;
//    }
//
//    int dim = 128;
//
//    int numCentroids = 256;
//    // Unfortunately it would take forever to add 24 GB in IVFPQ data,
//    // so just perform a small test with data allocated in the unified
//    // memory address space
//    size_t numAdd = 10000;
//    size_t numTrain = numCentroids * 40;
//    int numQuery = 10;
//    int k = 10;
//    int nprobe = 8;
//
//    std::vector<float> trainVecs = faiss::gpu::randVecs(numTrain, dim);
//    std::vector<float> addVecs = faiss::gpu::randVecs(numAdd, dim);
//
//    faiss::IndexFlatL2 quantizer(dim);
//    faiss::IndexIVFFlat cpuIndex(
//            &quantizer, dim, numCentroids, faiss::METRIC_L2);
//
//    cpuIndex.train(numTrain, trainVecs.data());
//    cpuIndex.add(numAdd, addVecs.data());
//    cpuIndex.nprobe = nprobe;
//
//    faiss::gpu::RmmGpuResources res;
//    res.noTempMemory();
//
//    faiss::gpu::GpuIndexIVFFlatConfig config;
//    config.device = device;
//    config.memorySpace = faiss::gpu::MemorySpace::Unified;
//
//    faiss::gpu::RaftIndexIVFFlat gpuIndex(
//            &res, dim, numCentroids, faiss::METRIC_L2, config);
//    gpuIndex.copyFrom(&cpuIndex);
//    gpuIndex.setNumProbes(nprobe);
//
//    faiss::gpu::compareIndices(
//            cpuIndex,
//            gpuIndex,
//            numQuery,
//            dim,
//            k,
//            "Unified Memory",
//            kF32MaxRelErr,
//            0.1f,
//            0.015f);
//}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
