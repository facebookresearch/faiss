/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>

namespace faiss {
namespace gpu {

template <
        typename CODEC_TYPE,
        typename METRIC_TYPE,
        int THREADS,
        int NUM_WARP_Q,
        int NUM_THREAD_Q>
void IVFINT_RUN(
        CODEC_TYPE& codec,
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        const int k,
        METRIC_TYPE metric,
        const bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res) {
    const auto nq = queries.getSize(0);
    const auto dim = queries.getSize(1);
    const auto nprobe = listIds.getSize(1);

    const auto stream = res->getDefaultStreamCurrentDevice();

    DeviceTensor<float, 3, true> distanceTemp(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), listIds.getSize(1), k});
    DeviceTensor<idx_t, 3, true> indicesTemp(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), listIds.getSize(1), k});

    const dim3 grid(nprobe, std::min(nq, (idx_t)getMaxGridCurrentDevice().y));

    if (useResidual) {
        ivfInterleavedScan<
                CODEC_TYPE,
                METRIC_TYPE,
                THREADS,
                NUM_WARP_Q,
                NUM_THREAD_Q,
                true><<<grid, THREADS, codec.getSmemSize(dim), stream>>>(
                queries,
                residualBase,
                listIds,
                listData.data(),
                listLengths.data(),
                codec,
                metric,
                k,
                distanceTemp,
                indicesTemp);
    } else {
        ivfInterleavedScan<
                CODEC_TYPE,
                METRIC_TYPE,
                THREADS,
                NUM_WARP_Q,
                NUM_THREAD_Q,
                false><<<grid, THREADS, codec.getSmemSize(dim), stream>>>(
                queries,
                residualBase,
                listIds,
                listData.data(),
                listLengths.data(),
                codec,
                metric,
                k,
                distanceTemp,
                indicesTemp);
    }

    runIVFInterleavedScan2(
            distanceTemp,
            indicesTemp,
            listIds,
            k,
            listIndices,
            indicesOptions,
            METRIC_TYPE::kDirection,
            outDistances,
            outIndices,
            stream);
}

template <typename METRIC_TYPE, int THREADS, int NUM_WARP_Q, int NUM_THREAD_Q>
void IVFINT_CODECS(
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        const int k,
        METRIC_TYPE metric,
        const bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res) {
    const auto dim = queries.getSize(1);

    const auto call_ivfint_run = [&](const auto& func, auto& codec) {
        func(codec,
             queries,
             listIds,
             listData,
             listIndices,
             indicesOptions,
             listLengths,
             k,
             metric,
             useResidual,
             residualBase,
             scalarQ,
             outDistances,
             outIndices,
             res);
    };

    if (!scalarQ) {
        using CodecT = CodecFloat;
        CodecT codec(dim * sizeof(float));
        call_ivfint_run(
                IVFINT_RUN<
                        CodecT,
                        METRIC_TYPE,
                        THREADS,
                        NUM_WARP_Q,
                        NUM_THREAD_Q>,
                codec);
    } else {
        switch (scalarQ->qtype) {
            case ScalarQuantizer::QuantizerType::QT_8bit: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_8bit, 1>;
                CodecT codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);
                call_ivfint_run(
                        IVFINT_RUN<
                                CodecT,
                                METRIC_TYPE,
                                THREADS,
                                NUM_WARP_Q,
                                NUM_THREAD_Q>,
                        codec);
            } break;
            case ScalarQuantizer::QuantizerType::QT_8bit_uniform: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_8bit_uniform,
                              1>;
                CodecT codec(
                        scalarQ->code_size,
                        scalarQ->trained[0],
                        scalarQ->trained[1]);
                call_ivfint_run(
                        IVFINT_RUN<
                                CodecT,
                                METRIC_TYPE,
                                THREADS,
                                NUM_WARP_Q,
                                NUM_THREAD_Q>,
                        codec);
            } break;
            case ScalarQuantizer::QuantizerType::QT_fp16: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_fp16, 1>;
                CodecT codec(scalarQ->code_size);
                call_ivfint_run(
                        IVFINT_RUN<
                                CodecT,
                                METRIC_TYPE,
                                THREADS,
                                NUM_WARP_Q,
                                NUM_THREAD_Q>,
                        codec);
            } break;
            case ScalarQuantizer::QuantizerType::QT_8bit_direct: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_8bit_direct,
                              1>;
                Codec<ScalarQuantizer::QuantizerType::QT_8bit_direct, 1> codec(
                        scalarQ->code_size);
                call_ivfint_run(
                        IVFINT_RUN<
                                CodecT,
                                METRIC_TYPE,
                                THREADS,
                                NUM_WARP_Q,
                                NUM_THREAD_Q>,
                        codec);
            } break;
            case ScalarQuantizer::QuantizerType::QT_6bit: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_6bit, 1>;
                Codec<ScalarQuantizer::QuantizerType::QT_6bit, 1> codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);
                call_ivfint_run(
                        IVFINT_RUN<
                                CodecT,
                                METRIC_TYPE,
                                THREADS,
                                NUM_WARP_Q,
                                NUM_THREAD_Q>,
                        codec);
            } break;
            case ScalarQuantizer::QuantizerType::QT_4bit: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1>;
                Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1> codec(
                        scalarQ->code_size,
                        scalarQ->gpuTrained.data(),
                        scalarQ->gpuTrained.data() + dim);
                call_ivfint_run(
                        IVFINT_RUN<
                                CodecT,
                                METRIC_TYPE,
                                THREADS,
                                NUM_WARP_Q,
                                NUM_THREAD_Q>,
                        codec);
            } break;
            case ScalarQuantizer::QuantizerType::QT_4bit_uniform: {
                using CodecT =
                        Codec<ScalarQuantizer::QuantizerType::QT_4bit_uniform,
                              1>;
                Codec<ScalarQuantizer::QuantizerType::QT_4bit_uniform, 1> codec(
                        scalarQ->code_size,
                        scalarQ->trained[0],
                        scalarQ->trained[1]);
                call_ivfint_run(
                        IVFINT_RUN<
                                CodecT,
                                METRIC_TYPE,
                                THREADS,
                                NUM_WARP_Q,
                                NUM_THREAD_Q>,
                        codec);
            } break;
            default:
                FAISS_ASSERT(false);
        }
    }
}

#define IVF_INTERLEAVED_SCAN_IMPL_ARGS     \
    (Tensor<float, 2, true> & queries,     \
     Tensor<idx_t, 2, true> & listIds,     \
     DeviceVector<void*> & listData,       \
     DeviceVector<void*> & listIndices,    \
     IndicesOptions indicesOptions,        \
     DeviceVector<idx_t> & listLengths,    \
     const int k,                          \
     faiss::MetricType metric_name,        \
     const bool useResidual,               \
     Tensor<float, 3, true>& residualBase, \
     GpuScalarQuantizer* scalarQ,          \
     Tensor<float, 2, true>& outDistances, \
     Tensor<idx_t, 2, true>& outIndices,   \
     GpuResources* res)

template <int THREADS, int NUM_WARP_Q, int NUM_THREAD_Q>
void IVF_METRICS IVF_INTERLEAVED_SCAN_IMPL_ARGS {
    FAISS_ASSERT(k <= NUM_WARP_Q);

    const auto call_codec = [&](const auto& func, const auto& metric) {
        func(queries,
             listIds,
             listData,
             listIndices,
             indicesOptions,
             listLengths,
             k,
             metric,
             useResidual,
             residualBase,
             scalarQ,
             outDistances,
             outIndices,
             res);
    };

    if (metric_name == MetricType::METRIC_L2) {
        L2Distance metric;
        call_codec(
                IVFINT_CODECS<L2Distance, THREADS, NUM_WARP_Q, NUM_THREAD_Q>,
                metric);
    } else if (metric_name == MetricType::METRIC_INNER_PRODUCT) {
        IPDistance metric;
        call_codec(
                IVFINT_CODECS<IPDistance, THREADS, NUM_WARP_Q, NUM_THREAD_Q>,
                metric);
    } else {
        FAISS_ASSERT(false);
    }

    CUDA_TEST_ERROR();
}

template <int THREADS, int NUM_WARP_Q, int NUM_THREAD_Q>
void ivfInterleavedScanImpl IVF_INTERLEAVED_SCAN_IMPL_ARGS;

#define IVF_INTERLEAVED_IMPL_HELPER(THREADS, NUM_WARP_Q, NUM_THREAD_Q) \
    template <>                                                        \
    void ivfInterleavedScanImpl<THREADS, NUM_WARP_Q, NUM_THREAD_Q>     \
            IVF_INTERLEAVED_SCAN_IMPL_ARGS {                           \
        IVF_METRICS<THREADS, NUM_WARP_Q, NUM_THREAD_Q>(                \
                queries,                                               \
                listIds,                                               \
                listData,                                              \
                listIndices,                                           \
                indicesOptions,                                        \
                listLengths,                                           \
                k,                                                     \
                metric_name,                                           \
                useResidual,                                           \
                residualBase,                                          \
                scalarQ,                                               \
                outDistances,                                          \
                outIndices,                                            \
                res);                                                  \
    }

#define IVF_INTERLEAVED_IMPL(...) IVF_INTERLEAVED_IMPL_HELPER(__VA_ARGS__)

// clang-format off
#define IVFINTERLEAVED_1_PARAMS    128,1,1
#define IVFINTERLEAVED_32_PARAMS   128,32,2
#define IVFINTERLEAVED_64_PARAMS   128,64,3
#define IVFINTERLEAVED_128_PARAMS  128,128,3
#define IVFINTERLEAVED_256_PARAMS  128,256,4
#define IVFINTERLEAVED_512_PARAMS  128,512,8
#define IVFINTERLEAVED_1024_PARAMS 128,1024,8
#define IVFINTERLEAVED_2048_PARAMS  64,2048,8
// clang-format on

} // namespace gpu
} // namespace faiss
