/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
        GpuResources* res);

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

template <int THREADS, int NUM_WARP_Q, int NUM_THREAD_Q>
void ivfInterleavedScanImpl(
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        const int k,
        faiss::MetricType metric_name,
        const bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res) {
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

} // namespace gpu
} // namespace faiss
