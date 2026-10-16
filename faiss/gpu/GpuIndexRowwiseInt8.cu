/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cublas_v2.h>
#include <faiss/IndexRowwiseInt8.h>
#include <faiss/gpu/GpuIndexRowwiseInt8.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <math_constants.h>
#include <algorithm>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceVector.cuh>
#include <limits>
#include <vector>

namespace faiss {
namespace gpu {

/// bank rows are padded to a multiple of this (int8 GEMM alignment); padding
/// rows have code 0, scale 0 and norm +inf, and are never returned
constexpr idx_t kRowPad = 64;
constexpr int kThreads = 256;

struct RowwiseInt8Data {
    RowwiseInt8Data(
            GpuResources* res,
            int device,
            MemorySpace space,
            cudaStream_t stream)
            : codes(res, AllocInfo(AllocType::FlatData, device, space, stream)),
              scales(res,
                     AllocInfo(AllocType::FlatData, device, space, stream)),
              norms(res,
                    AllocInfo(AllocType::FlatData, device, space, stream)) {}

    DeviceVector<int8_t> codes; // (rows, d), rows = num rounded up to kRowPad
    DeviceVector<float> scales; // (rows)
    DeviceVector<float> norms;  // (rows): s^2 |c|^2, +inf on padding rows
    idx_t num = 0;              // real vectors
};

namespace {

// block-wide reduction for blockDim.x a multiple of 32 (<= 1024)
template <typename T, typename Op>
__device__ T blockReduce(T v, Op op, T* smem) {
    for (int o = 16; o > 0; o >>= 1) {
        v = op(v, __shfl_xor_sync(0xffffffff, v, o));
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        smem[warp] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        T r = smem[0];
        for (int w = 1; w < (blockDim.x >> 5); w++) {
            r = op(r, smem[w]);
        }
        smem[0] = r;
    }
    __syncthreads();
    T r = smem[0];
    __syncthreads(); // smem may be reused
    return r;
}

struct MaxOp {
    __device__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

struct SumOp {
    __device__ int operator()(int a, int b) const {
        return a + b;
    }
};

// one block per row: s = max|x| / 127, c = round(x / s) in [-127, 127],
// norm = s^2 |c|^2 (same arithmetic as rowwise_int8_encode on the CPU)
__global__ void rowwiseEncode(
        const float* x,
        int d,
        int8_t* codes,
        float* scales,
        float* norms) {
    __shared__ float smaxBuf[32];
    __shared__ int ssumBuf[32];
    idx_t i = blockIdx.x;
    const float* xi = x + i * d;

    float m = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        m = fmaxf(m, fabsf(xi[j]));
    }
    m = blockReduce(m, MaxOp(), smaxBuf);
    float s = __fdiv_rn(m, 127.0f);
    if (s == 0.0f) {
        s = 1.0f;
    }

    int acc = 0;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float v = rintf(__fdiv_rn(xi[j], s)); // round half to even
        v = fminf(127.0f, fmaxf(-127.0f, v));
        int c = int(v);
        codes[i * d + j] = int8_t(c);
        acc += c * c;
    }
    acc = blockReduce(acc, SumOp(), ssumBuf);
    if (threadIdx.x == 0) {
        scales[i] = s;
        if (norms) {
            norms[i] = __fmul_rn(float(acc), __fmul_rn(s, s));
        }
    }
}

// one block per row: norm = s^2 |c|^2
__global__ void rowwiseNorms(
        const int8_t* codes,
        const float* scales,
        int d,
        float* norms) {
    __shared__ int ssumBuf[32];
    idx_t i = blockIdx.x;
    int acc = 0;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        int c = codes[i * d + j];
        acc += c * c;
    }
    acc = blockReduce(acc, SumOp(), ssumBuf);
    if (threadIdx.x == 0) {
        float s = scales[i];
        norms[i] = __fmul_rn(float(acc), __fmul_rn(s, s));
    }
}

__global__ void fillFloat(float* p, idx_t n, float v) {
    idx_t i = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = v;
    }
}

// in place, for one (query tile x bank tile) block of int32 dot products
// c_q . c_b (row-major, nb columns):
//   L2: |b|^2 - bias s_b^2 - 2 s_q s_b (c_q . c_b)
//       (|q|^2 - bias s_q^2 is added after selection)
//   IP: s_q s_b (c_q . c_b)
// padding rows (norm +inf) become +inf (L2) / -inf (IP)
__global__ void dotsToDistances(
        void* buf,
        idx_t nb,
        const float* qScales,
        const float* bScales,
        const float* bNorms,
        bool l2,
        float bias) {
    idx_t j = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= nb) {
        return;
    }
    idx_t e = idx_t(blockIdx.y) * nb + j;
    int dot = ((const int*)buf)[e];
    float nbj = bNorms[j];
    float out;
    if (isinf(nbj)) {
        out = l2 ? CUDART_INF_F : -CUDART_INF_F;
    } else {
        float ip = __fmul_rn(
                __fmul_rn(qScales[blockIdx.y], bScales[j]), float(dot));
        // all-zero vectors (s = 1 by convention) have no rounding error
        if (bias != 0 && nbj > 0) { // single rounding, as std::fma on the CPU
            nbj = __fmaf_rn(-bias, __fmul_rn(bScales[j], bScales[j]), nbj);
        }
        out = l2 ? __fsub_rn(nbj, __fmul_rn(2.0f, ip)) : ip;
    }
    ((float*)buf)[e] = out;
}

// norms[i] -= bias s_i^2 (L2 rounding-bias correction of the queries), except
// for all-zero vectors (norm 0), which have no rounding error
__global__ void subtractBias(
        float* norms,
        const float* scales,
        idx_t n,
        float bias) {
    idx_t i = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && norms[i] > 0) {
        norms[i] = __fmaf_rn(-bias, __fmul_rn(scales[i], scales[i]), norms[i]);
    }
}

// add |q|^2 (L2), clamp at 0, and turn padding / missing results into -1
__global__ void finalizeResults(
        float* D,
        idx_t* I,
        idx_t total,
        int k,
        const float* qNorms,
        idx_t num,
        bool l2) {
    idx_t e = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) {
        return;
    }
    idx_t id = I[e];
    if (id < 0 || id >= num) {
        I[e] = -1;
        D[e] = l2 ? CUDART_INF_F : -CUDART_INF_F;
    } else if (l2) {
        D[e] = fmaxf(__fadd_rn(D[e], qNorms[e / k]), 0.0f);
    }
}

__global__ void fillEmpty(float* D, idx_t* I, idx_t total, bool l2) {
    idx_t e = idx_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e < total) {
        D[e] = l2 ? CUDART_INF_F : -CUDART_INF_F;
        I[e] = -1;
    }
}

inline unsigned int blocksFor(idx_t n) {
    return (unsigned int)utils::divUp(n, (idx_t)kThreads);
}

} // namespace

GpuIndexRowwiseInt8::GpuIndexRowwiseInt8(
        GpuResourcesProvider* provider,
        int dims,
        faiss::MetricType metric,
        GpuIndexRowwiseInt8Config config)
        : GpuIndex(provider->getResources(), dims, metric, 0.0f, config),
          rwConfig_(config) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "GpuIndexRowwiseInt8: only L2 and inner product");
    FAISS_THROW_IF_NOT_MSG(
            dims % 4 == 0, "GpuIndexRowwiseInt8: dims must be a multiple of 4");
    FAISS_THROW_IF_NOT(config.tileQueries > 0 && config.tileQueries <= 65535);
    FAISS_THROW_IF_NOT(config.tileBank >= kRowPad);
    DeviceScope scope(config_.device);
    this->is_trained = true;
    data_.reset(new RowwiseInt8Data(
            resources_.get(),
            config_.device,
            config_.memorySpace,
            resources_->getDefaultStream(config_.device)));
}

GpuIndexRowwiseInt8::GpuIndexRowwiseInt8(
        GpuResourcesProvider* provider,
        const faiss::IndexRowwiseInt8* index,
        GpuIndexRowwiseInt8Config config)
        : GpuIndexRowwiseInt8(provider, index->d, index->metric_type, config) {
    copyFrom(index);
}

GpuIndexRowwiseInt8::~GpuIndexRowwiseInt8() {}

void GpuIndexRowwiseInt8::copyFrom(const faiss::IndexRowwiseInt8* index) {
    DeviceScope scope(config_.device);
    FAISS_THROW_IF_NOT(index->d == this->d);
    GpuIndex::copyFrom(index);
    reset();
    const idx_t chunk = 65536;
    std::vector<int8_t> c(chunk * d);
    std::vector<float> s(chunk);
    for (idx_t i0 = 0; i0 < index->ntotal; i0 += chunk) {
        idx_t n = std::min(chunk, index->ntotal - i0);
        index->get_int8(i0, n, c.data(), s.data());
        add_int8(n, c.data(), s.data());
    }
}

void GpuIndexRowwiseInt8::copyTo(faiss::IndexRowwiseInt8* index) const {
    DeviceScope scope(config_.device);
    GpuIndex::copyTo(index);
    index->code_size = this->d + sizeof(float);
    index->codes.clear();
    index->ntotal = 0;
    const idx_t chunk = 65536;
    std::vector<int8_t> c(chunk * d);
    std::vector<float> s(chunk);
    for (idx_t i0 = 0; i0 < this->ntotal; i0 += chunk) {
        idx_t n = std::min(chunk, this->ntotal - i0);
        get_int8(i0, n, c.data(), s.data());
        index->add_int8(n, c.data(), s.data());
    }
}

void GpuIndexRowwiseInt8::reset() {
    DeviceScope scope(config_.device);
    data_->codes.clear();
    data_->scales.clear();
    data_->norms.clear();
    data_->num = 0;
    this->ntotal = 0;
}

bool GpuIndexRowwiseInt8::addImplRequiresIDs_() const {
    return false;
}

void GpuIndexRowwiseInt8::addImpl_(idx_t n, const float* x, const idx_t* ids) {
    // current device already set, x is on the device
    FAISS_THROW_IF_NOT_MSG(!ids, "add_with_ids not supported");
    if (n == 0) {
        return;
    }
    auto stream = resources_->getDefaultStreamCurrentDevice();
    DeviceTensor<int8_t, 2, true> c(
            resources_.get(), makeTempAlloc(AllocType::Other, stream), {n, d});
    DeviceTensor<float, 1, true> s(
            resources_.get(), makeTempAlloc(AllocType::Other, stream), {n});
    rowwiseEncode<<<(unsigned int)n, kThreads, 0, stream>>>(
            x, d, c.data(), s.data(), nullptr);
    CUDA_TEST_ERROR();
    addCodes_(n, c.data(), s.data());
}

void GpuIndexRowwiseInt8::add_int8(
        idx_t n,
        const int8_t* codes,
        const float* scales) {
    DeviceScope scope(config_.device);
    if (n == 0) {
        return;
    }
    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto c = toDeviceTemporary<int8_t, 2>(
            resources_.get(),
            config_.device,
            const_cast<int8_t*>(codes),
            stream,
            {n, d});
    auto s = toDeviceTemporary<float, 1>(
            resources_.get(),
            config_.device,
            const_cast<float*>(scales),
            stream,
            {n});
    addCodes_(n, c.data(), s.data());
}

void GpuIndexRowwiseInt8::addCodes_(
        idx_t n,
        const int8_t* codesDev,
        const float* scalesDev) {
    auto stream = resources_->getDefaultStreamCurrentDevice();
    auto& D = *data_;
    idx_t num = D.num;
    idx_t newNum = num + n;
    idx_t rows = utils::roundUp(newNum, kRowPad);

    // drop the old padding, reserve the exact new size, append
    D.codes.resize(num * d, stream);
    D.scales.resize(num, stream);
    D.norms.resize(num, stream);
    D.codes.reserve(rows * d, stream);
    D.scales.reserve(rows, stream);
    D.norms.reserve(rows, stream);
    D.codes.append(codesDev, n * d, stream, true);
    D.scales.append(scalesDev, n, stream, true);
    D.norms.resize(newNum, stream);
    rowwiseNorms<<<(unsigned int)n, kThreads, 0, stream>>>(
            D.codes.data() + num * d,
            D.scales.data() + num,
            d,
            D.norms.data() + num);
    CUDA_TEST_ERROR();

    // padding rows
    idx_t pad = rows - newNum;
    D.codes.resize(rows * d, stream);
    D.scales.resize(rows, stream);
    D.norms.resize(rows, stream);
    if (pad > 0) {
        CUDA_VERIFY(cudaMemsetAsync(
                D.codes.data() + newNum * d, 0, pad * d, stream));
        CUDA_VERIFY(cudaMemsetAsync(
                D.scales.data() + newNum, 0, pad * sizeof(float), stream));
        fillFloat<<<blocksFor(pad), kThreads, 0, stream>>>(
                D.norms.data() + newNum,
                pad,
                std::numeric_limits<float>::infinity());
        CUDA_TEST_ERROR();
    }
    D.num = newNum;
    this->ntotal = newNum;
}

void GpuIndexRowwiseInt8::searchImpl_(
        idx_t n,
        const float* x,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // current device already set; x, distances, labels are on the device
    FAISS_THROW_IF_NOT_MSG(
            !params || !params->sel,
            "GpuIndexRowwiseInt8: search with an IDSelector is not supported");
    searchTiles_(
            n,
            x,
            nullptr,
            nullptr,
            k,
            distances,
            labels,
            rowwise_int8_bias_factor(params, d, metric_type));
}

void GpuIndexRowwiseInt8::search_int8(
        idx_t n,
        const int8_t* codes,
        const float* scales,
        int k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    DeviceScope scope(config_.device);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !params || !params->sel,
            "GpuIndexRowwiseInt8: search with an IDSelector is not supported");
    const float bias = rowwise_int8_bias_factor(params, d, metric_type);
    if (n == 0) {
        return;
    }
    auto res = resources_.get();
    auto stream = resources_->getDefaultStream(config_.device);
    auto qc = toDeviceTemporary<int8_t, 2>(
            res, config_.device, const_cast<int8_t*>(codes), stream, {n, d});
    auto qs = toDeviceTemporary<float, 1>(
            res, config_.device, const_cast<float*>(scales), stream, {n});
    auto outD = toDeviceTemporary<float, 2>(
            res, config_.device, distances, stream, {n, idx_t(k)});
    auto outI = toDeviceTemporary<idx_t, 2>(
            res, config_.device, labels, stream, {n, idx_t(k)});
    searchTiles_(
            n,
            nullptr,
            qc.data(),
            qs.data(),
            k,
            outD.data(),
            outI.data(),
            bias);
    fromDevice<float, 2>(outD, distances, stream);
    fromDevice<idx_t, 2>(outI, labels, stream);
}

void GpuIndexRowwiseInt8::searchTiles_(
        idx_t n,
        const float* x,
        const int8_t* qc,
        const float* qs,
        int k,
        float* distances,
        idx_t* labels,
        float bias) const {
    FAISS_THROW_IF_NOT_FMT(
            k <= GPU_MAX_SELECTION_K, "k must be <= %d", GPU_MAX_SELECTION_K);
    auto res = resources_.get();
    auto stream = resources_->getDefaultStreamCurrentDevice();
    const bool l2 = metric_type == METRIC_L2;
    const auto& D = *data_;

    if (D.num == 0) {
        fillEmpty<<<blocksFor(n * k), kThreads, 0, stream>>>(
                distances, labels, n * k, l2);
        CUDA_TEST_ERROR();
        return;
    }

    const idx_t rows = D.codes.size() / d;
    const idx_t tileQ = std::min(idx_t(rwConfig_.tileQueries), n);
    const idx_t tileB =
            std::min((idx_t(rwConfig_.tileBank) / kRowPad) * kRowPad, rows);
    const idx_t numBankTiles = utils::divUp(rows, tileB);

    DeviceTensor<int8_t, 2, true> qCodes(
            res, makeTempAlloc(AllocType::Other, stream), {tileQ, idx_t(d)});
    DeviceTensor<float, 1, true> qScales(
            res, makeTempAlloc(AllocType::Other, stream), {tileQ});
    DeviceTensor<float, 1, true> qNorms(
            res, makeTempAlloc(AllocType::Other, stream), {tileQ});
    // int32 dot products, turned into float distances in place
    DeviceTensor<int32_t, 1, true> dots(
            res, makeTempAlloc(AllocType::Other, stream), {tileQ * tileB});
    // per bank tile k best, when there is more than one bank tile
    idx_t bufCols = numBankTiles > 1 ? numBankTiles * k : 1;
    DeviceTensor<float, 2, true> bufD(
            res, makeTempAlloc(AllocType::Other, stream), {tileQ, bufCols});
    DeviceTensor<idx_t, 2, true> bufI(
            res, makeTempAlloc(AllocType::Other, stream), {tileQ, bufCols});

    auto handle = resources_->getBlasHandleCurrentDevice();
    cublasSetStream(handle, stream);
    const int32_t alpha = 1, beta = 0;

    for (idx_t q0 = 0; q0 < n; q0 += tileQ) {
        idx_t curQ = std::min(tileQ, n - q0);
        const int8_t* tq; // this tile's int8 queries and scales
        const float* ts;
        if (x) {
            rowwiseEncode<<<(unsigned int)curQ, kThreads, 0, stream>>>(
                    x + q0 * d,
                    d,
                    qCodes.data(),
                    qScales.data(),
                    qNorms.data());
            tq = qCodes.data();
            ts = qScales.data();
        } else {
            tq = qc + q0 * d;
            ts = qs + q0;
            rowwiseNorms<<<(unsigned int)curQ, kThreads, 0, stream>>>(
                    tq, ts, d, qNorms.data());
        }
        if (bias != 0) {
            subtractBias<<<blocksFor(curQ), kThreads, 0, stream>>>(
                    qNorms.data(), ts, curQ, bias);
        }
        CUDA_TEST_ERROR();

        Tensor<float, 2, true> outD(distances + q0 * k, {curQ, idx_t(k)});
        Tensor<idx_t, 2, true> outI(labels + q0 * k, {curQ, idx_t(k)});
        Tensor<float, 2, true> rowD(bufD.data(), {curQ, bufCols});
        Tensor<idx_t, 2, true> rowI(bufI.data(), {curQ, bufCols});

        for (idx_t t = 0; t < numBankTiles; t++) {
            idx_t b0 = t * tileB;
            idx_t curB = std::min(tileB, rows - b0); // multiple of kRowPad

            // C (column-major curB x curQ, ldc = curB) = A^T B with
            // A = bank codes (column-major d x curB), B = query codes
            // (column-major d x curQ): row-major (curQ, curB) dot products
            cublasStatus_t st = cublasGemmEx(
                    handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    int(curB),
                    int(curQ),
                    d,
                    &alpha,
                    D.codes.data() + b0 * d,
                    CUDA_R_8I,
                    d,
                    tq,
                    CUDA_R_8I,
                    d,
                    &beta,
                    dots.data(),
                    CUDA_R_32I,
                    int(curB),
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT);
            FAISS_THROW_IF_NOT_FMT(
                    st == CUBLAS_STATUS_SUCCESS,
                    "cublasGemmEx (int8 x int8 -> int32) failed: %d",
                    int(st));

            dim3 grid(blocksFor(curB), (unsigned int)curQ);
            dotsToDistances<<<grid, kThreads, 0, stream>>>(
                    dots.data(),
                    curB,
                    ts,
                    D.scales.data() + b0,
                    D.norms.data() + b0,
                    l2,
                    bias);
            CUDA_TEST_ERROR();

            Tensor<float, 2, true> dist((float*)dots.data(), {curQ, curB});
            if (numBankTiles == 1) {
                runBlockSelect(dist, outD, outI, !l2, k, stream);
            } else {
                auto colD = rowD.narrow(1, t * k, k);
                auto colI = rowI.narrow(1, t * k, k);
                runBlockSelect(dist, colD, colI, !l2, k, stream);
            }
        }

        if (numBankTiles > 1) {
            // tile-relative ids -> bank ids, then the k best over all tiles
            runIncrementIndex(rowI, k, tileB, stream);
            runBlockSelectPair(rowD, rowI, outD, outI, !l2, k, stream);
        }

        finalizeResults<<<blocksFor(curQ * k), kThreads, 0, stream>>>(
                outD.data(),
                outI.data(),
                curQ * k,
                k,
                qNorms.data(),
                D.num,
                l2);
        CUDA_TEST_ERROR();
    }
}

void GpuIndexRowwiseInt8::get_int8(
        idx_t i0,
        idx_t n,
        int8_t* codes,
        float* scales) const {
    DeviceScope scope(config_.device);
    FAISS_THROW_IF_NOT(i0 >= 0 && n >= 0 && i0 + n <= this->ntotal);
    if (n == 0) {
        return;
    }
    auto stream = resources_->getDefaultStream(config_.device);
    CUDA_VERIFY(cudaMemcpyAsync(
            codes,
            data_->codes.data() + i0 * d,
            n * d,
            cudaMemcpyDefault,
            stream));
    CUDA_VERIFY(cudaMemcpyAsync(
            scales,
            data_->scales.data() + i0,
            n * sizeof(float),
            cudaMemcpyDefault,
            stream));
    CUDA_VERIFY(cudaStreamSynchronize(stream));
}

void GpuIndexRowwiseInt8::reconstruct_n(idx_t i0, idx_t n, float* out) const {
    std::vector<int8_t> c(n * d);
    std::vector<float> s(n);
    get_int8(i0, n, c.data(), s.data());
    std::vector<float> x(n * d);
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            x[i * d + j] = float(c[i * d + j]) * s[i];
        }
    }
    CUDA_VERIFY(cudaMemcpy(
            out, x.data(), n * d * sizeof(float), cudaMemcpyDefault));
}

void GpuIndexRowwiseInt8::reconstruct(idx_t key, float* out) const {
    reconstruct_n(key, 1, out);
}

size_t GpuIndexRowwiseInt8::getBankBytes() const {
    return data_->codes.capacity() * sizeof(int8_t) +
            data_->scales.capacity() * sizeof(float) +
            data_->norms.capacity() * sizeof(float);
}

} // namespace gpu
} // namespace faiss
