#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/utils/Timer.h>
#include <faiss/utils/random.h>
#include <omp.h>
#include <algorithm>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/MultiSequence.cuh>
#include <iostream>
#include <random>
#include <utility>

struct RandomContext {
    int64_t seed = 0;
};

template <typename T, typename TVec2>
inline void heap_pop(T& size, float* values, TVec2* ids) {
    values--;
    ids--;
    float val = values[size];
    int i = 1, i1, i2;
    i1 = i << 1;
    while (i1 <= size) {
        i2 = i1 + 1;
        if (i2 == size + 1 || values[i1] < values[i2]) {
            if (val < values[i1])
                break;
            values[i] = values[i1];
            ids[i] = ids[i1];
            i = i1;
        } else {
            if (val < values[i2])
                break;
            values[i] = values[i2];
            ids[i] = ids[i2];
            i = i2;
        }
        i1 = i << 1;
    }
    values[i] = values[size];
    ids[i] = ids[size];
    size--;
}

template <typename T, typename TVec2>
inline void heap_push(
        T& size,
        float* values,
        TVec2* ids,
        float val,
        T id1,
        T id2) {
    size++;
    values--;
    ids--;
    unsigned i = size, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (val >= values[i_father])
            break;
        values[i] = values[i_father];
        ids[i] = ids[i_father];
        i = i_father;
    }
    values[i] = val;
    ids[i] = {id1, id2};
}

template <typename T, typename TVec2>
void multiSequence(
        unsigned w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
    float heapValues[w];
    TVec2 heapIds[w];
    T traversed[w], heapSize;
    memset(traversed, 0, w * sizeof(T));
    heapSize = 0;

    dr[0] = d1[0] + d2[0];
    ir[0] = {i1[0], i2[0]};
    T i = 0, j = 0;

    for (unsigned currentW = 1; currentW < w; currentW++) {
        traversed[i] = j + 1;

        if (traversed[i + 1] == j) {
            heap_push<T, TVec2>(
                    heapSize, heapValues, heapIds, d1[i + 1] + d2[j], i + 1, j);
        }
        if (i == 0 || traversed[i - 1] > j + 1) {
            heap_push<T, TVec2>(
                    heapSize, heapValues, heapIds, d1[i] + d2[j + 1], i, j + 1);
        }

        dr[currentW] = heapValues[0];
        i = heapIds[0].x;
        j = heapIds[0].y;
        ir[currentW] = {i1[i], i2[j]};
        heap_pop<T, TVec2>(heapSize, heapValues, heapIds);
    }
}

template <typename T, typename TVec2>
void multiSequence(
        T length,
        unsigned w,
        float* d1,
        T* i1,
        float* d2,
        T* i2,
        float* dr,
        TVec2* ir) {
    float heapValues[length];
    TVec2 heapIds[length];
    T traversed[length];
    T heapSize;
    memset(traversed, 0, length * sizeof(T));
    heapSize = 0;

    dr[0] = d1[0] + d2[0];
    ir[0] = {i1[0], i2[0]};
    T i = 0, j = 0;
    for (unsigned currentW = 1; currentW < w; currentW++) {
        traversed[i] = j + 1;

        if (i < length - 1 && (traversed[i + 1] == j)) {
            heap_push<T, TVec2>(
                    heapSize, heapValues, heapIds, d1[i + 1] + d2[j], i + 1, j);
        }
        if (j < length - 1 && (i == 0 || traversed[i - 1] > j + 1)) {
            heap_push<T, TVec2>(
                    heapSize, heapValues, heapIds, d1[i] + d2[j + 1], i, j + 1);
        }

        dr[currentW] = heapValues[0];
        i = heapIds[0].x;
        j = heapIds[0].y;
        ir[currentW] = {i1[i], i2[j]};
        heap_pop<T, TVec2>(heapSize, heapValues, heapIds);
    }
}

template <typename T, typename TVec2>
float run(
        int w,
        int numOfQueries,
        unsigned short numCoarseDistances,
        bool isGpu,
        float* elapsedTimeKernel = nullptr) {
    constexpr int NUM_CODEBOOKS = 2;
    unsigned inputSize = (unsigned)numOfQueries * numCoarseDistances;
    unsigned outputSize = (unsigned)numOfQueries * w;
    RandomContext randomContext;

    float elapsedTime = 0;
    if (elapsedTimeKernel) {
        *elapsedTimeKernel = 0;
    }

    float *d, *d1, *d2;
    T *ids, *i1, *i2;
    float* dr;
    TVec2* ir;

    d = new float[NUM_CODEBOOKS * inputSize];
    ids = new T[NUM_CODEBOOKS * inputSize];
    dr = new float[outputSize];
    ir = new TVec2[outputSize];
    d1 = d;
    d2 = d + inputSize;
    i1 = ids;
    i2 = ids + inputSize;

    faiss::float_rand(d1, inputSize, randomContext.seed);
    faiss::float_rand(d2, inputSize, randomContext.seed);

    for (unsigned i = 0; i < numOfQueries; i++) {
        std::sort(
                d1 + i * numCoarseDistances,
                d1 + i * numCoarseDistances + numCoarseDistances);
        std::sort(
                d2 + i * numCoarseDistances,
                d2 + i * numCoarseDistances + numCoarseDistances);
    }

    for (unsigned i = 0; i < numOfQueries; i++) {
        for (unsigned j = 0; j < numCoarseDistances; j++) {
            i1[i * numCoarseDistances + j] = j;
            i2[i * numCoarseDistances + j] = j;
        }
    }

    for (unsigned i = 0; i < numOfQueries; i++) {
        std::random_shuffle(
                &i1[i * numCoarseDistances], &i1[(i + 1) * numCoarseDistances]);
        std::random_shuffle(
                &i2[i * numCoarseDistances], &i2[(i + 1) * numCoarseDistances]);
    }

    faiss::gpu::StandardGpuResources provider;
    int device = 0;
    cudaStream_t stream =
            provider.getResources()->getDefaultStreamCurrentDevice();

    int nRuns = 5;
    for (int i = 0; i < nRuns; i++) {
        if (isGpu) {
            faiss::gpu::CpuTimer timer;

            auto inDistances = faiss::gpu::toDeviceTemporary<float, 3>(
                    provider.getResources().get(),
                    device,
                    const_cast<float*>(d),
                    stream,
                    {NUM_CODEBOOKS, numOfQueries, numCoarseDistances});

            auto inIndices = faiss::gpu::toDeviceTemporary<T, 3>(
                    provider.getResources().get(),
                    device,
                    const_cast<T*>(ids),
                    stream,
                    {NUM_CODEBOOKS, numOfQueries, numCoarseDistances});

            faiss::gpu::DeviceTensor<float, 2, true> outDistances(
                    provider.getResources().get(),
                    faiss::gpu::makeTempAlloc(
                            faiss::gpu::AllocType::Other, stream),
                    {numOfQueries, w});

            faiss::gpu::DeviceTensor<TVec2, 2, true> outIndices(
                    provider.getResources().get(),
                    faiss::gpu::makeTempAlloc(
                            faiss::gpu::AllocType::Other, stream),
                    {numOfQueries, w});

            faiss::gpu::KernelTimer kernelTimer(stream);

            faiss::gpu::runMultiSequence2(
                    numOfQueries,
                    numCoarseDistances,
                    w,
                    inDistances,
                    inIndices,
                    outDistances,
                    outIndices,
                    provider.getResources().get());

            faiss::gpu::fromDevice<float, 2>(outDistances, dr, stream);
            faiss::gpu::fromDevice<TVec2, 2>(outIndices, ir, stream);

            faiss::gpu::CudaEvent copyEnd(stream);
            copyEnd.cpuWaitOnEvent();

            if (elapsedTimeKernel) {
                *elapsedTimeKernel += kernelTimer.elapsedMilliseconds();
            }

            elapsedTime += timer.elapsedMilliseconds();
        } else {
            faiss::gpu::CpuTimer timer;

#pragma omp parallel for
            for (unsigned i = 0; i < numOfQueries; i++) {
                if (w <= numCoarseDistances) {
                    multiSequence<T, TVec2>(
                            w,
                            d1 + i * numCoarseDistances,
                            i1 + i * numCoarseDistances,
                            d2 + i * numCoarseDistances,
                            i2 + i * numCoarseDistances,
                            dr + i * w,
                            ir + i * w);
                } else {
                    multiSequence<T, TVec2>(
                            numCoarseDistances,
                            w,
                            d1 + i * numCoarseDistances,
                            i1 + i * numCoarseDistances,
                            d2 + i * numCoarseDistances,
                            i2 + i * numCoarseDistances,
                            dr + i * w,
                            ir + i * w);
                }
            }

            elapsedTime += timer.elapsedMilliseconds();
        }
    }

    if (elapsedTimeKernel) {
        *elapsedTimeKernel /= nRuns;
    }

    elapsedTime /= nRuns;

    delete[] d;
    delete[] ids;
    delete[] dr;
    delete[] ir;
    return elapsedTime;
}

int main(int argc, char** argv) {
    std::vector<int> wList = {
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::vector<int> numQueriesList = {1, 1000, 10000, 100000};
    for (int i = 0; i < wList.size(); i++) {
        int w = wList[i];
        int k = w;
        std::cout << "w: " << w << std::endl;
        std::cout << "k: " << k << std::endl;
        for (int j = 0; j < numQueriesList.size(); j++) {
            int numQueries = numQueriesList[j];
            std::cout << "  numQueries: " << numQueries << std::endl;
            float elapsedTime, elapseTimeCpu;
            float elapsedTimeKernel;
            elapsedTime = run<ushort, ushort2>(
                    w, numQueries, k, true, &elapsedTimeKernel);
            std::cout << "    Elapsed time GPU: " << elapsedTime << std::endl;
            std::cout << "    Elapsed time Kernel: " << elapsedTimeKernel
                      << std::endl;

            for (int numOfThreads = 1; numOfThreads <= 16; numOfThreads *= 2) {
                omp_set_num_threads(numOfThreads);
                elapseTimeCpu =
                        run<ushort, ushort2>(w, numQueries, k, false, nullptr);
                std::cout << "    Num. of threads: " << numOfThreads
                          << std::endl;
                std::cout << "      Elapsed time CPU: " << elapseTimeCpu
                          << std::endl;
                std::cout << "      Speedup: " << elapseTimeCpu / elapsedTime
                          << std::endl;
                std::cout << "      Speedup Kernel: "
                          << elapseTimeCpu / elapsedTimeKernel << std::endl;
            }
        }
    }
    return 0;
}
