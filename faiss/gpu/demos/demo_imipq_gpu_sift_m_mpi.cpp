/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/Index.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIMIPQ.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/utils/Timer.h>
#include <faiss/impl/ThreadedIndex.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <faiss/utils/vecs_storage.h>
#include <mpi.h>
#include <omp.h>
#include <sys/types.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include <cuda_profiler_api.h>

void synchronizeDevices(int deviceIdInit, int nGpus) {
    for (int i = deviceIdInit; i < nGpus; ++i) {
        faiss::gpu::DeviceScope scope(i);
        CUDA_VERIFY(cudaDeviceSynchronize());
    }
}

void processPrint(int processRank, std::string str) {
    std::stringstream out;
    out << "Process " << processRank << " << " << str << std::endl;
    std::cout << out.str();
}

void processPrint(int processRank, std::stringstream& str) {
    processPrint(processRank, str.str());
}

template <class C>
void merge_tables(
        long n,
        long k,
        int nProcesses,
        float* distances,
        faiss::idx_t* labels,
        const std::vector<float>& all_distances,
        const std::vector<faiss::idx_t>& all_labels,
        const std::vector<long>& translations) {
    if (k == 0) {
        return;
    }

    long stride = n * k;
#pragma omp parallel
    {
        std::vector<int> buf(2 * nProcesses);
        int* pointer = buf.data();
        int* processIds = pointer + nProcesses;
        std::vector<float> buf2(nProcesses);
        float* heap_vals = buf2.data();
#pragma omp for
        for (long i = 0; i < n; i++) {
            // the heap maps values to the process where they are
            // produced.
            const float* D_in = all_distances.data() + i * k;
            const faiss::idx_t* I_in = all_labels.data() + i * k;
            int heap_size = 0;

            for (long currRank = 0; currRank < nProcesses; currRank++) {
                pointer[currRank] = 0;
                if (I_in[stride * currRank] >= 0) {
                    faiss::heap_push<C>(
                            ++heap_size,
                            heap_vals,
                            processIds,
                            D_in[stride * currRank],
                            currRank);
                }
            }

            float* D = distances + i * k;
            faiss::idx_t* I = labels + i * k;

            for (int j = 0; j < k; j++) {
                if (heap_size == 0) {
                    I[j] = -1;
                    D[j] = C::neutral();
                } else {
                    // pop best element
                    int currRank = processIds[0];
                    int& p = pointer[currRank];
                    D[j] = heap_vals[0];
                    I[j] = I_in[stride * currRank + p] + translations[currRank];

                    faiss::heap_pop<C>(heap_size--, heap_vals, processIds);
                    p++;
                    if (p < k && I_in[stride * currRank + p] >= 0) {
                        faiss::heap_push<C>(
                                ++heap_size,
                                heap_vals,
                                processIds,
                                D_in[stride * currRank + p],
                                currRank);
                    }
                }
            }
        }
    }
}

void search(
        int processRank,
        int totalGpus,
        bool useGpu,
        int nProcesses,
        bool shardPerProcess,
        int numIndexingVecs,
        int remainingIndexingVecs,
        const faiss::Index* index,
        float* queries,
        int* groundTruth,
        size_t numQueries,
        int kBegin,
        int kEnd,
        int groundTruthK,
        int nRuns,
        bool verbose = false) {
    std::vector<int> kList = {
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    clock_t tStart, tEnd, tMid;
    double tGpu, tGpuStd, tCpu, tCpuStd, tTotal, tTotalStd;

    if (nRuns <= 0) {
        nRuns = 1;
    }

    int totalNumQueries = numQueries;
    int remainingNumQueries = totalNumQueries % nProcesses;
    if (!shardPerProcess) {
        if (nProcesses > 0) {
            numQueries = totalNumQueries / nProcesses;
            if (processRank == 0) {
                numQueries += remainingNumQueries;
            } else {
                queries += remainingNumQueries + numQueries * processRank;
            }
        }
    }

    std::stringstream queriesOut;
    queriesOut << "total # queries: " << totalNumQueries << ", ";
    queriesOut << "# queries per proccess: " << numQueries;
    processPrint(processRank, queriesOut);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = kBegin > 0 ? kBegin : 0; i < kEnd && i < kList.size(); i++) {
        int k = kList[i];
        std::stringstream kOut;
        kOut << "k: " << k;
        processPrint(processRank, kOut);
        MPI_Barrier(MPI_COMM_WORLD);

        try {
            std::vector<float> heapDistances, outDistances;
            std::vector<faiss::idx_t> heapLabels, outLabels;
            int outNum = numQueries * k;

            if (processRank == 0) {
                outDistances.resize(nProcesses * outNum);
                outLabels.resize(nProcesses * outNum);
            } else {
                outDistances.resize(outNum);
                outLabels.resize(outNum);
            }

            tGpu = 0;
            tGpuStd = 0;
            tCpu = 0;
            tCpuStd = 0;
            tTotal = 0;
            tTotalStd = 0;
            for (int j = 0; j < nRuns; j++) {
                faiss::gpu::CpuTimer timer;

                tStart = clock();

                index->search(
                        numQueries,
                        queries,
                        k,
                        outDistances.data(),
                        outLabels.data());

                if (useGpu) {
                    synchronizeDevices(0, totalGpus);
                }

                tEnd = clock();
                tGpu += (double)(tEnd - tStart) / CLOCKS_PER_SEC;
                tGpuStd += timer.elapsedMilliseconds();

                MPI_Barrier(MPI_COMM_WORLD);

                tMid = clock();

                if (nProcesses > 1) {
                    if (processRank != 0) {
                        // send distances & labels
                        const int rankToReceive = 0;
                        const int messageTag = 0;
                        MPI_Send(
                                outDistances.data(),
                                outDistances.size(),
                                MPI_FLOAT,
                                rankToReceive,
                                messageTag,
                                MPI_COMM_WORLD);
                        MPI_Send(
                                outLabels.data(),
                                outLabels.size(),
                                MPI_LONG_LONG,
                                rankToReceive,
                                messageTag,
                                MPI_COMM_WORLD);
                        if (verbose) {
                            processPrint(processRank, "Data sent");
                        }
                    } else {
                        heapDistances.resize(outNum);
                        heapLabels.resize(outNum);

                        const int messageTag = 0;
                        float* distanceAddress = outDistances.data();
                        faiss::idx_t* labelsAddress = outLabels.data();
                        for (int sendingRank = 1; sendingRank < nProcesses;
                             sendingRank++) {
                            // receive the distances + labels right next the
                            // previous ones
                            distanceAddress += outNum;
                            labelsAddress += outNum;
                            MPI_Recv(
                                    distanceAddress,
                                    outNum,
                                    MPI_FLOAT,
                                    sendingRank,
                                    messageTag,
                                    MPI_COMM_WORLD,
                                    MPI_STATUS_IGNORE);
                            MPI_Recv(
                                    labelsAddress,
                                    outNum,
                                    MPI_LONG_LONG,
                                    sendingRank,
                                    messageTag,
                                    MPI_COMM_WORLD,
                                    MPI_STATUS_IGNORE);
                            if (verbose) {
                                processPrint(processRank, "Data received");
                            }
                        }

                        if (shardPerProcess) {
                            // In case we split the index per processes, we must
                            // shift the received labels
                            std::vector<long> translations(nProcesses, 0);
                            translations[0] = 0;
                            translations[1] =
                                    remainingIndexingVecs + numIndexingVecs;
                            for (int currRank = 1; currRank + 1 < nProcesses;
                                 currRank++) {
                                translations[currRank + 1] =
                                        translations[currRank] +
                                        numIndexingVecs;
                            }
                            merge_tables<faiss::CMin<float, int>>(
                                    numQueries,
                                    k,
                                    nProcesses,
                                    heapDistances.data(),
                                    heapLabels.data(),
                                    outDistances,
                                    outLabels,
                                    translations);
                        }
                    }
                }
                tEnd = clock();

                tCpu += (double)(tEnd - tMid) / CLOCKS_PER_SEC;
                tCpuStd += timer.elapsedMilliseconds() - tGpuStd;
                tTotal = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
                tTotalStd += timer.elapsedMilliseconds();
            }

            std::stringstream nRunsOut;
            nRunsOut << "Average number of runs: " << nRuns;
            processPrint(processRank, nRunsOut);

            std::stringstream timeGpuOut;
            timeGpuOut << "IMIPQ search time on Device Only (seconds): "
                       << tGpu / nRuns << std::endl;
            timeGpuOut << "std timer (millis): " << tGpuStd / nRuns;
            processPrint(processRank, timeGpuOut);

            std::stringstream timeCpuOut;
            timeCpuOut << "IMIPQ search time on CPU (seconds): " << tCpu / nRuns
                       << std::endl;
            timeCpuOut << "std timer (millis): " << tCpuStd / nRuns;
            processPrint(processRank, timeCpuOut);

            MPI_Barrier(MPI_COMM_WORLD);

            if (processRank == 0) {
                std::stringstream timeOut;
                timeOut << "IMIPQ search time on GPU (seconds): "
                        << tTotal / nRuns << std::endl;
                timeOut << "std timer (millis): " << tTotalStd / nRuns;
                processPrint(processRank, timeOut);

                float* outDistancesData;
                faiss::idx_t* outLabelsData;
                if (nProcesses == 1) {
                    outDistancesData = outDistances.data();
                    outLabelsData = outLabels.data();
                } else {
                    // receive + merge distances & labels
                    outDistancesData = heapDistances.data();
                    outLabelsData = heapLabels.data();
                }

                if (verbose) {
                    std::stringstream resultOut;
                    resultOut << "# Result:" << std::endl;
                    for (int a = 0; a < numQueries; a++) {
                        resultOut << "  ## Query " << a << " :" << std::endl;
                        for (int b = 0; b < k; b++) {
                            int resPos = a * k + b;
                            resultOut << "    [" << b << "]"
                                      << outDistancesData[resPos] << ", "
                                      << outLabelsData[resPos] << std::endl;
                        }
                    }
                    processPrint(processRank, resultOut);
                }

                std::stringstream recallOut;
                if (groundTruth != nullptr) {
                    int n_1 = 0, n_10 = 0, n_100 = 0, n_1000, n_1024 = 0;
                    for (int a = 0; a < numQueries; a++) {
                        faiss::idx_t firstGrounTruthId =
                                groundTruth[a * groundTruthK];
                        for (int b = 0; b < k; b++) {
                            if (outLabelsData[a * k + b] == firstGrounTruthId) {
                                if (b < 1) {
                                    n_1++;
                                }
                                if (b < 10) {
                                    n_10++;
                                }
                                if (b < 100) {
                                    n_100++;
                                }
                                if (b < 1000) {
                                    n_1000++;
                                }
                                if (b < 1024) {
                                    n_1024++;
                                }
                                break;
                            }
                        }
                    }
                    recallOut << std::endl;
                    recallOut << "R@1 = " << n_1 / double(numQueries)
                              << std::endl;
                    recallOut << "R@10 = " << n_10 / double(numQueries)
                              << std::endl;
                    recallOut << "R@100 = " << n_100 / double(numQueries)
                              << std::endl;
                    recallOut << "R@1000 = " << n_1000 / double(numQueries)
                              << std::endl;
                    recallOut << "R@1024 = " << n_1024 / double(numQueries);
                } else {
                    recallOut << std::endl;
                    recallOut << "R@1 = NOT COMPUTED" << std::endl;
                    recallOut << "R@10 = NOT COMPUTED" << std::endl;
                    recallOut << "R@100 = NOT COMPUTED" << std::endl;
                    recallOut << "R@1000 = NOT COMPUTED" << std::endl;
                    recallOut << "R@1024 = NOT COMPUTED";
                }
                processPrint(processRank, recallOut);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        } catch (const std::exception& e) {
            std::stringstream eOut;
            MPI_Barrier(MPI_COMM_WORLD);
            if (useGpu) {
                synchronizeDevices(0, totalGpus);
            }
            eOut << "K EXCEPTION: " << e.what() << std::endl;
            processPrint(processRank, eOut);
            if (i == 0 || i == kBegin) {
                throw;
            }
        } catch (...) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (useGpu) {
                synchronizeDevices(0, totalGpus);
            }
            processPrint(processRank, "K UNKNOWN EXCEPTION");
            if (i == 0 || i == kBegin) {
                throw;
            }
        }
    }
}

struct RandomContext {
    int64_t seed = 0;
};

float* fvecs_rand(size_t num, size_t dim, RandomContext& randomContext) {
    size_t size = num * dim;
    float* vecs = new float[size];
    faiss::float_rand(vecs, size, randomContext.seed);
    ++randomContext.seed;
    return vecs;
}

float* vecs_load(
        bool isVecFloat,
        std::string fileName,
        size_t num,
        int d,
        RandomContext& randomContext,
        size_t numOffset = 0) {
    int dRead;
    if (fileName.empty()) {
        return fvecs_rand(num, d, randomContext);
    }
    float* vecs;
    if (isVecFloat) {
        vecs = faiss::fvecs_read(fileName.c_str(), num, numOffset, &dRead);
    } else {
        vecs = faiss::bvecs_read(fileName.c_str(), num, numOffset, &dRead);
    }
    assert(d == dRead);
    return vecs;
}

template <class TVec>
TVec* vecs_replicate(TVec* vecs, size_t size, int numReplicas) {
    TVec* replacaVecs = new TVec[size * numReplicas];
    for (size_t i = 0; i < numReplicas; i++) {
        for (size_t j = 0; j < size; j++) {
            replacaVecs[j + i * size] = vecs[j];
        }
    }
    return replacaVecs;
}

size_t roundMemAllocUp(size_t size) {
    return faiss::gpu::utils::roundUp(size, (size_t)256);
}

size_t roundMemAllocDown(size_t size) {
    return size / 256 * 256;
}

size_t calcFixedMemSize(
        std::unordered_map<faiss::gpu::AllocType, size_t> allocSizePerTypeMap) {
    size_t fixedMemSize = 0;
    for (auto&& allocSizePerType : allocSizePerTypeMap) {
        size_t allocSize = roundMemAllocUp(allocSizePerType.second);
        fixedMemSize += roundMemAllocUp(allocSize);
    }
    return fixedMemSize;
}

template <class IndexT>
size_t calcStructureMemSize(
        size_t d,
        size_t coarseCodebookSize,
        size_t numSubQuantizers,
        size_t nbitsSubQuantizer);

template <>
size_t calcStructureMemSize<faiss::gpu::GpuIndexIMIPQ>(
        size_t d,
        size_t coarseCodebookSize,
        size_t numSubQuantizers,
        size_t nbitsSubQuantizer) {
    size_t subCodebookSize = 1 << nbitsSubQuantizer;
    size_t coarseQuantizerMemSize =
            roundMemAllocUp(d * coarseCodebookSize * sizeof(float));
    size_t normMemSize =
            roundMemAllocUp(2 * coarseCodebookSize * sizeof(float));
    size_t productQuantizerMemSize =
            2 * roundMemAllocUp(d * subCodebookSize * sizeof(float));
    size_t precomputedMemSize = roundMemAllocUp(
            coarseCodebookSize * subCodebookSize * numSubQuantizers *
            sizeof(float));
    size_t listOffsetMemSize = roundMemAllocUp(
            coarseCodebookSize * coarseCodebookSize * sizeof(unsigned int));
    return subCodebookSize + coarseQuantizerMemSize + normMemSize +
            productQuantizerMemSize + precomputedMemSize + listOffsetMemSize;
}

template <>
size_t calcStructureMemSize<faiss::gpu::GpuIndexIVFPQ>(
        size_t d,
        size_t coarseCodebookSize,
        size_t numSubQuantizers,
        size_t nbitsSubQuantizer) {
    size_t subCodebookSize = 1 << nbitsSubQuantizer;
    size_t coarseQuantizerMemSize =
            roundMemAllocUp(d * coarseCodebookSize * sizeof(float));
    size_t normMemSize = roundMemAllocUp(coarseCodebookSize * sizeof(float));
    size_t productQuantizerMemSize =
            2 * roundMemAllocUp(d * subCodebookSize * sizeof(float));
    size_t precomputedMemSize = roundMemAllocUp(
            coarseCodebookSize * subCodebookSize * numSubQuantizers *
            sizeof(float));
    size_t codesPointersMemSize =
            roundMemAllocUp(coarseCodebookSize * sizeof(void*));
    size_t idsPointersMemSize =
            roundMemAllocUp(coarseCodebookSize * sizeof(void*));
    size_t listsLengthsMemSize =
            roundMemAllocUp(coarseCodebookSize * sizeof(int));
    return subCodebookSize + coarseQuantizerMemSize + normMemSize +
            productQuantizerMemSize + precomputedMemSize +
            codesPointersMemSize + idsPointersMemSize + listsLengthsMemSize;
}

void initResourcesMultiGpu(
        int processRank,
        int deviceIdInit,
        int ngpus,
        std::unordered_map<faiss::gpu::AllocType, size_t>&
                allocSizePerTypeMapPerGpu,
        size_t tempMemory,
        std::vector<faiss::gpu::GpuResourcesProvider*>& resVector,
        std::vector<int>& devs,
        bool allocLogging,
        int pinnedMemoryMode) {
    std::stringstream out;
    out << "Device List: ";
    int currDevice = deviceIdInit;
    for (int i = 0; i < ngpus; i++) {
        faiss::gpu::StandardGpuResources* res;
        res = new faiss::gpu::StandardGpuResources(allocSizePerTypeMapPerGpu);
        res->setLogMemoryAllocations(allocLogging);
        res->setTempMemory(tempMemory);
        if (pinnedMemoryMode == 0) {
            res->setPinnedMemory(0);
        }
        resVector.push_back(res);
        devs.push_back(currDevice);
        out << currDevice << ",";
        currDevice++;
    }
    processPrint(processRank, out);
}

void printDeviceMemory(
        size_t devFree,
        size_t devTotal,
        int deviceId,
        int processRank) {
    std::stringstream out;
    out << std::endl;
    out << "-------Memory-------" << std::endl;
    out << "Device: " << deviceId << std::endl;
    out << "Free: " << devFree << std::endl;
    out << "Total: " << devTotal << std::endl;
    processPrint(processRank, out);
}

void printDeviceMemory(int deviceId = 0, int processRank = 0) {
    size_t devFree = 0;
    size_t devTotal = 0;
    faiss::gpu::DeviceScope scope(deviceId);
    CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
    printDeviceMemory(devFree, devTotal, deviceId, processRank);
}

void printAllDevicesMemory(
        bool print,
        int deviceIdInit = 0,
        int ngpus = 1,
        int processRank = 0) {
    if (!print) {
        return;
    }
    std::stringstream out;
    for (int deviceId = deviceIdInit; deviceId < ngpus; deviceId++) {
        printDeviceMemory(deviceId, processRank);
    }
}

void getAvailableMemoryPerDevice(
        size_t& devFree,
        size_t& devTotal,
        int deviceIdInit = 0,
        int ngpus = 1,
        int nProcessesPerGpu = 1) {
    size_t currDevFree = 0;
    size_t currDevTotal = 0;
    bool devFreeIsSet = false;

    devFree = 0;
    devTotal = 0;
    int deviceId = deviceIdInit;
    for (int i = 0; i < ngpus; i++) {
        faiss::gpu::DeviceScope scope(i);
        CUDA_VERIFY(cudaMemGetInfo(&currDevFree, &currDevTotal));
        if (!devFreeIsSet) {
            devFreeIsSet = true;
            devFree = currDevFree;
            devTotal = currDevTotal;
        } else {
            devFree = std::min(devFree, currDevFree);
            devTotal = std::min(devTotal, currDevTotal);
        }
        deviceId++;
    }
    devFree /= nProcessesPerGpu;
    devTotal /= nProcessesPerGpu;
}

template <class IndexT>
IndexT* loadIndexToCpu(int processRank, std::string fileName) {
    IndexT* indexCpu = nullptr;
    if (!fileName.empty()) {
        FILE* f = fopen(fileName.c_str(), "rb");
        if (f) {
            clock_t tStart, tEnd;
            double tGpu;
            fclose(f);
            tStart = clock();
            indexCpu =
                    dynamic_cast<IndexT*>(faiss::read_index(fileName.c_str()));
            tEnd = clock();
            tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
            std::stringstream out;
            out << "Time to load index from memory: " << tGpu;
            processPrint(processRank, out);
        }
    }
    return indexCpu;
}

template <class IndexT>
int buildCoarseQuantizer(
        int processRank,
        IndexT* imipqGpu,
        std::string fileNameCoarseQuantizer,
        bool isVecFloat,
        std::string fileNameTraining,
        int numTrainingVecs,
        int d,
        RandomContext& randomContext,
        size_t readOffset = 0) {
    std::unique_ptr<faiss::Index> indexCpuTrainedOnly(
            loadIndexToCpu<faiss::IndexIVFPQ>(
                    processRank, fileNameCoarseQuantizer));
    if (indexCpuTrainedOnly) {
        imipqGpu->copyFrom(
                dynamic_cast<faiss::IndexIVFPQ*>(indexCpuTrainedOnly.get()));
        return 0;
    }
    // train
    clock_t tStart, tEnd;
    double tGpu;
    std::unique_ptr<float> trainingVecs(vecs_load(
            isVecFloat,
            fileNameTraining,
            numTrainingVecs,
            d,
            randomContext,
            readOffset));
    tStart = clock();
    imipqGpu->train(numTrainingVecs, trainingVecs.get());
    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::stringstream outTrain;
    outTrain << "IMIPQ train time on GPU: " << tGpu;
    processPrint(processRank, outTrain);

    // save coase quantizer
    if (processRank == 0 && !fileNameCoarseQuantizer.empty()) {
        tStart = clock();
        indexCpuTrainedOnly.reset(faiss::gpu::index_gpu_to_cpu(imipqGpu));
        faiss::gpu::CudaEvent cloneEnd(
                imipqGpu->getResources()->getDefaultStreamCurrentDevice());
        cloneEnd.cpuWaitOnEvent();
        faiss::write_index(
                indexCpuTrainedOnly.get(), fileNameCoarseQuantizer.c_str());
        tEnd = clock();
        tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
        std::stringstream outSave;
        outSave << "IMIPQ writting coarse quantizer: " << tGpu;
        processPrint(processRank, outSave);
    }
    return 1;
}

template <class IndexT>
void reserveIndexingSpace(
        int processRank,
        IndexT* imipqGpu,
        bool isVecFloat,
        std::string fileNameIndexing,
        int numIndexingVecs,
        int d,
        size_t numVecsTile,
        RandomContext& randomContext,
        size_t readOffset = 0) {
    clock_t tStart, tEnd;
    double tGpu;
    tStart = clock();
    for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
        size_t currentNumVecsTile = std::min(numVecsTile, numIndexingVecs - i);

        std::unique_ptr<float> indexingVecs(vecs_load(
                isVecFloat,
                fileNameIndexing,
                currentNumVecsTile,
                d,
                randomContext,
                readOffset + i));

        imipqGpu->updateExpectedNumAddsPerList(
                currentNumVecsTile, indexingVecs.get());
        faiss::gpu::CudaEvent updateEnd(
                imipqGpu->getResources()->getDefaultStreamCurrentDevice());
        updateEnd.cpuWaitOnEvent();
    }

    imipqGpu->applyExpectedNumAddsPerList();
    faiss::gpu::CudaEvent applyEnd(
            imipqGpu->getResources()->getDefaultStreamCurrentDevice());
    applyEnd.cpuWaitOnEvent();

    imipqGpu->resetExpectedNumAddsPerList();

    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::stringstream out;
    out << "IMIPQ reserve time on GPU: " << tGpu;
    processPrint(processRank, out);
}

template <class IndexT>
void addToIndex(
        int processRank,
        IndexT* imipqGpu,
        bool isVecFloat,
        std::string fileNameIndexing,
        int numIndexingVecs,
        int d,
        size_t numVecsTile,
        RandomContext& randomContext,
        size_t readOffset = 0) {
    clock_t tStart, tEnd;
    double tGpu;
    tStart = clock();
    for (size_t i = 0; i < numIndexingVecs; i += numVecsTile) {
        size_t currentNumVecsTile = std::min(numVecsTile, numIndexingVecs - i);
        std::unique_ptr<float> indexingVecs(vecs_load(
                isVecFloat,
                fileNameIndexing,
                currentNumVecsTile,
                d,
                randomContext,
                i));
        imipqGpu->add(currentNumVecsTile, indexingVecs.get());
    }
    tEnd = clock();
    tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
    std::stringstream out;
    out << "IMIPQ add time on GPU: " << tGpu;
    processPrint(processRank, out);
}

template <class ConfigT>
ConfigT getConfig(
        faiss::gpu::IndicesOptions indiceOptions,
        int deviceIdInit,
        int pinnedMemoryMode,
        int usePrecomputed);

template <>
faiss::gpu::GpuIndexIMIPQConfig getConfig<faiss::gpu::GpuIndexIMIPQConfig>(
        faiss::gpu::IndicesOptions indiceOptions,
        int deviceIdInit,
        int pinnedMemoryMode,
        int usePrecomputeds) {
    faiss::gpu::GpuIndexIMIPQConfig config;
    config.memorySpace = faiss::gpu::MemorySpace::Fixed;
    // config.multiIndexConfig.memorySpace = faiss::gpu::MemorySpace::Fixed;
    config.indicesOptions = indiceOptions;
    config.usePrecomputedTables = true;
    config.device = deviceIdInit;

    if (pinnedMemoryMode == 2) {
        config.forcePinnedMemory = true;
    }
    return config;
}

template <>
faiss::gpu::GpuIndexIVFPQConfig getConfig<faiss::gpu::GpuIndexIVFPQConfig>(
        faiss::gpu::IndicesOptions indiceOptions,
        int deviceIdInit,
        int pinnedMemoryMode,
        int usePrecomputed) {
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.memorySpace = faiss::gpu::MemorySpace::Fixed;
    // config.multiIndexConfig.memorySpace = faiss::gpu::MemorySpace::Fixed;
    config.indicesOptions = indiceOptions;
    config.usePrecomputedTables = usePrecomputed;
    config.device = deviceIdInit;
    return config;
}

template <class ConfigT, class IndexT>
IndexT* getIndex(
        faiss::gpu::StandardGpuResources& res,
        int d,
        int coarseCodebookSize,
        int numSubQuantizers,
        int nbitsSubQuantizer,
        ConfigT& config);

template <>
faiss::gpu::GpuIndexIMIPQ* getIndex(
        faiss::gpu::StandardGpuResources& res,
        int d,
        int coarseCodebookSize,
        int numSubQuantizers,
        int nbitsSubQuantizer,
        faiss::gpu::GpuIndexIMIPQConfig& config) {
    return new faiss::gpu::GpuIndexIMIPQ(
            &res,
            d,
            coarseCodebookSize,
            numSubQuantizers,
            nbitsSubQuantizer,
            config);
}

template <>
faiss::gpu::GpuIndexIVFPQ* getIndex(
        faiss::gpu::StandardGpuResources& res,
        int d,
        int coarseCodebookSize,
        int numSubQuantizers,
        int nbitsSubQuantizer,
        faiss::gpu::GpuIndexIVFPQConfig& config) {
    return new faiss::gpu::GpuIndexIVFPQ(
            &res,
            d,
            coarseCodebookSize,
            numSubQuantizers,
            nbitsSubQuantizer,
            faiss::METRIC_L2,
            config);
}

template <class ConfigT>
faiss::gpu::GpuMultipleClonerOptions getMultiGpuConfig(
        ConfigT& res,
        int useShards,
        int verbose);

template <>
faiss::gpu::GpuMultipleClonerOptions getMultiGpuConfig(
        faiss::gpu::GpuIndexIMIPQConfig& config,
        int useShards,
        int verbose) {
    faiss::gpu::GpuMultipleClonerOptions options;
    options.memorySpace = config.memorySpace;
    options.indicesOptions = config.indicesOptions;
    options.usePrecomputed = config.usePrecomputedTables;
    options.precomputeCodesOnCpu = config.precomputeCodesOnCpu;
    options.forcePinnedMemory = config.forcePinnedMemory;
    options.shard = useShards;
    options.shard_type = 1;
    options.verbose = verbose;
    return options;
}

template <>
faiss::gpu::GpuMultipleClonerOptions getMultiGpuConfig(
        faiss::gpu::GpuIndexIVFPQConfig& config,
        int useShards,
        int verbose) {
    faiss::gpu::GpuMultipleClonerOptions options;
    options.memorySpace = config.memorySpace;
    options.indicesOptions = config.indicesOptions;
    options.usePrecomputed = config.usePrecomputedTables;
    options.precomputeCodesOnCpu = config.precomputeCodesOnCpu;
    options.shard = useShards;
    options.shard_type = 1;
    options.verbose = verbose;
    return options;
}

template <class ConfigT, class IndexT>
void demo(
        bool isVecFloat,
        int d,
        int coarseCodebookSize,
        int numSubQuantizers,
        int nbitsSubQuantizer,
        std::string fileNameTraining,
        size_t numTrainingVecs,
        std::string fileNameIndexing,
        size_t numIndexingVecs,
        std::string fileNameQueries,
        size_t queriesOffset,
        std::string fileNameGroundTruth,
        int numQueriesBegin,
        int numQueriesEnd,
        int nprobeBegin,
        int nprobeEnd,
        int kBegin,
        int kEnd,
        int ngpus,
        bool useShards,
        int nProcesses,
        int processRank,
        bool sharedGpuProcess,
        bool shardPerProcess,
        size_t safeMemMargin,
        std::string fileNameCoarseQuantizer,
        std::string fileNameIndex,
        bool profile,
        bool allocLogging,
        bool verbose,
        int nRuns,
        int pinnedMemoryMode,
        int usePrecomputed,
        int useMultiIndex,
        int useGpu,
        bool printGpuMemory,
        int numQueryReplicas,
        bool copyPerShard) {
    if (useGpu) {
        CUDA_VERIFY(cudaProfilerStop());
    }

    RandomContext randomContext;

    int numDevices = faiss::gpu::getNumDevices();

    int totalGpus = ngpus;
    int totalNumIndexingVecs = numIndexingVecs;
    int remainingIndexingVecs = totalNumIndexingVecs % nProcesses;
    int deviceIdInit = 0;
    int nProcessesPerGpu = 1;

    if (sharedGpuProcess) {
        nProcessesPerGpu = nProcesses;
    } else {
        ngpus /= nProcesses;
        deviceIdInit = processRank * ngpus;
    }

    if (shardPerProcess && !copyPerShard) {
        numIndexingVecs /= nProcesses;
        // the first process manages the remaining number of vecs
        if (processRank == 0) {
            numIndexingVecs += remainingIndexingVecs;
        }
    }

    size_t devFree = 0;
    size_t devTotal = 0;

    if (processRank == 0) {
        std::stringstream deviceStatusStr;
        deviceStatusStr << "# Available Devices: " << numDevices << std::endl;
        deviceStatusStr << "# Devices: " << totalGpus << std::endl;
        deviceStatusStr << "# Processes: " << nProcesses << std::endl;
        deviceStatusStr << "# Devices per process: " << ngpus << std::endl;
        deviceStatusStr << "# Processes per Gpu " << nProcessesPerGpu
                        << std::endl;
        processPrint(processRank, deviceStatusStr);
        printAllDevicesMemory(printGpuMemory, 0, ngpus);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!sharedGpuProcess) {
        assert(totalGpus % nProcesses == 0);
    }

    assert(ngpus <= numDevices);

    std::stringstream outDev;
    outDev << "(first device ID) " << deviceIdInit;
    processPrint(processRank, outDev);

    getAvailableMemoryPerDevice(
            devFree, devTotal, deviceIdInit, ngpus, nProcessesPerGpu);

    // Let's ensure the memory isn't changed
    MPI_Barrier(MPI_COMM_WORLD);

    size_t numIndexingVecsPerGpu;
    if (useShards) {
        numIndexingVecsPerGpu = numIndexingVecs / ngpus;
    } else {
        numIndexingVecsPerGpu = numIndexingVecs;
    }

    faiss::gpu::IndicesOptions indiceOptions = faiss::gpu::INDICES_32_BIT;

    size_t devFreeLimit = std::min(devFree, safeMemMargin);

    size_t imiStructureMemSize = calcStructureMemSize<IndexT>(
            d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer);

    auto allocSizePerTypeMap = IndexT::getInvListsAllocSizePerTypeInfo(
            numIndexingVecs,
            numSubQuantizers,
            nbitsSubQuantizer,
            false,
            indiceOptions);
    auto allocSizePerTypeMapPerGpu = IndexT::getInvListsAllocSizePerTypeInfo(
            numIndexingVecsPerGpu,
            numSubQuantizers,
            nbitsSubQuantizer,
            false,
            indiceOptions);

    size_t fixedMemSize = calcFixedMemSize(allocSizePerTypeMap);
    size_t fixedMemSizePerGpu = calcFixedMemSize(allocSizePerTypeMapPerGpu);

    devFreeLimit = std::max(devFreeLimit, fixedMemSize + imiStructureMemSize);
    devFreeLimit =
            std::max(devFreeLimit, fixedMemSizePerGpu + imiStructureMemSize);

    std::stringstream memoryInfoStr1;
    memoryInfoStr1 << std::endl;
    memoryInfoStr1 << "fixedMemSize: " << fixedMemSize << std::endl;
    memoryInfoStr1 << "fixedMemSizePerGpu: " << fixedMemSizePerGpu << std::endl;
    memoryInfoStr1 << "imiStructureMemSize: " << imiStructureMemSize
                   << std::endl;
    memoryInfoStr1 << "devFree: " << devFree << std::endl;
    memoryInfoStr1 << "safeMemMargin: " << safeMemMargin << std::endl;
    memoryInfoStr1 << "devFreeLimit: " << devFreeLimit << std::endl;
    processPrint(processRank, memoryInfoStr1);

    assert(devFreeLimit >= fixedMemSize + imiStructureMemSize);
    assert(devFreeLimit >= fixedMemSizePerGpu + imiStructureMemSize);

    MPI_Barrier(MPI_COMM_WORLD);

    size_t tempMemory = roundMemAllocDown(
            devFreeLimit - fixedMemSize - imiStructureMemSize);
    size_t tempMemoryPerGpu = roundMemAllocDown(
            devFreeLimit - fixedMemSizePerGpu - imiStructureMemSize);

    std::stringstream memoryInfoStr2;
    memoryInfoStr2 << "tempMemory: " << tempMemory << std::endl;
    memoryInfoStr2 << "tempMemoryPerGpu: " << tempMemoryPerGpu << std::endl;
    ;

    processPrint(processRank, memoryInfoStr2);

    // set maximum available memory for tiling over vectors while adding them to
    // the GPU
    size_t maxAddTileSize = (size_t)4 * 1024 * 1024 * 1024;
    size_t numVecsTile = maxAddTileSize / (d * sizeof(float));
    numVecsTile = std::min(numVecsTile, numIndexingVecs);
    numVecsTile = std::min(numVecsTile, (size_t)10000);
    numVecsTile = std::max(numVecsTile, (size_t)1);

    ConfigT config = getConfig<ConfigT>(
            indiceOptions, deviceIdInit, pinnedMemoryMode, usePrecomputed);

    std::cout << "config.device: " << config.device << std::endl;
    std::cout << "config.usePrecomputedTables: " << config.usePrecomputedTables
              << std::endl;

    std::vector<faiss::gpu::GpuResourcesProvider*> resVector;
    std::vector<int> devs;
    std::unique_ptr<faiss::Index> finalIndex;
    std::unique_ptr<faiss::Index> indexCpu;

    { // Build Index
        bool fileNameIndexIsEmpty = fileNameIndex.empty();

        int storedRank = processRank;
        if (!shardPerProcess) {
            storedRank = 0;
        }

        if (nProcesses > 1) {
            std::stringstream finaNameIndexPostfix;
            finaNameIndexPostfix << "_rank" << storedRank << "_numVecs"
                                 << numIndexingVecs;
            fileNameIndex.append(finaNameIndexPostfix.str());
        }

        std::stringstream loadIndexStart;
        loadIndexStart << "Index - loading: " << fileNameIndex;
        processPrint(processRank, loadIndexStart);
        indexCpu.reset(
                loadIndexToCpu<faiss::IndexIVFPQ>(processRank, fileNameIndex));
        if (!indexCpu) {
            { // indexing
                faiss::gpu::StandardGpuResources res(allocSizePerTypeMap);
                res.setLogMemoryAllocations(allocLogging);
                res.setTempMemory(tempMemory);
                std::unique_ptr<IndexT> imipqGpu(getIndex<ConfigT, IndexT>(
                        res,
                        d,
                        coarseCodebookSize,
                        numSubQuantizers,
                        nbitsSubQuantizer,
                        config));
                imipqGpu->verbose = verbose;

                size_t indexToAddOffset = 0;

                if (processRank == 0) {
                    std::stringstream outTrainStart, outEnd;
                    outTrainStart << "Coarse quantizer - loading: "
                                  << fileNameCoarseQuantizer;
                    outTrainStart << " (seed == " << randomContext.seed << ")";
                    processPrint(processRank, outTrainStart);
                    // train or load the coarse quantizer
                    buildCoarseQuantizer(
                            processRank,
                            imipqGpu.get(),
                            fileNameCoarseQuantizer,
                            isVecFloat,
                            fileNameTraining,
                            numTrainingVecs,
                            d,
                            randomContext);
                    outEnd << "Coarse quantizer - loaded: "
                           << fileNameCoarseQuantizer;
                    processPrint(processRank, outEnd);
                    // Just ensure the seed is correct in case the coarse
                    // quantizer was loaded
                    randomContext.seed = numTrainingVecs;
                }

                MPI_Barrier(MPI_COMM_WORLD);

                if (processRank != 0) {
                    // train or load the coarse quantizer
                    std::stringstream readStart, readEnd;
                    readStart << "Coarse quantizer - reading: "
                              << fileNameCoarseQuantizer;
                    processPrint(processRank, readStart);

                    std::unique_ptr<faiss::IndexIVFPQ> indexCpuTrainedOnly(
                            loadIndexToCpu<faiss::IndexIVFPQ>(
                                    processRank, fileNameCoarseQuantizer));
                    imipqGpu->copyFrom(indexCpuTrainedOnly.get());
                    readEnd << "Coarse quantizer - loaded";
                    processPrint(processRank, readEnd);

                    // Let's ensure every vector if different for everyProccess
                    // process
                    if (shardPerProcess) {
                        indexToAddOffset = remainingIndexingVecs +
                                numIndexingVecs * processRank;
                    } else {
                        if (copyPerShard) {
                            indexToAddOffset = numIndexingVecs * processRank;
                        } else {
                            indexToAddOffset = 0;
                        }
                    }
                    randomContext.seed = numTrainingVecs + indexToAddOffset;
                }

                if (processRank == 0) {
                    printAllDevicesMemory(printGpuMemory, 0, ngpus);
                }

                MPI_Barrier(MPI_COMM_WORLD);

                std::stringstream addToIndexStart;
                addToIndexStart << "Adding " << numIndexingVecs
                                << " vectors from " << totalNumIndexingVecs
                                << " to index with " << indexToAddOffset
                                << " offset";
                addToIndexStart << " (seed == " << randomContext.seed << ")";
                processPrint(processRank, addToIndexStart);

                int64_t initSeed = 0;
                int64_t endSeed = 0;

                // save current initial seed for using it again while adding the
                // vectors
                initSeed = randomContext.seed;

                reserveIndexingSpace(
                        processRank,
                        imipqGpu.get(),
                        isVecFloat,
                        fileNameIndexing,
                        numIndexingVecs,
                        d,
                        numVecsTile,
                        randomContext,
                        indexToAddOffset);

                std::stringstream reserved;
                reserved << "space reserved" << std::endl;

                processPrint(processRank, reserved);

                // save it for assertion
                endSeed = randomContext.seed;

                if (processRank == 0) {
                    printAllDevicesMemory(printGpuMemory, 0, ngpus);
                }

                MPI_Barrier(MPI_COMM_WORLD);

                randomContext.seed = initSeed;

                addToIndex(
                        processRank,
                        imipqGpu.get(),
                        isVecFloat,
                        fileNameIndexing,
                        numIndexingVecs,
                        d,
                        numVecsTile,
                        randomContext,
                        indexToAddOffset);

                assert(randomContext.seed == endSeed);

                if (processRank == 0) {
                    printAllDevicesMemory(printGpuMemory, 0, ngpus);
                }

                MPI_Barrier(MPI_COMM_WORLD);

                indexCpu.reset(faiss::gpu::index_gpu_to_cpu(imipqGpu.get()));
                faiss::gpu::CudaEvent cloneEnd(
                        imipqGpu->getResources()
                                ->getDefaultStreamCurrentDevice());
                cloneEnd.cpuWaitOnEvent();
            }

            if (!fileNameIndexIsEmpty) {
                if (shardPerProcess || processRank == 0) {
                    std::stringstream writeIndexStart;
                    writeIndexStart << "writing: " << fileNameIndex << "...";
                    processPrint(processRank, writeIndexStart);
                    faiss::write_index(indexCpu.get(), fileNameIndex.c_str());
                    processPrint(processRank, "done");
                }
            }
        }

        std::stringstream loadIndexEnd;
        loadIndexEnd << "Index - built: " << fileNameIndex;
        processPrint(processRank, loadIndexEnd);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (profile) {
        if (useGpu) {
            clock_t tStart, tEnd;
            double tGpu;

            faiss::gpu::GpuMultipleClonerOptions options =
                    getMultiGpuConfig<ConfigT>(config, useShards, verbose);

            processPrint(processRank, "Ininting resource for multiple GPUs");
            initResourcesMultiGpu(
                    processRank,
                    deviceIdInit,
                    ngpus,
                    allocSizePerTypeMapPerGpu,
                    tempMemoryPerGpu,
                    resVector,
                    devs,
                    allocLogging,
                    pinnedMemoryMode);

            processPrint(
                    processRank, "Moving index from cpu to multiple GPUs: ");
            try {
                tStart = clock();
                finalIndex.reset(faiss::gpu::index_cpu_to_gpu_multiple(
                        resVector, devs, indexCpu.get(), &options));
                if (useGpu) {
                    synchronizeDevices(0, totalGpus);
                }
                tEnd = clock();
                tGpu = (double)(tEnd - tStart) / CLOCKS_PER_SEC;
                std::stringstream indexMovedOut;
                indexMovedOut << "Index moved in " << tGpu;
                processPrint(processRank, indexMovedOut);
                indexCpu.release();
            } catch (const std::exception& e) {
                std::stringstream eOut;
                eOut << "Multi-GPU Clone exception: " << e.what() << std::endl;
                processPrint(processRank, eOut);
                throw;
            }
        } else {
            faiss::IndexIVFPQ* ivfpqCpu =
                    dynamic_cast<faiss::IndexIVFPQ*>(indexCpu.get());
            if (usePrecomputed) {
                if (useMultiIndex) {
                    ivfpqCpu->use_precomputed_table = 2;
                } else {
                    ivfpqCpu->use_precomputed_table = 1;
                }
            }
            finalIndex = std::move(indexCpu);
        }

        if (processRank == 0) {
            printAllDevicesMemory(printGpuMemory, 0, ngpus);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> numQueriesList = {
                1,
                1000,
                8192,
                10000,
                16384,
                32768,
                65536,
                100000,
                131072,
                262144,
                524288,
                1048576,
                2097152,
                4194304,
                8388608,
                16777216};
        numQueriesEnd = std::min(numQueriesEnd, (int)numQueriesList.size());
        std::vector<int> nprobeList = {1,    2,     4,     8,    16,   32,
                                       64,   128,   256,   512,  1024, 2048,
                                       2194, 2352,  2521,  2702, 2896, 4096,
                                       8192, 16384, 32768, 65536};

        std::unique_ptr<float> queries;
        std::unique_ptr<int> groundTruth;
        int dRead;

        size_t numQueriesSearch = (size_t)numQueriesList[numQueriesEnd - 1];

        queries.reset(vecs_load(
                isVecFloat,
                fileNameQueries,
                numQueriesSearch,
                d,
                randomContext,
                queriesOffset));
        if (numQueryReplicas > 1) {
            queries.reset(vecs_replicate<float>(
                    queries.get(), numQueriesSearch * d, numQueryReplicas));
        }

        if (!fileNameGroundTruth.empty()) {
            groundTruth.reset(faiss::ivecs_read(
                    fileNameGroundTruth.c_str(), numQueriesSearch, 0, &dRead));
            if (numQueryReplicas > 1) {
                groundTruth.reset(vecs_replicate<int>(
                        groundTruth.get(),
                        numQueriesSearch * dRead,
                        numQueryReplicas));
            }
        }

        if (useGpu) {
            CUDA_VERIFY(cudaProfilerStart());
            synchronizeDevices(0, totalGpus);
        }

        for (int i = numQueriesBegin > 0 ? numQueriesBegin : 0;
             i < numQueriesEnd;
             i++) {
            int numQueries = numQueriesList[i];

            std::stringstream numQueriesOut;
            numQueriesOut << "numOfQueries: " << numQueries
                          << " ===============";
            processPrint(processRank, numQueriesOut);

            try {
                for (int j = nprobeBegin > 0 ? nprobeBegin : 0;
                     j < nprobeEnd && j < nprobeList.size();
                     j++) {
                    int nprobe = nprobeList[j];
                    std::stringstream numProbeOut;
                    numProbeOut << "nprobe: " << nprobe << "---------";
                    processPrint(processRank, numProbeOut);

                    try {
                        if (useGpu) {
                            faiss::ThreadedIndex<faiss::Index>* threadedIndex =
                                    dynamic_cast<faiss::ThreadedIndex<
                                            faiss::Index>*>(finalIndex.get());

                            if (threadedIndex) {
                                // multi GPU
                                for (int k = 0; k < threadedIndex->count();
                                     k++) {
                                    IndexT* imipqGpu = dynamic_cast<IndexT*>(
                                            threadedIndex->at(k));
                                    imipqGpu->nprobe = nprobe;
                                    imipqGpu->verbose = verbose;
                                    std::stringstream searchInfoOut;
                                    searchInfoOut
                                            << "Gpu: " << imipqGpu->getDevice()
                                            << ", maxListLength: "
                                            << imipqGpu->getMaxListLength()
                                            << ", nlist: " << imipqGpu->nlist
                                            << ", ntotal: " << imipqGpu->ntotal;
                                    processPrint(processRank, searchInfoOut);
                                }
                            } else {
                                // single GPU
                                IndexT* imipqGpu =
                                        dynamic_cast<IndexT*>(finalIndex.get());
                                std::cout << "typeid: "
                                          << typeid(finalIndex.get()).name()
                                          << std::endl;
                                assert(imipqGpu);
                                imipqGpu->nprobe = nprobe;
                                imipqGpu->verbose = verbose;
                                std::stringstream searchInfoOut;
                                searchInfoOut
                                        << "Gpu: " << imipqGpu->getDevice()
                                        << ", maxListLength: "
                                        << imipqGpu->getMaxListLength()
                                        << ", nlist: " << imipqGpu->nlist
                                        << ", ntotal: " << imipqGpu->ntotal;
                                processPrint(processRank, searchInfoOut);
                            }
                        } else {
                            faiss::IndexIVFPQ* ivfpqCpu =
                                    dynamic_cast<faiss::IndexIVFPQ*>(
                                            finalIndex.get());
                            ivfpqCpu->nprobe = nprobe;
                            finalIndex->verbose = verbose;
                            std::stringstream searchInfoOut;
                            searchInfoOut << "CPU "
                                          << ", nlist: " << ivfpqCpu->nlist
                                          << ", ntotal: " << ivfpqCpu->ntotal;
                            processPrint(processRank, searchInfoOut);
                        }

                        search(processRank,
                               totalGpus,
                               useGpu,
                               nProcesses,
                               shardPerProcess,
                               numIndexingVecs,
                               remainingIndexingVecs,
                               finalIndex.get(),
                               queries.get(),
                               groundTruth.get(),
                               numQueries,
                               kBegin,
                               kEnd,
                               dRead,
                               nRuns);

                    } catch (...) {
                        MPI_Barrier(MPI_COMM_WORLD);
                        processPrint(processRank, "NPROBE UNKNOWN EXCEPTION");
                        if (j == 0 || j == nprobeBegin) {
                            throw;
                        }
                    }
                }
            } catch (...) {
                MPI_Barrier(MPI_COMM_WORLD);
                processPrint(processRank, "QUERY UNKNOWN EXCEPTION");
            }
        }

        if (useGpu) {
            CUDA_VERIFY(cudaProfilerStop());
        }

        for (int i = 0; i < resVector.size(); i++) {
            delete resVector[i];
        }
    }
}

int main(int argc, char** argv) {
    if (argc <= 18) {
        std::cout << "There must be 18 or more parameters" << std::endl;
        return 1;
    }

    int d, coarseCodebookSize, numSubQuantizers, nbitsSubQuantizer,
            queriesOffset, numQueriesBegin, numQueriesEnd, kBegin, kEnd,
            nprobeBegin, nprobeEnd, isFloat, numThreads, ngpus, useShards,
            sharedGpuProcess, shardPerProcess, profile, allocLogging, verbose,
            nRuns, pinnedMemoryMode, usePrecomputed, useMultiIndex, useGpu,
            printGpuMemory, numQueryReplicas, copyPerShard;
    size_t numTrainingVecs, numIndexingVecs;
    std::string fileNameTraining, fileNameIndexing, fileNameQueries,
            fileNameGroundTruth, fileNameCoarseQuantizer, fileNameIndex;
    size_t safeMemMargin;

    std::cout << "argv[1]: " << argv[1] << std::endl;
    d = std::stoi(argv[1]);

    std::cout << "argv[2]: " << argv[2] << std::endl;
    coarseCodebookSize = std::stoi(argv[2]);

    std::cout << "argv[3]: " << argv[3] << std::endl;
    numSubQuantizers = std::stoi(argv[3]);

    std::cout << "argv[4]: " << argv[4] << std::endl;
    nbitsSubQuantizer = std::stoi(argv[4]);

    std::cout << "argv[5]: " << argv[5] << std::endl;
    fileNameTraining = argv[5];

    std::cout << "argv[6]: " << argv[6] << std::endl;
    numTrainingVecs = std::stoul(argv[6]);

    std::cout << "argv[7]: " << argv[7] << std::endl;
    fileNameIndexing = argv[7];

    std::cout << "argv[8]: " << argv[8] << std::endl;
    numIndexingVecs = std::stoul(argv[8]);

    std::cout << "argv[9]: " << argv[9] << std::endl;
    fileNameQueries = argv[9];

    std::cout << "argv[10]: " << argv[10] << std::endl;
    queriesOffset = std::stoul(argv[10]);

    std::cout << "argv[11]: " << argv[11] << std::endl;
    fileNameGroundTruth = argv[11];

    std::cout << "argv[12]: " << argv[12] << std::endl;
    numQueriesBegin = std::stoi(argv[12]);

    std::cout << "argv[13]: " << argv[13] << std::endl;
    numQueriesEnd = std::stoi(argv[13]);

    std::cout << "argv[14]: " << argv[14] << std::endl;
    nprobeBegin = std::stoi(argv[14]);

    std::cout << "argv[15]: " << argv[15] << std::endl;
    nprobeEnd = std::stoi(argv[15]);

    std::cout << "argv[16]: " << argv[16] << std::endl;
    kBegin = std::stoi(argv[16]);

    std::cout << "argv[17]: " << argv[17] << std::endl;
    kEnd = std::stoi(argv[17]);

    std::cout << "argv[18]: " << argv[18] << std::endl;
    isFloat = std::stoi(argv[18]);

    std::cout << "argv[19]: " << argv[19] << std::endl;
    numThreads = argc > 19 ? std::stoi(argv[19]) : 1;

    std::cout << "argv[20]: " << argv[20] << std::endl;
    ngpus = argc > 20 ? std::stoi(argv[20]) : 2;

    std::cout << "argv[21]: " << argv[21] << std::endl;
    useShards = argc > 21 ? std::stoi(argv[21]) : 0;

    std::cout << "argv[22]: " << argv[22] << std::endl;
    sharedGpuProcess = argc > 22 ? std::stoi(argv[22]) : 0;

    std::cout << "argv[23]: " << argv[23] << std::endl;
    shardPerProcess = argc > 23 ? std::stoi(argv[23]) : 1;

    std::cout << "argv[24]: " << argv[24] << std::endl;
    safeMemMargin = argc > 24 ? std::stoul(argv[24]) : 0;

    std::cout << "argv[25]: " << argv[25] << std::endl;
    fileNameCoarseQuantizer = argc > 25 ? argv[25] : "";

    std::cout << "argv[26]: " << argv[26] << std::endl;
    fileNameIndex = argc > 26 ? argv[26] : "";

    std::cout << "argv[27]: " << argv[27] << std::endl;
    profile = argc > 27 ? std::stoi(argv[27]) : 1;

    std::cout << "argv[28]: " << argv[28] << std::endl;
    allocLogging = argc > 28 ? std::stoi(argv[28]) : 0;

    std::cout << "argv[29]: " << argv[29] << std::endl;
    verbose = argc > 29 ? std::stoi(argv[29]) : 0;

    std::cout << "argv[30]: " << argv[30] << std::endl;
    nRuns = argc > 30 ? std::stoi(argv[30]) : 5;

    std::cout << "argv[31]: " << argv[31] << std::endl;
    pinnedMemoryMode = argc > 31 ? std::stoi(argv[31]) : 1;

    std::cout << "argv[32]: " << argv[32] << std::endl;
    usePrecomputed = argc > 32 ? std::stoi(argv[32]) : 1;

    std::cout << "argv[33]: " << argv[33] << std::endl;
    useMultiIndex = argc > 33 ? std::stoi(argv[33]) : 1;

    std::cout << "argv[34]: " << argv[34] << std::endl;
    useGpu = argc > 34 ? std::stoi(argv[34]) : 1;

    std::cout << "argv[35]: " << argv[35] << std::endl;
    printGpuMemory = argc > 35 ? std::stoi(argv[35]) : 1;

    std::cout << "argv[36]: " << argv[36] << std::endl;
    numQueryReplicas = argc > 36 ? std::stoi(argv[36]) : 0;

    std::cout << "argv[37]: " << argv[37] << std::endl;
    copyPerShard = argc > 37 ? std::stoi(argv[37]) : 0;

    int nProcesses, processRank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    std::cout << "numThreads: " << numThreads << std::endl;

    omp_set_num_threads(numThreads);

    std::cout << std::setprecision(6) << std::fixed;

    if (useMultiIndex) {
        demo<faiss::gpu::GpuIndexIMIPQConfig, faiss::gpu::GpuIndexIMIPQ>(
                isFloat,
                d,
                coarseCodebookSize,
                numSubQuantizers,
                nbitsSubQuantizer,
                fileNameTraining,
                numTrainingVecs,
                fileNameIndexing,
                numIndexingVecs,
                fileNameQueries,
                queriesOffset,
                fileNameGroundTruth,
                numQueriesBegin,
                numQueriesEnd,
                nprobeBegin,
                nprobeEnd,
                kBegin,
                kEnd,
                ngpus,
                useShards,
                nProcesses,
                processRank,
                sharedGpuProcess,
                shardPerProcess,
                safeMemMargin,
                fileNameCoarseQuantizer,
                fileNameIndex,
                profile,
                allocLogging,
                verbose,
                nRuns,
                pinnedMemoryMode,
                usePrecomputed,
                useMultiIndex,
                useGpu,
                printGpuMemory,
                numQueryReplicas,
                copyPerShard);
    } else {
        demo<faiss::gpu::GpuIndexIVFPQConfig, faiss::gpu::GpuIndexIVFPQ>(
                isFloat,
                d,
                coarseCodebookSize,
                numSubQuantizers,
                nbitsSubQuantizer,
                fileNameTraining,
                numTrainingVecs,
                fileNameIndexing,
                numIndexingVecs,
                fileNameQueries,
                queriesOffset,
                fileNameGroundTruth,
                numQueriesBegin,
                numQueriesEnd,
                nprobeBegin,
                nprobeEnd,
                kBegin,
                kEnd,
                ngpus,
                useShards,
                nProcesses,
                processRank,
                sharedGpuProcess,
                shardPerProcess,
                safeMemMargin,
                fileNameCoarseQuantizer,
                fileNameIndex,
                profile,
                allocLogging,
                verbose,
                nRuns,
                pinnedMemoryMode,
                usePrecomputed,
                useMultiIndex,
                useGpu,
                printGpuMemory,
                numQueryReplicas,
                copyPerShard);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
