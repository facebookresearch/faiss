#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPQ.h>
#include <faiss/gpu/GpuIndexIMIPQ.h>
#include <faiss/gpu/GpuMultiIndex2.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/utils.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <string>
#include <utility>

void testAdd(
        int d,
        int coarseCodebookSize,
        int numSubQuantizers,
        int bitsPerCode,
        int numOfTrainingVecs,
        bool verbose = false) {
    FAISS_ASSERT(d % numSubQuantizers == 0);

    std::vector<float> trainVecs = faiss::gpu::randVecs(numOfTrainingVecs, d);

    { // multiple adds length
        faiss::gpu::StandardGpuResources res;
        // res.noTempMemory();
        faiss::gpu::GpuIndexIMIPQ imipqGpu(
                &res, d, coarseCodebookSize, numSubQuantizers, bitsPerCode);
        imipqGpu.train(numOfTrainingVecs, trainVecs.data());
        int numOfIndexVecs = 1;
        int numOfAdds = 3;
        std::vector<float> indexVecs = faiss::gpu::randVecs(numOfIndexVecs, d);
        for (int i = 0; i < numOfAdds; i++) {
            imipqGpu.updateExpectedNumAddsPerList(
                    numOfIndexVecs, indexVecs.data());
            faiss::gpu::CudaEvent updateEnd(
                    res.getResources()->getDefaultStreamCurrentDevice());
            updateEnd.cpuWaitOnEvent();
        }
        imipqGpu.applyExpectedNumAddsPerList();
        imipqGpu.resetExpectedNumAddsPerList();
        for (int i = 0; i < numOfAdds; i++) {
            imipqGpu.add(numOfIndexVecs, indexVecs.data());
        }
        EXPECT_EQ(numOfIndexVecs * numOfAdds, imipqGpu.getAllListsLength());
    }

    { // checking inverted lists
        std::vector<int> numOfIndexVecsList = {1, 10};
        for (int i = 0; i < numOfIndexVecsList.size(); i++) {
            faiss::gpu::StandardGpuResources res;
            // res.noTempMemory();
            faiss::gpu::GpuMultiIndex2 multiIndex(&res, d, coarseCodebookSize);
            faiss::gpu::GpuIndexIMIPQ imipqGpu(
                    &res, d, coarseCodebookSize, numSubQuantizers, bitsPerCode);

            multiIndex.train(numOfTrainingVecs, trainVecs.data());
            imipqGpu.train(numOfTrainingVecs, trainVecs.data());

            // checking initial length
            EXPECT_EQ(0, imipqGpu.getAllListsLength());

            std::vector<float> indexVecs =
                    faiss::gpu::randVecs(numOfIndexVecsList[i], d);

            imipqGpu.updateExpectedNumAddsPerList(
                    numOfIndexVecsList[i], indexVecs.data());
            faiss::gpu::CudaEvent updateEnd(
                    res.getResources()->getDefaultStreamCurrentDevice());
            updateEnd.cpuWaitOnEvent();
            imipqGpu.applyExpectedNumAddsPerList();
            imipqGpu.resetExpectedNumAddsPerList();
            imipqGpu.add(numOfIndexVecsList[i], indexVecs.data());

            // checking total length
            EXPECT_EQ(numOfIndexVecsList[i], imipqGpu.getAllListsLength());

            std::vector<faiss::idx_t> outLabels(numOfIndexVecsList[i]);
            std::vector<float> outDistances(numOfIndexVecsList[i]);
            multiIndex.search(
                    numOfIndexVecsList[i],
                    indexVecs.data(),
                    1,
                    outDistances.data(),
                    outLabels.data());

            std::vector<float> residuals(numOfIndexVecsList[i] * d);
            multiIndex.compute_nearest_residual_n(
                    numOfIndexVecsList[i], indexVecs.data(), residuals.data());

            int subCodebookSize = 1 << bitsPerCode;
            std::vector<faiss::IndexFlatL2> subCentroidsIndex(numSubQuantizers);
            std::vector<float> subCentroids = imipqGpu.getPQCentroids();
            int subDim = d / numSubQuantizers;

            if (verbose) {
                std::cout << "Index vecs:" << std::endl;
                for (int k = 0; k < numOfIndexVecsList[i]; k++) {
                    for (int l = 0; l < d; l++) {
                        std::cout << " " << indexVecs[k * d + l];
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;

                std::cout << "residuals:" << std::endl;
                for (int k = 0; k < numOfIndexVecsList[i]; k++) {
                    for (int l = 0; l < d; l++) {
                        std::cout << " " << residuals[k * d + l];
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }

            for (int j = 0; j < subCentroidsIndex.size(); j++) {
                float* subCentroidsView =
                        subCentroids.data() + j * subCodebookSize * subDim;
                subCentroidsIndex[j].code_size = sizeof(float) * subDim;
                subCentroidsIndex[j].d = subDim;
                subCentroidsIndex[j].add(subCodebookSize, subCentroidsView);
            }

            if (verbose) {
                for (int j = 0; j < numSubQuantizers; j++) {
                    std::cout << "flat-index: " << j << std::endl;
                    for (int k = 0; k < subCodebookSize; k++) {
                        for (int l = 0; l < subDim; l++) {
                            auto xb = subCentroidsIndex[j].get_xb();
                            std::cout << " " << xb[k * subDim + l];
                        }
                        std::cout << std::endl;
                    }
                }
                std::cout << std::endl;
            }

            std::vector<float> subResiduals(residuals.size());
            faiss::fvec_split(
                    subResiduals.data(),
                    numSubQuantizers,
                    residuals.data(),
                    numOfIndexVecsList[i],
                    subDim);

            if (verbose) {
                for (int j = 0; j < numSubQuantizers; j++) {
                    std::cout << "sub residuals: " << j << std::endl;
                    for (int k = 0; k < numOfIndexVecsList[i]; k++) {
                        for (int l = 0; l < subDim; l++) {
                            std::cout << " "
                                      << subResiduals
                                                 [(j * numOfIndexVecsList[i] +
                                                   k) * subDim +
                                                  l];
                        }
                        std::cout << std::endl;
                    }
                }
                std::cout << std::endl;
            }

            std::vector<faiss::idx_t> nearestSubCentroid(
                    numSubQuantizers * numOfIndexVecsList[i]);
            std::vector<float> nearestSubCentroidDistance(
                    numSubQuantizers * numOfIndexVecsList[i]);
            for (int j = 0; j < numSubQuantizers; j++) {
                subCentroidsIndex[j].search(
                        numOfIndexVecsList[i],
                        &subResiduals[j * numOfIndexVecsList[i] * subDim],
                        1,
                        &nearestSubCentroidDistance[j * numOfIndexVecsList[i]],
                        &nearestSubCentroid[j * numOfIndexVecsList[i]]);
            }

            if (verbose) {
                for (int j = 0; j < numSubQuantizers; j++) {
                    std::cout << "Nearest Indices: " << j << std::endl;
                    for (int k = 0; k < numOfIndexVecsList[i]; k++) {
                        std::cout << " "
                                  << nearestSubCentroid
                                             [j * numOfIndexVecsList[i] + k];
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;

                for (int j = 0; j < numSubQuantizers; j++) {
                    std::cout << "Nearest Distances: " << j << std::endl;
                    for (int k = 0; k < numOfIndexVecsList[i]; k++) {
                        std::cout << " "
                                  << nearestSubCentroidDistance
                                             [j * numOfIndexVecsList[i] + k];
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }

            std::vector<std::vector<uint8_t>> invListCodes(
                    coarseCodebookSize * coarseCodebookSize);
            std::vector<std::vector<faiss::idx_t>> invListIds(
                    coarseCodebookSize * coarseCodebookSize);
            for (int j = 0; j < invListIds.size(); j++) {
                invListCodes[j] = imipqGpu.getListVectorData(j);
                invListIds[j] = imipqGpu.getListIndices(j);
            }

            if (verbose) {
                for (int j = 0; j < invListCodes.size(); j++) {
                    std::cout << "Inv list: " << j << std::endl;
                    for (int k = 0; k < imipqGpu.getListLength(j); k++) {
                        std::cout << "Id " << invListIds[j][k] << " code:";
                        for (int l = 0; l < numSubQuantizers; l++) {
                            std::cout
                                    << " "
                                    << (int)invListCodes[j]
                                                        [k * numSubQuantizers +
                                                         l];
                        }
                        std::cout << std::endl;
                    }
                }
                std::cout << std::endl;
            }

            std::vector<int> expectedListLength(
                    coarseCodebookSize * coarseCodebookSize);
            for (int j = 0; j < expectedListLength.size(); j++) {
                expectedListLength[j] = 0;
            }

            for (int j = 0; j < numOfIndexVecsList[i]; j++) {
                int listId = outLabels[j];
                int currentListVec = expectedListLength[listId];

                // checking id
                EXPECT_EQ(j, invListIds[listId][currentListVec]);

                // checking code
                for (int k = 0; k < numSubQuantizers; k++) {
                    int codeId = currentListVec * numSubQuantizers + k;
                    int nearesSubCentroidId = k * numOfIndexVecsList[i] + j;
                    EXPECT_EQ(
                            (int)invListCodes[listId][codeId],
                            (int)nearestSubCentroid[nearesSubCentroidId])
                            << "Id: " << j
                            << " currentListVec: " << currentListVec
                            << " listId: " << listId << " codeId: " << codeId;
                }
                expectedListLength[listId]++;
            }

            // checking length
            for (int j = 0; j < expectedListLength.size(); j++) {
                EXPECT_EQ(expectedListLength[j], imipqGpu.getListLength(j));
            }
        }
    }
}

void testPrecomputedCodes(
        int d,
        int coarseCodebookSize,
        int numSubQuantizers,
        int bitsPerCode,
        int numOfTrainingVecs,
        bool verbose = false) {
    FAISS_ASSERT(d % numSubQuantizers == 0);
    faiss::gpu::StandardGpuResources res;
    // res.noTempMemory();

    faiss::gpu::GpuIndexIMIPQConfig config;
    config.usePrecomputedTables = true;
    faiss::gpu::GpuIndexIMIPQ imipqGpu(
            &res, d, coarseCodebookSize, numSubQuantizers, bitsPerCode, config);
    std::vector<float> trainVecs = faiss::gpu::randVecs(numOfTrainingVecs, d);
    imipqGpu.train(numOfTrainingVecs, trainVecs.data());

    faiss::gpu::GpuMultiIndex2* multiIndex = imipqGpu.getQuantizer();
    std::vector<float> centroids = multiIndex->getCentroids();

    if (verbose) {
        for (int i = 0; i < multiIndex->getNumCodebooks(); i++) {
            std::cout << "Codebook: " << i << std::endl;
            for (int j = 0; j < multiIndex->getCodebookSize(); j++) {
                for (int k = 0; k < multiIndex->getSubDim(); k++) {
                    std::cout << " "
                              << centroids
                                         [(i * multiIndex->getCodebookSize() +
                                           j) * multiIndex->getSubDim() +
                                          k];
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    FAISS_ASSERT(numSubQuantizers % multiIndex->getNumCodebooks() == 0);
    int numSubQuantizersPerCodebook =
            numSubQuantizers / multiIndex->getNumCodebooks();
    int subQuantizerDim = d / numSubQuantizers;
    std::vector<float> coarseSubCentroids(centroids.size());
    for (int i = 0; i < multiIndex->getNumCodebooks(); i++) {
        faiss::fvec_split(
                &coarseSubCentroids
                        [i * multiIndex->getCodebookSize() *
                         multiIndex->getSubDim()],
                numSubQuantizersPerCodebook,
                &centroids
                        [i * multiIndex->getCodebookSize() *
                         multiIndex->getSubDim()],
                multiIndex->getCodebookSize(),
                subQuantizerDim);
    }

    if (verbose) {
        for (int i = 0; i < numSubQuantizers; i++) {
            std::cout << "Coarse sub centroids: " << i << std::endl;
            for (int j = 0; j < multiIndex->getCodebookSize(); j++) {
                for (int k = 0; k < subQuantizerDim; k++) {
                    std::cout << " "
                              << coarseSubCentroids
                                         [(i * multiIndex->getCodebookSize() +
                                           j) * subQuantizerDim +
                                          k];
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    std::vector<float> subCentroids = imipqGpu.getPQCentroids();
    int subCodebookSize = 1 << bitsPerCode;

    if (verbose) {
        for (int i = 0; i < numSubQuantizers; i++) {
            std::cout << "Sub quantizer centroids: " << i << std::endl;
            for (int j = 0; j < subCodebookSize; j++) {
                for (int k = 0; k < subQuantizerDim; k++) {
                    std::cout << " "
                              << subCentroids
                                         [(i * subCodebookSize + j) *
                                                  subQuantizerDim +
                                          k];
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    std::vector<float> subCentroidsNorms(numSubQuantizers * subCodebookSize);

    if (verbose) {
        std::cout << "Sub quantizer norms:" << std::endl;
    }
    for (int i = 0; i < subCentroidsNorms.size(); i++) {
        subCentroidsNorms[i] = 0;
        if (verbose) {
            std::cout << " [" << i << "] =>";
        }
        for (int j = 0; j < subQuantizerDim; j++) {
            auto value = subCentroids[i * subQuantizerDim + j];
            subCentroidsNorms[i] += value * value;
            if (verbose) {
                std::cout << " + (" << value << " * " << value << ") ";
            }
        }
        if (verbose) {
            std::cout << "= " << subCentroidsNorms[i] << std::endl;
        }
    }
    if (verbose) {
        std::cout << std::endl;
    }

    std::vector<float> precomputedCodes = imipqGpu.getPrecomputedCodesVec();
    std::vector<float> expectedPreComputedCodes(
            multiIndex->getCodebookSize() * numSubQuantizers * subCodebookSize);

    for (int i = 0; i < multiIndex->getCodebookSize(); i++) {
        for (int j = 0; j < numSubQuantizers; j++) {
            for (int k = 0; k < subCodebookSize; k++) {
                int tableId = (i * numSubQuantizers + j) * subCodebookSize + k;
                expectedPreComputedCodes[tableId] = 0;
                for (int l = 0; l < subQuantizerDim; l++) {
                    int coarseSubId = (j * multiIndex->getCodebookSize() + i) *
                                    subQuantizerDim +
                            l;
                    int subId = (j * subCodebookSize + k) * subQuantizerDim + l;
                    expectedPreComputedCodes[tableId] +=
                            coarseSubCentroids[coarseSubId] *
                            subCentroids[subId];
                }
                expectedPreComputedCodes[tableId] *= 2;
                int normId = j * subCodebookSize + k;
                expectedPreComputedCodes[tableId] += subCentroidsNorms[normId];
                EXPECT_EQ(
                        precomputedCodes[tableId],
                        expectedPreComputedCodes[tableId]);
            }
        }
    }
}

bool compareIndexEntry(std::pair<long, float> x, std::pair<long, float> y) {
    return (x.second < y.second);
}

void testSearchPrecomputedCodes(
        int d,
        int numSubQuantizers,
        int bitsPerCode,
        bool verbose = false) {
    FAISS_ASSERT(d % numSubQuantizers == 0);

    int subCodebookSize = 1 << bitsPerCode;

    std::vector<int> coarseCodebookSizeList = {1, 2, 4, 4};
    std::vector<int> numOfIndexList = {1, 10, 4096, 1};
    for (int param1Idx = 0; param1Idx < coarseCodebookSizeList.size();
         param1Idx++) {
        int coarseCodebookSize = coarseCodebookSizeList[param1Idx];
        int numOfIndexVecs = numOfIndexList[param1Idx];

        if (verbose) {
            std::cout << "coarseCodebookSize: " << coarseCodebookSize
                      << std::endl;
            std::cout << "numOfIndexVecs: " << numOfIndexVecs << std::endl;
        }

        faiss::gpu::StandardGpuResources res;
        // res.noTempMemory();

        faiss::gpu::GpuIndexIMIPQConfig config;
        config.usePrecomputedTables = true;
        faiss::gpu::GpuIndexIMIPQ imipqGpu(
                &res,
                d,
                coarseCodebookSize,
                numSubQuantizers,
                bitsPerCode,
                config);

        int numOfTrainingVecs = (subCodebookSize > coarseCodebookSize)
                ? subCodebookSize * 39
                : coarseCodebookSize * 39;
        std::vector<float> trainVecs =
                faiss::gpu::randVecs(numOfTrainingVecs, d);
        imipqGpu.train(numOfTrainingVecs, trainVecs.data());

        std::vector<float> indexVecs = faiss::gpu::randVecs(numOfIndexVecs, d);
        imipqGpu.updateExpectedNumAddsPerList(numOfIndexVecs, indexVecs.data());
        faiss::gpu::CudaEvent updateEnd(
                res.getResources()->getDefaultStreamCurrentDevice());
        updateEnd.cpuWaitOnEvent();
        imipqGpu.applyExpectedNumAddsPerList();
        imipqGpu.resetExpectedNumAddsPerList();
        imipqGpu.add(numOfIndexVecs, indexVecs.data());

        std::vector<int> searchKList = {1, 10, 1024};
        std::vector<int> numOfQueriesList = {1, 10, 10};
        for (int param2Idx = 0; param2Idx < searchKList.size(); param2Idx++) {
            int searchK = searchKList[param2Idx];
            int numOfQueries = numOfQueriesList[param2Idx];

            if (verbose) {
                std::cout << "searchK: " << searchK << std::endl;
                std::cout << "numOfQueries: " << numOfQueries << std::endl;
            }

            std::vector<float> queries = faiss::gpu::randVecs(numOfQueries, d);
            std::vector<int> nprobeList = {
                    1, coarseCodebookSize * coarseCodebookSize};

            for (int param3Idx = 0; param3Idx < nprobeList.size();
                 param3Idx++) {
                int nprobe = nprobeList[param3Idx];

                if (verbose) {
                    std::cout << "nprobe: " << nprobe << std::endl;
                }

                std::vector<float> term1(numOfQueries * nprobe);
                std::vector<std::pair<ushort, ushort>> outLabelsPair(
                        numOfQueries * nprobe);
                faiss::gpu::GpuMultiIndex2* multiIndex =
                        imipqGpu.getQuantizer();
                multiIndex->search_pair(
                        numOfQueries,
                        queries.data(),
                        nprobe,
                        term1.data(),
                        outLabelsPair.data());

                if (verbose) {
                    std::cout << "Term 1:" << std::endl;
                    for (int i = 0; i < numOfQueries; i++) {
                        std::cout << "Query: " << i << std::endl;
                        for (int j = 0; j < nprobe; j++) {
                            std::cout << " ("
                                      << outLabelsPair[i * nprobe + j].first
                                      << ", "
                                      << outLabelsPair[i * nprobe + j].second
                                      << "):  " << term1[i * nprobe + j];
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }

                //(centroid id)(sub q)(code id)
                std::vector<float> term2 = imipqGpu.getPrecomputedCodesVec();

                if (verbose) {
                    std::cout << "Term 2:" << std::endl;
                    for (int i = 0; i < coarseCodebookSize; i++) {
                        std::cout << "Coarse centroid id: " << i << std::endl;
                        for (int j = 0; j < numSubQuantizers; j++) {
                            std::cout << "SubQuantizer: " << j << std::endl;
                            for (int k = 0; k < subCodebookSize; k++) {
                                std::cout << " "
                                          << term2[(i * numSubQuantizers + j) *
                                                           subCodebookSize +
                                                   k];
                            }
                            std::cout << std::endl;
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                }

                // (sub q)(code id)(sub dim)
                std::vector<float> subCentroids = imipqGpu.getPQCentroids();
                int subQuantizerDim = d / numSubQuantizers;

                if (verbose) {
                    std::cout << "Queries:" << std::endl;
                    for (int i = 0; i < numOfQueries; i++) {
                        for (int j = 0; j < d; j++) {
                            std::cout << " " << queries[i * d + j];
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;

                    for (int i = 0; i < numSubQuantizers; i++) {
                        std::cout << "Sub quantizer centroids: " << i
                                  << std::endl;
                        for (int j = 0; j < subCodebookSize; j++) {
                            for (int k = 0; k < subQuantizerDim; k++) {
                                std::cout << " "
                                          << subCentroids
                                                     [(i * subCodebookSize +
                                                       j) * subQuantizerDim +
                                                      k];
                            }
                            std::cout << std::endl;
                        }
                    }
                    std::cout << std::endl;
                }

                std::vector<float> term3(
                        numOfQueries * numSubQuantizers * subCodebookSize);
                for (int i = 0; i < numOfQueries; i++) {
                    for (int j = 0; j < numSubQuantizers; j++) {
                        for (int k = 0; k < subCodebookSize; k++) {
                            int term3Idx = (i * numSubQuantizers + j) *
                                            subCodebookSize +
                                    k;
                            term3[term3Idx] = 0;
                            for (int l = 0; l < subQuantizerDim; l++) {
                                int queryIdx = i * d + j * subQuantizerDim + l;
                                int subIdx = (j * subCodebookSize + k) *
                                                subQuantizerDim +
                                        l;
                                term3[term3Idx] += queries[queryIdx] *
                                        subCentroids[subIdx];
                            }
                            term3[term3Idx] *= -2;
                        }
                    }
                }

                if (verbose) {
                    std::cout << "Term 3:" << std::endl;
                    for (int i = 0; i < numOfQueries; i++) {
                        std::cout << "Query:" << i << std::endl;
                        for (int j = 0; j < numSubQuantizers; j++) {
                            std::cout << "SubQuantizer:" << j << std::endl;
                            for (int k = 0; k < subCodebookSize; k++) {
                                std::cout << " "
                                          << term3[(i * numSubQuantizers + j) *
                                                           subCodebookSize +
                                                   k];
                            }
                            std::cout << std::endl;
                        }
                    }
                    std::cout << std::endl;
                }

                if (verbose) {
                    std::vector<float> indexTerm3 =
                            imipqGpu.calcTerm3(numOfQueries, queries.data());
                    std::cout << "Index Term 3:" << std::endl;
                    for (int i = 0; i < numOfQueries; i++) {
                        std::cout << "Query:" << i << std::endl;
                        for (int j = 0; j < numSubQuantizers; j++) {
                            std::cout << "SubQuantizer:" << j << std::endl;
                            for (int k = 0; k < subCodebookSize; k++) {
                                std::cout << " "
                                          << indexTerm3
                                                     [(i * numSubQuantizers +
                                                       j) * subCodebookSize +
                                                      k];
                            }
                            std::cout << std::endl;
                        }
                    }
                    std::cout << std::endl;
                }

                std::vector<std::vector<uint8_t>> invListCodes(
                        coarseCodebookSize * coarseCodebookSize);
                std::vector<std::vector<faiss::idx_t>> invListIds(
                        coarseCodebookSize * coarseCodebookSize);
                for (int j = 0; j < invListIds.size(); j++) {
                    invListCodes[j] = imipqGpu.getListVectorData(j);
                    invListIds[j] = imipqGpu.getListIndices(j);
                }

                if (verbose) {
                    for (int i = 0; i < invListCodes.size(); i++) {
                        std::cout << "Inv list: " << i << std::endl;
                        for (int j = 0; j < imipqGpu.getListLength(i); j++) {
                            std::cout << "Id " << invListIds[i][j] << " code:";
                            for (int k = 0; k < numSubQuantizers; k++) {
                                std::cout << " "
                                          << (int)invListCodes
                                                     [i]
                                                     [j * numSubQuantizers + k];
                            }
                            std::cout << std::endl;
                        }
                    }
                    std::cout << std::endl;
                }

                std::vector<float> expectedOutDistances(numOfQueries * searchK);
                std::vector<float> expectedOutLabels(numOfQueries * searchK);
                std::vector<float> term23(numSubQuantizers * subCodebookSize);

                FAISS_ASSERT(
                        numSubQuantizers % multiIndex->getNumCodebooks() == 0);
                int numSubQuantizersPerCodebook =
                        numSubQuantizers / multiIndex->getNumCodebooks();

                for (int i = 0; i < numOfQueries; i++) {
                    if (verbose) {
                        std::cout << "Query: " << i << std::endl;
                    }

                    std::vector<std::pair<long, float>> allListsOuts;

                    for (int j = 0; j < nprobe; j++) {
                        std::pair<ushort, ushort> listId2 =
                                outLabelsPair[i * nprobe + j];
                        int listId = multiIndex->toMultiIndex(listId2);

                        if (verbose) {
                            std::cout << "List id: (";
                            std::cout << listId2.first << ", " << listId2.second
                                      << ") = ";
                            std::cout << listId << std::endl;
                        }

                        for (int k = 0; k < numSubQuantizersPerCodebook; k++) {
                            for (int l = 0; l < subCodebookSize; l++) {
                                term23[k * subCodebookSize + l] =
                                        term2[(listId2.first *
                                                       numSubQuantizers +
                                               k) * subCodebookSize +
                                              l] +
                                        term3[(i * numSubQuantizers + k) *
                                                      subCodebookSize +
                                              l];

                                if (verbose) {
                                    std::cout
                                            << term2[(listId2.first *
                                                              numSubQuantizers +
                                                      k) * subCodebookSize +
                                                     l];
                                    std::cout << " + "
                                              << term3[(i * numSubQuantizers +
                                                        k) * subCodebookSize +
                                                       l];
                                    std::cout << " = "
                                              << term23[k * subCodebookSize + l]
                                              << std::endl;
                                }
                            }
                        }

                        for (int k = numSubQuantizersPerCodebook;
                             k < 2 * numSubQuantizersPerCodebook;
                             k++) {
                            for (int l = 0; l < subCodebookSize; l++) {
                                term23[k * subCodebookSize + l] =
                                        term2[(listId2.second *
                                                       numSubQuantizers +
                                               k) * subCodebookSize +
                                              l] +
                                        term3[(i * numSubQuantizers + k) *
                                                      subCodebookSize +
                                              l];

                                if (verbose) {
                                    std::cout
                                            << term2[(listId2.second *
                                                              numSubQuantizers +
                                                      k) * subCodebookSize +
                                                     l];
                                    std::cout << " + "
                                              << term3[(i * numSubQuantizers +
                                                        k) * subCodebookSize +
                                                       l];
                                    std::cout << " = "
                                              << term23[k * subCodebookSize + l]
                                              << std::endl;
                                }
                            }
                        }

                        if (verbose) {
                            std::cout << "Term 23" << std::endl;
                            for (int k = 0; k < numSubQuantizers; k++) {
                                for (int l = 0; l < subCodebookSize; l++) {
                                    std::cout
                                            << " "
                                            << term23[k * subCodebookSize + l];
                                }
                                std::cout << std::endl;
                            }
                            std::cout << std::endl;
                        }

                        std::vector<std::pair<long, float>> listOuts;
                        for (int k = 0; k < imipqGpu.getListLength(listId);
                             k++) {
                            float result = term1[i * nprobe + j];

                            if (verbose) {
                                std::cout << "Code: " << k << std::endl;
                                std::cout << "Distance: "
                                          << term1[i * nprobe + j];
                            }

                            for (int l = 0; l < numSubQuantizers; l++) {
                                int subIdx = (int)
                                        invListCodes[listId]
                                                    [k * numSubQuantizers + l];
                                result += term23[l * subCodebookSize + subIdx];

                                if (verbose) {
                                    std::cout << " + "
                                              << term23[l * subCodebookSize +
                                                        subIdx];
                                }
                            }

                            listOuts.push_back(std::make_pair(
                                    invListIds[listId][k], result));

                            if (verbose) {
                                std::cout << " = " << result << std::endl;
                                std::cout << "Id: " << invListIds[listId][k]
                                          << std::endl;
                            }
                        }

                        std::sort(
                                listOuts.begin(),
                                listOuts.end(),
                                compareIndexEntry);
                        int limit = std::min(searchK, (int)listOuts.size());
                        for (int k = 0; k < limit; k++) {
                            allListsOuts.push_back(listOuts[k]);
                        }

                        if (verbose) {
                            std::cout << std::endl;
                        }
                    }

                    std::sort(
                            allListsOuts.begin(),
                            allListsOuts.end(),
                            compareIndexEntry);
                    for (int j = 0; j < searchK; j++) {
                        if (j < allListsOuts.size()) {
                            expectedOutLabels[i * searchK + j] =
                                    allListsOuts[j].first;
                            expectedOutDistances[i * searchK + j] =
                                    allListsOuts[j].second;
                        } else {
                            expectedOutLabels[i * searchK + j] = -1;
                        }
                    }

                    if (verbose) {
                        for (int j = 0; j < searchK; j++) {
                            std::cout << expectedOutLabels[i * searchK + j]
                                      << ": "
                                      << expectedOutDistances[i * searchK + j]
                                      << std::endl;
                        }
                        std::cout << std::endl;
                    }
                }

                imipqGpu.setNumProbes(nprobe);
                std::vector<float> outDistances(numOfQueries * searchK);
                std::vector<faiss::idx_t> outLabels(numOfQueries * searchK);
                imipqGpu.search(
                        numOfQueries,
                        queries.data(),
                        searchK,
                        outDistances.data(),
                        outLabels.data());

                if (verbose) {
                    std::cout << "Search: " << std::endl;
                    for (int i = 0; i < numOfQueries; i++) {
                        std::cout << "Query: " << i << std::endl;
                        for (int j = 0; j < searchK; j++) {
                            std::cout << outLabels[i * searchK + j] << ": "
                                      << outDistances[i * searchK + j]
                                      << std::endl;
                        }
                    }
                }

                for (int i = 0; i < numOfQueries; i++) {
                    for (int j = 0; j < searchK; j++) {
                        if (expectedOutLabels[i * searchK + j] != -1) {
                            EXPECT_EQ(
                                    expectedOutDistances[i * searchK + j],
                                    outDistances[i * searchK + j]);
                        } else {
                            EXPECT_EQ(
                                    expectedOutLabels[i * searchK + j],
                                    outLabels[i * searchK + j]);
                        }
                    }
                }
            }
        }
    }
}

void testCopyPrecomputedCodesFrom(
        int d,
        int nbitsCoarseQuantizer,
        int coarseCodebookSize,
        int numSubQuantizers,
        int nbitsSubQuantizer,
        int numOfTrainingVecs) {
    constexpr int M = 2;

    faiss::MultiIndexQuantizer multiIndexCpu(d, M, nbitsCoarseQuantizer);
    size_t nlist = coarseCodebookSize * coarseCodebookSize;
    faiss::IndexIVFPQ imipqCpu(
            &multiIndexCpu, d, nlist, numSubQuantizers, nbitsSubQuantizer);
    imipqCpu.quantizer_trains_alone = 1;

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexIMIPQConfig config;

    config.usePrecomputedTables = true;
    config.precomputeCodesOnCpu = true;

    faiss::gpu::GpuIndexIMIPQ imipqGpu(
            &res,
            d,
            coarseCodebookSize,
            numSubQuantizers,
            nbitsSubQuantizer,
            config);

    {
        std::vector<float> vecs = faiss::gpu::randVecs(numOfTrainingVecs, d);
        imipqCpu.train(numOfTrainingVecs, vecs.data());
        imipqGpu.train(numOfTrainingVecs, vecs.data());
    }

    imipqCpu.use_precomputed_table = 2;
    imipqCpu.precompute_table();
    imipqGpu.copyPrecomputedCodesFrom(imipqCpu.precomputed_table.data());

    {
        std::vector<float> term2Gpu = imipqGpu.getPrecomputedCodesVec();

        for (size_t i = 0; i < term2Gpu.size(); i++) {
            EXPECT_EQ(term2Gpu[i], imipqCpu.precomputed_table[i]);
        }
    }
}

void testCopyFrom(
        int d,
        int nbitsCoarseQuantizer,
        int coarseCodebookSize,
        int numSubQuantizers,
        int nbitsSubQuantizer,
        int numOfTrainingVecs,
        int numOfIndexVecs) {
    constexpr int M = 2;

    faiss::MultiIndexQuantizer multiIndexCpu(d, M, nbitsCoarseQuantizer);
    size_t nlist = coarseCodebookSize * coarseCodebookSize;
    faiss::IndexIVFPQ imipqCpu(
            &multiIndexCpu, d, nlist, numSubQuantizers, nbitsSubQuantizer);

    imipqCpu.quantizer_trains_alone = 1;

    {
        std::vector<float> vecs = faiss::gpu::randVecs(numOfTrainingVecs, d);
        imipqCpu.train(numOfTrainingVecs, vecs.data());
    }

    if (numOfIndexVecs > 0) {
        std::vector<float> indexVecs = faiss::gpu::randVecs(numOfIndexVecs, d);
        imipqCpu.add(numOfIndexVecs, indexVecs.data());
    }

    imipqCpu.use_precomputed_table = 2;
    imipqCpu.precompute_table();

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexIMIPQConfig config;

    config.usePrecomputedTables = true;

    faiss::gpu::GpuIndexIMIPQ imipqGpu(&res, &imipqCpu, config);

    faiss::gpu::GpuMultiIndex2* multiIndexGpu = imipqGpu.getQuantizer();

    EXPECT_EQ(multiIndexGpu->ntotal, multiIndexCpu.ntotal);
    EXPECT_EQ(multiIndexGpu->getCodebookSize(), coarseCodebookSize);
    EXPECT_EQ(multiIndexGpu->ntotal, coarseCodebookSize * coarseCodebookSize);
    EXPECT_EQ(multiIndexGpu->getNumCodebooks(), multiIndexCpu.pq.M);
    EXPECT_EQ(multiIndexGpu->getSubDim(), multiIndexCpu.pq.dsub);

    {
        std::vector<float> gpuCoarseCentroids = multiIndexGpu->getCentroids();
        EXPECT_EQ(gpuCoarseCentroids, multiIndexCpu.pq.centroids);
    }

    EXPECT_EQ(imipqGpu.getNumSubQuantizers(), imipqCpu.pq.M);
    EXPECT_EQ(imipqGpu.getBitsPerCode(), imipqCpu.pq.nbits);

    {
        std::vector<float> gpuPQCentroids = imipqGpu.getPQCentroids();
        EXPECT_EQ(gpuPQCentroids, imipqCpu.pq.centroids);
    }

    {
        std::vector<float> term2 = imipqGpu.getPrecomputedCodesVec();
        EXPECT_TRUE(term2.size() > 0);
    }

    EXPECT_EQ(imipqCpu.nlist, imipqGpu.nlist);
    for (int i = 0; i < imipqGpu.nlist; i++) {
        std::vector<faiss::idx_t> gpuIndices = imipqGpu.getListIndices(i);
        const faiss::idx_t* cpuIndices = imipqCpu.invlists->get_ids(i);
        EXPECT_EQ(gpuIndices.size(), imipqCpu.invlists->list_size(i));

        for (int j = 0; j < gpuIndices.size(); j++) {
            EXPECT_EQ(gpuIndices[j], cpuIndices[j]);
        }

        const uint8_t* cpuCodes = imipqCpu.invlists->get_codes(i);
        std::vector<uint8_t> gpuCodes = imipqGpu.getListVectorData(i);
        for (int j = 0; j < gpuCodes.size(); j++) {
            EXPECT_EQ(gpuCodes[j], cpuCodes[j]);
        }
    }
}

void testCopyTo(
        int d,
        int coarseCodebookSize,
        int numSubQuantizers,
        int nbitsSubQuantizer,
        int numOfTrainingVecs,
        int numOfIndexVecs) {
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexIMIPQConfig config;

    config.usePrecomputedTables = true;

    faiss::gpu::GpuIndexIMIPQ imipqGpu(
            &res,
            d,
            coarseCodebookSize,
            numSubQuantizers,
            nbitsSubQuantizer,
            config);

    {
        std::vector<float> vecs = faiss::gpu::randVecs(numOfTrainingVecs, d);
        imipqGpu.train(numOfTrainingVecs, vecs.data());
    }

    if (numOfIndexVecs > 0) {
        std::vector<float> indexVecs = faiss::gpu::randVecs(numOfIndexVecs, d);
        imipqGpu.updateExpectedNumAddsPerList(numOfIndexVecs, indexVecs.data());
        faiss::gpu::CudaEvent updateEnd(
                res.getResources()->getDefaultStreamCurrentDevice());
        updateEnd.cpuWaitOnEvent();
        imipqGpu.applyExpectedNumAddsPerList();
        imipqGpu.resetExpectedNumAddsPerList();
        imipqGpu.add(numOfIndexVecs, indexVecs.data());
    }

    faiss::IndexIVFPQ imipqCpu;

    imipqGpu.copyTo(&imipqCpu);

    faiss::MultiIndexQuantizer* multiIndexCpu =
            dynamic_cast<faiss::MultiIndexQuantizer*>(imipqCpu.quantizer);

    EXPECT_TRUE(multiIndexCpu);

    faiss::gpu::GpuMultiIndex2* multiIndexGpu = imipqGpu.getQuantizer();

    EXPECT_EQ(multiIndexCpu->ntotal, multiIndexGpu->ntotal);
    EXPECT_EQ(multiIndexCpu->ntotal, coarseCodebookSize * coarseCodebookSize);
    EXPECT_EQ(multiIndexCpu->pq.M, multiIndexGpu->getNumCodebooks());
    EXPECT_EQ(multiIndexCpu->pq.dsub, multiIndexGpu->getSubDim());

    {
        std::vector<float> gpuCoarseCentroids = multiIndexGpu->getCentroids();
        EXPECT_EQ(multiIndexCpu->pq.centroids, gpuCoarseCentroids);
    }

    EXPECT_EQ(imipqCpu.pq.M, imipqGpu.getNumSubQuantizers());
    EXPECT_EQ(imipqCpu.pq.nbits, imipqGpu.getBitsPerCode());
    EXPECT_TRUE(imipqCpu.precomputed_table.size() > 0);
    EXPECT_TRUE(imipqCpu.use_precomputed_table == 2);
    EXPECT_TRUE(imipqCpu.quantizer_trains_alone == 1);

    {
        std::vector<float> gpuPQCentroids = imipqGpu.getPQCentroids();
        EXPECT_EQ(imipqCpu.pq.centroids, gpuPQCentroids);
    }

    EXPECT_EQ(imipqCpu.nlist, imipqGpu.nlist);
    for (int i = 0; i < imipqGpu.nlist; i++) {
        std::vector<faiss::idx_t> gpuIndices = imipqGpu.getListIndices(i);
        const faiss::idx_t* cpuIndices = imipqCpu.invlists->get_ids(i);
        EXPECT_EQ(gpuIndices.size(), imipqCpu.invlists->list_size(i));

        for (int j = 0; j < gpuIndices.size(); j++) {
            EXPECT_EQ(gpuIndices[j], cpuIndices[j]);
        }

        const uint8_t* cpuCodes = imipqCpu.invlists->get_codes(i);
        std::vector<uint8_t> gpuCodes = imipqGpu.getListVectorData(i);
        for (int j = 0; j < gpuCodes.size(); j++) {
            EXPECT_EQ(gpuCodes[j], cpuCodes[j]);
        }
    }
}

void testComparePrecomputedCodesWithCpu(
        int d,
        int coarseCodebookSize,
        int numSubQuantizers,
        int nbitsSubQuantizer,
        int numOfTrainingVecs) {
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexIMIPQConfig config;

    config.usePrecomputedTables = true;

    faiss::gpu::GpuIndexIMIPQ imipqGpu(
            &res,
            d,
            coarseCodebookSize,
            numSubQuantizers,
            nbitsSubQuantizer,
            config);

    {
        std::vector<float> vecs = faiss::gpu::randVecs(numOfTrainingVecs, d);
        imipqGpu.train(numOfTrainingVecs, vecs.data());
    }

    faiss::IndexIVFPQ imipqCpu;

    imipqGpu.copyTo(&imipqCpu);

    {
        std::vector<float> term2Gpu = imipqGpu.getPrecomputedCodesVec();

        for (size_t i = 0; i < term2Gpu.size(); i++) {
            EXPECT_EQ(term2Gpu[i], imipqCpu.precomputed_table[i]);
        }
    }
}

TEST(TestGpuIndexIMIPQ, testGetNumLists) {
    constexpr int d = 2;
    constexpr int numSubQuantizers = 2;
    constexpr int bitsPerCode = 8;
    std::vector<int> coarseCodebookSizeList = {1, 2};
    for (int i = 0; i < coarseCodebookSizeList.size(); i++) {
        faiss::gpu::StandardGpuResources res;
        // res.noTempMemory();
        faiss::gpu::GpuIndexIMIPQ imipqGpu(
                &res,
                d,
                coarseCodebookSizeList[i],
                numSubQuantizers,
                bitsPerCode);
        EXPECT_EQ(
                imipqGpu.getNumLists(),
                coarseCodebookSizeList[i] * coarseCodebookSizeList[i]);
    }
}

TEST(TestGpuIndexIMIPQ, testAdd) {
    int d, coarseCodebookSize, numSubQuantizers, bitsPerCode, numOfTrainingVecs;
    d = 4;
    coarseCodebookSize = 2;
    numSubQuantizers = 2;
    bitsPerCode = 8;
    numOfTrainingVecs = (1 << bitsPerCode) * 39;
    testAdd(d,
            coarseCodebookSize,
            numSubQuantizers,
            bitsPerCode,
            numOfTrainingVecs);
}

TEST(TestGpuIndexIMIPQ, testPrecomputedCodes) {
    int d, coarseCodebookSize, numSubQuantizers, bitsPerCode, numOfTrainingVecs;
    d = 4;
    coarseCodebookSize = 2;
    numSubQuantizers = 2;
    bitsPerCode = 8;
    numOfTrainingVecs = (1 << bitsPerCode) * 39;
    testPrecomputedCodes(
            d,
            coarseCodebookSize,
            numSubQuantizers,
            bitsPerCode,
            numOfTrainingVecs);
}

TEST(TestGpuIndexIMIPQ, testSearchPrecomputedCodes) {
    int d, numSubQuantizers, bitsPerCode;
    d = 4;
    numSubQuantizers = 4;
    bitsPerCode = 8;
    testSearchPrecomputedCodes(d, numSubQuantizers, bitsPerCode);
}

TEST(TestGpuIndexIMIPQ, copyPrecomputedCodesFrom) {
    std::vector<int> dList = {4};
    std::vector<int> nbitsList = {2};
    std::vector<int> numCentroidsPerCodebookList = {4};
    int numSubQuantizers = 2;
    int bitsPerCode = 8;

    for (int i = 0; i < dList.size(); i++) {
        for (int j = 0; j < numCentroidsPerCodebookList.size(); j++) {
            int numTrainingVecs = std::max(
                                          (1 << bitsPerCode),
                                          numCentroidsPerCodebookList[j]) *
                    39;
            testCopyPrecomputedCodesFrom(
                    dList[i],
                    nbitsList[j],
                    numCentroidsPerCodebookList[j],
                    numSubQuantizers,
                    bitsPerCode,
                    numTrainingVecs);
        }
    }
}

TEST(TestGpuIndexIMIPQ, copyFrom) {
    std::vector<int> dList = {2, 2, 2, 2, 2, 2, 4};
    std::vector<int> nbitsList = {0, 0, 0, 1, 1, 1, 1};
    std::vector<int> numCentroidsPerCodebookList = {1, 1, 1, 2, 2, 2, 2};
    std::vector<int> numOfIndexVecsList = {2, 2, 2, 2, 2, 2, 4};
    int numSubQuantizers = 2;
    int bitsPerCode = 8;

    for (int i = 0; i < dList.size(); i++) {
        int numTrainingVecs =
                std::max((1 << bitsPerCode), numCentroidsPerCodebookList[i]) *
                39;
        testCopyFrom(
                dList[i],
                nbitsList[i],
                numCentroidsPerCodebookList[i],
                numSubQuantizers,
                bitsPerCode,
                numTrainingVecs,
                numOfIndexVecsList[i]);
    }
}

TEST(TestGpuIndexIMIPQ, copyTo) {
    std::vector<int> dList = {2, 2, 2, 2, 2, 2, 4};
    std::vector<int> numCentroidsPerCodebookList = {1, 1, 1, 2, 2, 2, 2};
    std::vector<int> numOfIndexVecsList = {2, 2, 2, 2, 2, 2, 4};
    int numSubQuantizers = 2;
    int bitsPerCode = 8;

    for (int i = 0; i < dList.size(); i++) {
        int numTrainingVecs =
                std::max((1 << bitsPerCode), numCentroidsPerCodebookList[i]) *
                39;
        testCopyTo(
                dList[i],
                numCentroidsPerCodebookList[i],
                numSubQuantizers,
                bitsPerCode,
                numTrainingVecs,
                numOfIndexVecsList[i]);
    }
}

TEST(TestGpuIndexIMIPQ, comparePrecomputedCodesWithCpu) {
    std::vector<int> dList = {4};
    std::vector<int> numCentroidsPerCodebookList = {1, 4, 6};
    int numSubQuantizers = 2;
    int bitsPerCode = 8;

    for (int i = 0; i < dList.size(); i++) {
        for (int j = 0; j < numCentroidsPerCodebookList.size(); j++) {
            int numTrainingVecs = std::max(
                                          (1 << bitsPerCode),
                                          numCentroidsPerCodebookList[j]) *
                    39;
            testComparePrecomputedCodesWithCpu(
                    dList[i],
                    numCentroidsPerCodebookList[j],
                    numSubQuantizers,
                    bitsPerCode,
                    numTrainingVecs);
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
