/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/invlists/InvertedLists.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/IMIAppend.cuh>
#include <faiss/gpu/impl/IMIBase.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace faiss {
namespace gpu {

IMIBase::IMIBase(
        GpuResources* resources,
        MultiIndex2* quantizer,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : resources_(resources),
          quantizer_(quantizer),
          dim_(quantizer->getDim()),
          numLists_(quantizer->getSize()),
          interleavedLayout_(interleavedLayout),
          indicesOptions_(indicesOptions),
          space_(space),
          deviceListData_(
                  resources,
                  AllocInfo(
                          AllocType::InvListData,
                          getCurrentDevice(),
                          space_,
                          resources_->getDefaultStreamCurrentDevice())),
          deviceListIndices_(
                  resources,
                  AllocInfo(
                          AllocType::InvListIndices,
                          getCurrentDevice(),
                          space_,
                          resources_->getDefaultStreamCurrentDevice())),
          deviceListOffsets_(
                  resources,
                  makeDevAlloc(
                          AllocType::Other,
                          resources->getDefaultStreamCurrentDevice())),
          currentListLengths_(new std::vector<int>()),
          maxListLength_(0),
          isMemoryReserved_(false),
          numVecs_(0) {
    FAISS_ASSERT(numLists_ < std::numeric_limits<int>::max());
    reset();
}

IMIBase::~IMIBase() {}

void IMIBase::reserveMemory(
        const std::unordered_map<int, int>* expectedNumAddsPerList,
        int numVecs) {
    if (!expectedNumAddsPerList || expectedNumAddsPerList->empty() ||
        numVecs == 0) {
        return;
    }

    reset(numVecs);

    HostTensor<unsigned int, 1, true> newlistStartOffsets({numLists_ + 1});

    int lastListId = -1;
    unsigned int offset = 0;
    for (int listId = 0; listId < numLists_; listId++) {
        newlistStartOffsets[listId] = offset;

        auto entry = expectedNumAddsPerList->find(listId);
        if (entry != expectedNumAddsPerList->end()) {
            FAISS_ASSERT(listId == entry->first);
            auto& numAdds = entry->second;
            offset += numAdds;
            lastListId = listId;
        }
    }

    if (lastListId >= 0) {
        FAISS_ASSERT(offset <= std::numeric_limits<int>::max());
        for (int listId = lastListId + 1; listId < numLists_ + 1; listId++) {
            newlistStartOffsets[listId] = offset;
        }

        auto stream = resources_->getDefaultStreamCurrentDevice();

        DeviceTensor<unsigned int, 1, true> deviceListOffsetsTensor(
                deviceListOffsets_.data(), {(int)deviceListOffsets_.size()});
        DeviceTensor<unsigned int, 1, true> newlistStartOffsetsDevice(
                resources_,
                makeTempAlloc(AllocType::Other, stream),
                newlistStartOffsets);

        runIMIUpdateStartOffsets(
                deviceListOffsetsTensor, newlistStartOffsetsDevice, stream);

        isMemoryReserved_ = true;
    }
}

void IMIBase::reset() {
    deviceListData_.clear();
    deviceListIndices_.clear();
    deviceListOffsets_.clear();
    currentListLengths_->clear();

    maxListLength_ = 0;

    isMemoryReserved_ = false;
}

void IMIBase::reset(int numVecs) {
    numVecs_ = numVecs;

    deviceListData_.clear();
    deviceListIndices_.clear();
    deviceListOffsets_.clear();
    currentListLengths_->clear();

    for (size_t i = 0; i < numLists_; ++i) {
        listOffsetToUserIndex_.emplace_back(std::vector<idx_t>());
    }

    auto stream = resources_->getDefaultStreamCurrentDevice();

    auto dataNumBytes = getGpuVectorsEncodingSize_(numVecs_);
    deviceListData_.reserve(dataNumBytes, stream);

    size_t indicesNumBytes = numVecs_ *
            (indicesOptions_ == INDICES_32_BIT ? sizeof(int) : sizeof(idx_t));
    deviceListIndices_.reserve(indicesNumBytes, stream);

    deviceListOffsets_.reserve(numLists_ + 1, stream);
    deviceListOffsets_.resize(numLists_ + 1, stream);
    DeviceTensor<unsigned int, 1, true> deviceListOffsetsTensor(
            deviceListOffsets_.data(), {(int)deviceListOffsets_.size()});
    thrust::fill(
            thrust::cuda::par.on(stream),
            deviceListOffsetsTensor.data(),
            deviceListOffsetsTensor.end(),
            0);

    currentListLengths_->reserve(numLists_);
    currentListLengths_->resize(numLists_);
    for (auto& length : *currentListLengths_) {
        length = 0;
    }

    maxListLength_ = 0;

    isMemoryReserved_ = false;
}

int IMIBase::getMaxListLength() const {
    return maxListLength_;
}

int IMIBase::getDim() const {
    return dim_;
}

size_t IMIBase::getNumLists() const {
    return numLists_;
}

int IMIBase::getListLength(int listId) {
    FAISS_THROW_IF_NOT_FMT(
            listId < numLists_,
            "IVF list %d is out of bounds (%d lists total)",
            listId,
            numLists_);

    if (!isMemoryReserved_) {
        return 0;
    }

    auto stream = resources_->getDefaultStreamCurrentDevice();
    int listLength = 0;
    unsigned int offsets[2];
    fromDevice<unsigned int>(
            deviceListOffsets_.data() + listId, offsets, 2, stream);
    listLength = offsets[1] - offsets[0];

    CudaEvent copyEnd(stream);
    copyEnd.cpuWaitOnEvent();
    return listLength;
}

int IMIBase::getAllListsLength() {
    int length = 0;
    for (int i = 0; i < numLists_; i++) {
        length += getListLength(i);
    }
    return length;
}

unsigned int IMIBase::getListOffset(int listId) {
    FAISS_THROW_IF_NOT_FMT(
            listId < numLists_,
            "IVF list %d is out of bounds (%d lists total)",
            listId,
            numLists_);

    if (!isMemoryReserved_) {
        return 0;
    }

    auto stream = resources_->getDefaultStreamCurrentDevice();
    unsigned int offset = 0;
    fromDevice<unsigned int>(
            deviceListOffsets_.data() + listId, &offset, 1, stream);
    CudaEvent copyEnd(resources_->getDefaultStreamCurrentDevice());
    copyEnd.cpuWaitOnEvent();
    return offset;
}

std::vector<idx_t> IMIBase::getListIndices(int listId) {
    FAISS_THROW_IF_NOT_FMT(
            listId < numLists_,
            "IVF list %d is out of bounds (%d lists total)",
            listId,
            numLists_);

    if (!isMemoryReserved_) {
        return std::vector<idx_t>();
    }

    int listLength = getListLength(listId);

    if (listLength == 0) {
        return std::vector<idx_t>();
    }

    int listOffset = getListOffset(listId);
    auto stream = resources_->getDefaultStreamCurrentDevice();
    if (indicesOptions_ == INDICES_32_BIT) {
        std::vector<int> intInd(listLength);
        fromDevice<int>(
                (int*)deviceListIndices_.data() + listOffset,
                intInd.data(),
                listLength,
                stream);
        CudaEvent copyEnd(resources_->getDefaultStreamCurrentDevice());
        copyEnd.cpuWaitOnEvent();

        std::vector<idx_t> out(intInd.size());
        for (size_t i = 0; i < intInd.size(); ++i) {
            out[i] = (idx_t)intInd[i];
        }

        return out;
    } else if (indicesOptions_ == INDICES_64_BIT) {
        std::vector<idx_t> out(listLength);
        fromDevice<idx_t>(
                (idx_t*)deviceListIndices_.data() + listOffset,
                out.data(),
                listLength,
                stream);
        CudaEvent copyEnd(resources_->getDefaultStreamCurrentDevice());
        copyEnd.cpuWaitOnEvent();

        return out;
    } else if (indicesOptions_ == INDICES_CPU) {
        // The data is not stored on the GPU
        FAISS_ASSERT(listId < listOffsetToUserIndex_.size());

        auto& userIds = listOffsetToUserIndex_[listId];

        // this will return a copy
        return userIds;
    } else {
        // unhandled indices type (includes INDICES_IVF)
        FAISS_ASSERT(false);
        return std::vector<idx_t>();
    }
}

std::vector<uint8_t> IMIBase::getListVectorData(int listId, bool gpuFormat) {
    FAISS_THROW_IF_NOT_FMT(
            listId < numLists_,
            "IVF list %d is out of bounds (%d lists total)",
            listId,
            numLists_);

    if (!isMemoryReserved_) {
        return std::vector<uint8_t>();
    }

    int listLength = getListLength(listId);

    if (listLength == 0) {
        return std::vector<uint8_t>();
    }

    size_t listLengthNumBytes = getGpuVectorsEncodingSize_(listLength);
    unsigned int listOffset = getListOffset(listId);
    size_t listOffsetNumBytes = getGpuVectorsEncodingSize_(listOffset);

    std::vector<uint8_t> gpuCodes(listLengthNumBytes);
    auto stream = resources_->getDefaultStreamCurrentDevice();
    fromDevice<uint8_t>(
            deviceListData_.data() + listOffsetNumBytes,
            gpuCodes.data(),
            listLengthNumBytes,
            stream);
    CudaEvent copyEnd(resources_->getDefaultStreamCurrentDevice());
    copyEnd.cpuWaitOnEvent();
    if (gpuFormat) {
        return gpuCodes;
    } else {
        // The GPU layout may be different than the CPU layout (e.g., vectors
        // rather than dimensions interleaved), translate back if necessary
        return translateCodesFromGpu_(std::move(gpuCodes), listLength);
    }
}

void IMIBase::copyInvertedListsFrom(InvertedLists* ivf) {
    FAISS_ASSERT(ivf->nlist == numLists_);

    auto stream = resources_->getDefaultStreamCurrentDevice();
    HostTensor<unsigned int, 1, true> newlistStartOffsets({numLists_ + 1});
    unsigned int offset = 0;
    std::vector<uint8_t> codesVector;
    std::vector<idx_t> idsVector;

    for (size_t i = 0; i < numLists_; ++i) {
        size_t listSizeSt = ivf->list_size(i);

        // GPU index can only support max int entries per list
        FAISS_THROW_IF_NOT_FMT(
                listSizeSt <= (size_t)std::numeric_limits<int>::max(),
                "GPU inverted list can only support "
                "%zu entries; %zu found",
                (size_t)std::numeric_limits<int>::max(),
                listSizeSt);

        int listSize = listSizeSt;

        const uint8_t* codes = (const uint8_t*)ivf->get_codes(i);
        const idx_t* ids = ivf->get_ids(i);

        size_t listOffsetNumBytes = getGpuVectorsEncodingSize_(offset);
        size_t listLengthNumBytes = getGpuVectorsEncodingSize_(listSize);

        codesVector.resize(listLengthNumBytes);
        idsVector.resize(listSize);

        memcpy(codesVector.data(), codes, listLengthNumBytes);
        memcpy(idsVector.data(), ids, listSize * sizeof(idx_t));

        DeviceTensor<uint8_t, 1, true, long> deviceListDataTensor(
                deviceListData_.data() + listOffsetNumBytes,
                {(long)listLengthNumBytes});
        deviceListDataTensor.copyFrom(codesVector, stream);

        if (indicesOptions_ == INDICES_64_BIT) {
            DeviceTensor<idx_t, 1, true> deviceListIndexTensor(
                    (idx_t*)deviceListIndices_.data() + offset, {listSize});
            deviceListIndexTensor.copyFrom(idsVector, stream);
        } else {
            DeviceTensor<idx_t, 1, true> deviceListIndexTensorIdxT(
                    resources_,
                    makeTempAlloc(AllocType::Other, stream),
                    {listSize});
            deviceListIndexTensorIdxT.copyFrom(idsVector, stream);

            DeviceTensor<int, 1, true> deviceListIndexTensor(
                    (int*)deviceListIndices_.data() + offset, {listSize});
            convertTensor(
                    stream, deviceListIndexTensorIdxT, deviceListIndexTensor);
        }

        newlistStartOffsets[i] = offset;
        offset += (unsigned int)listSize;
        maxListLength_ = std::max(maxListLength_, listSize);

        CudaEvent copyEnd(stream);
        copyEnd.cpuWaitOnEvent();
    }

    newlistStartOffsets[numLists_] = offset;

    DeviceTensor<unsigned int, 1, true> deviceListOffsetsTensor(
            deviceListOffsets_.data(), {(int)deviceListOffsets_.size()});
    DeviceTensor<unsigned int, 1, true> newlistStartOffsetsDevice(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            newlistStartOffsets);

    runIMIUpdateStartOffsets(
            deviceListOffsetsTensor, newlistStartOffsetsDevice, stream);

    isMemoryReserved_ = true;
}

void IMIBase::copyInvertedListsTo(InvertedLists* ivf) {
    for (int i = 0; i < numLists_; ++i) {
        auto listIndices = getListIndices(i);
        auto listData = getListVectorData(i, false);

        ivf->add_entries(
                i, listIndices.size(), listIndices.data(), listData.data());
    }
}

int IMIBase::addVectors(
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
    FAISS_ASSERT(isMemoryReserved_);
    FAISS_ASSERT(vecs.getSize(0) % quantizer_->getNumCodebooks() == 0);
    FAISS_ASSERT(vecs.getSize(1) * quantizer_->getNumCodebooks() == dim_);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    int vecsPerCodebook = vecs.getSize(0) / quantizer_->getNumCodebooks();

    FAISS_ASSERT(vecsPerCodebook == indices.getSize(0));

    // Determine which IVF lists we need to append to

    // We don't actually need this
    DeviceTensor<float, 2, true> listDistance(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecsPerCodebook, 1});
    // We use this
    DeviceTensor<ushort2, 2, true> listIds2d(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecsPerCodebook, 1});

    quantizer_->query(vecs, 1, listDistance, listIds2d, false);

    // Copy the lists that we wish to append to back to the CPU
    // FIXME: really this can be into pinned memory and a true async
    // copy on a different stream; we can start the copy early, but it's
    // tiny
    auto listIdsHost = listIds2d.copyToVector(stream);

    // Now we add the encoded vectors to the individual lists
    // First, make sure that there is space available for adding the new
    // encoded vectors and indices

    // list id -> vectors being added
    std::unordered_map<int, std::vector<int>> listToVectorIds;

    // vector id -> which list it is being appended to
    std::vector<int> vectorIdToList(vecsPerCodebook);

    // vector id -> offset in list
    // (we already have vector id -> list id in listIds)
    std::vector<int> listOffsetHost(listIdsHost.size());

    // Number of valid vectors that we actually add; we return this
    int numAdded = 0;

    for (int i = 0; i < listIdsHost.size(); ++i) {
        // Add vector could be invalid (contains NaNs etc)
        if (listIdsHost[i].x == std::numeric_limits<unsigned short>::max() ||
            listIdsHost[i].y == std::numeric_limits<unsigned short>::max()) {
            listOffsetHost[i] = -1;
            vectorIdToList[i] = -1;
            continue;
        }

        auto listId = quantizer_->toMultiIndex(listIdsHost[i]);

        FAISS_ASSERT(listId < numLists_);
        ++numAdded;
        vectorIdToList[i] = listId;

        int offset = currentListLengths_->operator[](listId);

        auto it = listToVectorIds.find(listId);
        if (it != listToVectorIds.end()) {
            offset += it->second.size();
            it->second.push_back(i);
        } else {
            listToVectorIds[listId] = std::vector<int>{i};
        }

        listOffsetHost[i] = offset;
    }

    // If we didn't add anything (all invalid vectors that didn't map to IVF
    // clusters), no need to continue
    if (numAdded == 0) {
        return 0;
    }

    // unique lists being added to
    std::vector<int> uniqueLists;

    for (auto& vecs : listToVectorIds) {
        uniqueLists.push_back(vecs.first);
    }

    std::sort(uniqueLists.begin(), uniqueLists.end());

    // In the same order as uniqueLists, list the vectors being added to that
    // list contiguously (unique list 0 vectors ...)(unique list 1 vectors ...)
    // ...
    std::vector<int> vectorsByUniqueList;

    // For each of the unique lists, the start offset in vectorsByUniqueList
    std::vector<int> uniqueListVectorStart;

    // For each of the unique lists, where we start appending in that list by
    // the vector offset
    std::vector<int> uniqueListStartOffset;

    // For each of the unique lists, find the vectors which should be appended
    // to that list
    for (auto ul : uniqueLists) {
        uniqueListVectorStart.push_back(vectorsByUniqueList.size());

        FAISS_ASSERT(listToVectorIds.count(ul) != 0);

        // The vectors we are adding to this list
        auto& vecs = listToVectorIds[ul];
        vectorsByUniqueList.insert(
                vectorsByUniqueList.end(), vecs.begin(), vecs.end());

        // How many vectors we previously had (which is where we start appending
        // on the device)
        uniqueListStartOffset.push_back(currentListLengths_->operator[](ul));
    }

    // We terminate uniqueListVectorStart with the overall number of vectors
    // being added, which could be different than vecs.getSize(0) as some
    // vectors could be invalid
    uniqueListVectorStart.push_back(vectorsByUniqueList.size());

    {
        // Recalculate currentListLengths_ and max list length
        for (auto& counts : listToVectorIds) {
            auto listId = counts.first;
            int numVecsToAdd = counts.second.size();
            currentListLengths_->operator[](listId) += numVecsToAdd;

            // This is used by the multi-pass query to decide how much scratch
            // space to allocate for intermediate results
            maxListLength_ = std::max(
                    maxListLength_, currentListLengths_->operator[](listId));
        }
    }

    // If we're maintaining the indices on the CPU side, update our
    // map. We already resized our map above.
    if (indicesOptions_ == INDICES_CPU) {
        // We need to maintain the indices on the CPU side
        HostTensor<idx_t, 1, true> hostIndices(indices, stream);

        for (int i = 0; i < hostIndices.getSize(0); ++i) {
            // Add vector could be invalid (contains NaNs etc)
            if (listIdsHost[i].x ==
                        std::numeric_limits<unsigned short>::max() ||
                listIdsHost[i].y ==
                        std::numeric_limits<unsigned short>::max()) {
                continue;
            }

            auto listId = quantizer_->toMultiIndex(listIdsHost[i]);

            int offset = listOffsetHost[i];
            FAISS_ASSERT(offset >= 0);

            FAISS_ASSERT(listId < listOffsetToUserIndex_.size());
            auto& userIndices = listOffsetToUserIndex_[listId];

            FAISS_ASSERT(offset < userIndices.size());
            userIndices[offset] = hostIndices[i];
        }
    }

    // Copy the offsets to the GPU
    auto listIdsDevice = listIds2d.downcastOuter<1>();
    auto listOffsetDevice =
            toDeviceTemporary(resources_, listOffsetHost, stream);
    auto uniqueListsDevice = toDeviceTemporary(resources_, uniqueLists, stream);
    auto vectorsByUniqueListDevice =
            toDeviceTemporary(resources_, vectorsByUniqueList, stream);
    auto uniqueListVectorStartDevice =
            toDeviceTemporary(resources_, uniqueListVectorStart, stream);
    auto uniqueListStartOffsetDevice =
            toDeviceTemporary(resources_, uniqueListStartOffset, stream);

    // Actually encode and append the vectors
    appendVectors_(
            vecs,
            indices,
            uniqueListsDevice,
            vectorsByUniqueListDevice,
            uniqueListVectorStartDevice,
            uniqueListStartOffsetDevice,
            listIdsDevice,
            listOffsetDevice,
            stream);

    // We added this number
    return numAdded;
}

} // namespace gpu
} // namespace faiss
