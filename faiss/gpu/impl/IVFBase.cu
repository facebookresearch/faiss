/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <faiss/gpu/impl/IVFBase.cuh>
#include <faiss/InvertedLists.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <limits>
#include <thrust/host_vector.h>
#include <unordered_map>

namespace faiss { namespace gpu {

IVFBase::DeviceIVFList::DeviceIVFList(GpuResources* res, const AllocInfo& info)
    : data(res, info),
      numVecs(0) {
}

IVFBase::IVFBase(GpuResources* resources,
                 faiss::MetricType metric,
                 float metricArg,
                 FlatIndex* quantizer,
                 IndicesOptions indicesOptions,
                 MemorySpace space) :
    resources_(resources),
    metric_(metric),
    metricArg_(metricArg),
    quantizer_(quantizer),
    dim_(quantizer->getDim()),
    numLists_(quantizer->getSize()),
    indicesOptions_(indicesOptions),
    space_(space),
    maxListLength_(0) {
  reset();
}

IVFBase::~IVFBase() {
}

void
IVFBase::reserveMemory(size_t numVecs) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  auto vecsPerList = numVecs / deviceListData_.size();
  if (vecsPerList < 1) {
    return;
  }

  auto bytesPerDataList = getGpuVectorsEncodingSize_(vecsPerList);

  for (auto& list : deviceListData_) {
    list->data.reserve(bytesPerDataList, stream);
  }

  if ((indicesOptions_ == INDICES_32_BIT) ||
      (indicesOptions_ == INDICES_64_BIT)) {
    // Reserve for index lists as well
    size_t bytesPerIndexList = vecsPerList *
      (indicesOptions_ == INDICES_32_BIT ? sizeof(int) : sizeof(long));

    for (auto& list : deviceListIndices_) {
      list->data.reserve(bytesPerIndexList, stream);
    }
  }

  // Update device info for all lists, since the base pointers may
  // have changed
  updateDeviceListInfo_(stream);
}

void
IVFBase::reset() {
  deviceListData_.clear();
  deviceListIndices_.clear();
  deviceListDataPointers_.clear();
  deviceListIndexPointers_.clear();
  deviceListLengths_.clear();
  listOffsetToUserIndex_.clear();

  auto info = AllocInfo(AllocType::IVFLists,
                        getCurrentDevice(),
                        space_,
                        resources_->getDefaultStreamCurrentDevice());

  for (size_t i = 0; i < numLists_; ++i) {
    deviceListData_.emplace_back(
      std::unique_ptr<DeviceIVFList>(new DeviceIVFList(resources_, info)));

    deviceListIndices_.emplace_back(
      std::unique_ptr<DeviceIVFList>(new DeviceIVFList(resources_, info)));

    listOffsetToUserIndex_.emplace_back(std::vector<long>());
  }

  deviceListDataPointers_.resize(numLists_, nullptr);
  deviceListIndexPointers_.resize(numLists_, nullptr);
  deviceListLengths_.resize(numLists_, 0);
  maxListLength_ = 0;
}

int
IVFBase::getDim() const {
  return dim_;
}

size_t
IVFBase::reclaimMemory() {
  // Reclaim all unused memory exactly
  return reclaimMemory_(true);
}

size_t
IVFBase::reclaimMemory_(bool exact) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  size_t totalReclaimed = 0;

  for (int i = 0; i < deviceListData_.size(); ++i) {
    auto& data = deviceListData_[i]->data;
    totalReclaimed += data.reclaim(exact, stream);

    deviceListDataPointers_[i] = data.data();
  }

  for (int i = 0; i < deviceListIndices_.size(); ++i) {
    auto& indices = deviceListIndices_[i]->data;
    totalReclaimed += indices.reclaim(exact, stream);

    deviceListIndexPointers_[i] = indices.data();
  }

  // Update device info for all lists, since the base pointers may
  // have changed
  updateDeviceListInfo_(stream);

  return totalReclaimed;
}

void
IVFBase::updateDeviceListInfo_(cudaStream_t stream) {
  std::vector<int> listIds(deviceListData_.size());
  for (int i = 0; i < deviceListData_.size(); ++i) {
    listIds[i] = i;
  }

  updateDeviceListInfo_(listIds, stream);
}

void
IVFBase::updateDeviceListInfo_(const std::vector<int>& listIds,
                               cudaStream_t stream) {
  HostTensor<int, 1, true>
    hostListsToUpdate({(int) listIds.size()});
  HostTensor<int, 1, true>
    hostNewListLength({(int) listIds.size()});
  HostTensor<void*, 1, true>
    hostNewDataPointers({(int) listIds.size()});
  HostTensor<void*, 1, true>
    hostNewIndexPointers({(int) listIds.size()});

  for (int i = 0; i < listIds.size(); ++i) {
    auto listId = listIds[i];
    auto& data = deviceListData_[listId];
    auto& indices = deviceListIndices_[listId];

    hostListsToUpdate[i] = listId;
    hostNewListLength[i] = data->numVecs;
    hostNewDataPointers[i] = data->data.data();
    hostNewIndexPointers[i] = indices->data.data();
  }

  // Copy the above update sets to the GPU
  DeviceTensor<int, 1, true> listsToUpdate(
    resources_, makeTempAlloc(AllocType::Other, stream), hostListsToUpdate);
  DeviceTensor<int, 1, true> newListLength(
    resources_, makeTempAlloc(AllocType::Other, stream), hostNewListLength);
  DeviceTensor<void*, 1, true> newDataPointers(
    resources_, makeTempAlloc(AllocType::Other, stream), hostNewDataPointers);
  DeviceTensor<void*, 1, true> newIndexPointers(
    resources_, makeTempAlloc(AllocType::Other, stream), hostNewIndexPointers);

  // Update all pointers to the lists on the device that may have
  // changed
  runUpdateListPointers(listsToUpdate,
                        newListLength,
                        newDataPointers,
                        newIndexPointers,
                        deviceListLengths_,
                        deviceListDataPointers_,
                        deviceListIndexPointers_,
                        stream);
}

size_t
IVFBase::getNumLists() const {
  return numLists_;
}

int
IVFBase::getListLength(int listId) const {
  FAISS_ASSERT(listId < deviceListLengths_.size());

  return deviceListLengths_[listId];
}

std::vector<long>
IVFBase::getListIndices(int listId) const {
  FAISS_ASSERT(listId < numLists_);

  if (indicesOptions_ == INDICES_32_BIT) {
    FAISS_ASSERT(listId < deviceListIndices_.size());

    auto intInd = deviceListIndices_[listId]->data.copyToHost<int>(
      resources_->getDefaultStreamCurrentDevice());

    std::vector<long> out(intInd.size());
    for (size_t i = 0; i < intInd.size(); ++i) {
      out[i] = (long) intInd[i];
    }

    return out;
  } else if (indicesOptions_ == INDICES_64_BIT) {
    FAISS_ASSERT(listId < deviceListIndices_.size());

    return deviceListIndices_[listId]->data.copyToHost<long>(
      resources_->getDefaultStreamCurrentDevice());
  } else if (indicesOptions_ == INDICES_CPU) {
    FAISS_ASSERT(listId < deviceListData_.size());
    FAISS_ASSERT(listId < listOffsetToUserIndex_.size());

    auto& userIds = listOffsetToUserIndex_[listId];

    // We should have the same number of indices on the CPU as we do vectors
    // encoded on the GPU
    FAISS_ASSERT(userIds.size() == deviceListData_[listId]->numVecs);

    // this will return a copy
    return userIds;
  } else {
    // unhandled indices type (includes INDICES_IVF)
    FAISS_ASSERT(false);
    return std::vector<long>();
  }
}

std::vector<unsigned char>
IVFBase::getListVectors(int listId) const {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  FAISS_ASSERT(listId < deviceListData_.size());
  auto& list = deviceListData_[listId];

  auto gpuCodes = list->data.copyToHost<unsigned char>(stream);

  // The GPU layout may be different than the CPU layout (e.g., vectors rather
  // than dimensions interleaved), translate back if necessary
  return translateCodesFromGpu_(std::move(gpuCodes), list->numVecs);
}

void
IVFBase::copyInvertedListsFrom(const InvertedLists* ivf) {
  size_t nlist = ivf ? ivf->nlist : 0;
  for (size_t i = 0; i < nlist; ++i) {
    size_t listSize = ivf->list_size(i);

    // GPU index can only support max int entries per list
    FAISS_THROW_IF_NOT_FMT(listSize <=
                           (size_t) std::numeric_limits<int>::max(),
                           "GPU inverted list can only support "
                           "%zu entries; %zu found",
                           (size_t) std::numeric_limits<int>::max(),
                           listSize);

    addEncodedVectorsToList_(i, ivf->get_codes(i), ivf->get_ids(i), listSize);
  }
}

void
IVFBase::copyInvertedListsTo(InvertedLists* ivf) {
  for (int i = 0; i < numLists_; ++i) {
    auto listIndices = getListIndices(i);
    auto listData = getListVectors(i);

    ivf->add_entries(i,
                     listIndices.size(),
                     listIndices.data(),
                     (const uint8_t*) listData.data());
  }
}

void
IVFBase::addEncodedVectorsToList_(int listId,
                                  const void* codes,
                                  const long* indices,
                                  size_t numVecs) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // This list must already exist
  FAISS_ASSERT(listId < deviceListData_.size());

  // This list must currently be empty
  auto& listCodes = deviceListData_[listId];
  FAISS_ASSERT(listCodes->data.size() == 0);
  FAISS_ASSERT(listCodes->numVecs == 0);

  // If there's nothing to add, then there's nothing we have to do
  if (numVecs == 0) {
    return;
  }

  // The GPU might have a different layout of the memory
  auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
  auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

  // We only have int32 length representaz3tions on the GPU per each
  // list; the length is in sizeof(char)
  FAISS_ASSERT(gpuListSizeInBytes <= (size_t) std::numeric_limits<int>::max());

  // Translate the codes as needed to our preferred form
  std::vector<unsigned char> codesV(cpuListSizeInBytes);
  std::memcpy(codesV.data(), codes, cpuListSizeInBytes);
  auto translatedCodes = translateCodesToGpu_(std::move(codesV), numVecs);

  listCodes->data.append(translatedCodes.data(),
                         gpuListSizeInBytes,
                         stream,
                         true /* exact reserved size */);
  listCodes->numVecs = numVecs;

  // Handle the indices as well
  addIndicesFromCpu_(listId, indices, numVecs);

  deviceListDataPointers_[listId] = listCodes->data.data();
  deviceListLengths_[listId] = numVecs;

  // We update this as well, since the multi-pass algorithm uses it
  maxListLength_ = std::max(maxListLength_, (int) numVecs);

  // device_vector add is potentially happening on a different stream
  // than our default stream
  if (resources_->getDefaultStreamCurrentDevice() != 0) {
    streamWait({stream}, {0});
  }
}

void
IVFBase::addIndicesFromCpu_(int listId,
                            const long* indices,
                            size_t numVecs) {
  auto stream = resources_->getDefaultStreamCurrentDevice();

  // This list must currently be empty
  auto& listIndices = deviceListIndices_[listId];
  FAISS_ASSERT(listIndices->data.size() == 0);
  FAISS_ASSERT(listIndices->numVecs == 0);

  if (indicesOptions_ == INDICES_32_BIT) {
    // Make sure that all indices are in bounds
    std::vector<int> indices32(numVecs);
    for (size_t i = 0; i < numVecs; ++i) {
      auto ind = indices[i];
      FAISS_ASSERT(ind <= (long) std::numeric_limits<int>::max());
      indices32[i] = (int) ind;
    }

    static_assert(sizeof(int) == 4, "");

    listIndices->data.append((unsigned char*) indices32.data(),
                             numVecs * sizeof(int),
                             stream,
                             true /* exact reserved size */);

  } else if (indicesOptions_ == INDICES_64_BIT) {
    listIndices->data.append((unsigned char*) indices,
                             numVecs * sizeof(long),
                             stream,
                             true /* exact reserved size */);
  } else if (indicesOptions_ == INDICES_CPU) {
    // indices are stored on the CPU
    FAISS_ASSERT(listId < listOffsetToUserIndex_.size());

    auto& userIndices = listOffsetToUserIndex_[listId];
    userIndices.insert(userIndices.begin(), indices, indices + numVecs);
  } else {
    // indices are not stored
    FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
  }

  deviceListIndexPointers_[listId] = listIndices->data.data();
}


int
IVFBase::addVectors(Tensor<float, 2, true>& vecs,
                    Tensor<long, 1, true>& indices) {
  FAISS_ASSERT(vecs.getSize(0) == indices.getSize(0));
  FAISS_ASSERT(vecs.getSize(1) == dim_);

  auto stream = resources_->getDefaultStreamCurrentDevice();

  // Number of valid vectors that we actually add; we return this
  int numAdded = 0;

  // Determine which IVF lists we need to append to

  // We don't actually need this
  DeviceTensor<float, 2, true> listDistance(
    resources_, makeTempAlloc(AllocType::Other, stream), {vecs.getSize(0), 1});
  // We use this
  DeviceTensor<int, 2, true> listIds2d(
    resources_, makeTempAlloc(AllocType::Other, stream), {vecs.getSize(0), 1});
  auto listIds = listIds2d.view<1>({vecs.getSize(0)});

  quantizer_->query(vecs, 1, metric_, metricArg_,
                    listDistance, listIds2d, false);

  // Copy the lists that we wish to append to back to the CPU
  // FIXME: really this can be into pinned memory and a true async
  // copy on a different stream; we can start the copy early, but it's
  // tiny
  HostTensor<int, 1, true> listIdsHost(listIds, stream);

  // Now we add the encoded vectors to the individual lists
  // First, make sure that there is space available for adding the new
  // encoded vectors and indices

  // list id -> # being added
  std::unordered_map<int, int> assignCounts;

  // vector id -> offset in list
  // (we already have vector id -> list id in listIds)
  HostTensor<int, 1, true> listOffsetHost({listIdsHost.getSize(0)});

  for (int i = 0; i < listIdsHost.getSize(0); ++i) {
    int listId = listIdsHost[i];

    // Add vector could be invalid (contains NaNs etc)
    if (listId < 0) {
      listOffsetHost[i] = -1;
      continue;
    }

    FAISS_ASSERT(listId < numLists_);
    ++numAdded;

    int offset = deviceListData_[listId]->numVecs;

    auto it = assignCounts.find(listId);
    if (it != assignCounts.end()) {
      offset += it->second;
      it->second++;
    } else {
      assignCounts[listId] = 1;
    }

    listOffsetHost[i] = offset;
  }

  // If we didn't add anything (all invalid vectors that didn't map to IVF
  // clusters), no need to continue
  if (numAdded == 0) {
    return 0;
  }

  // We need to resize the data structures for the inverted lists on
  // the GPUs, which means that they might need reallocation, which
  // means that their base address may change. Figure out the new base
  // addresses, and update those in a batch on the device
  {
    // Resize all of the lists that we are appending to
    for (auto& counts : assignCounts) {
      auto& codes = deviceListData_[counts.first];
      codes->data.resize(
        getGpuVectorsEncodingSize_(codes->numVecs + counts.second), stream);

      int newNumVecs = codes->numVecs + counts.second;
      codes->numVecs = newNumVecs;

      auto& indices = deviceListIndices_[counts.first];
      if ((indicesOptions_ == INDICES_32_BIT) ||
          (indicesOptions_ == INDICES_64_BIT)) {
        size_t indexSize =
          (indicesOptions_ == INDICES_32_BIT) ? sizeof(int) : sizeof(long);

        indices->data.resize(
          indices->data.size() + counts.second * indexSize, stream);
        indices->numVecs = newNumVecs;

      } else if (indicesOptions_ == INDICES_CPU) {
        // indices are stored on the CPU side
        FAISS_ASSERT(counts.first < listOffsetToUserIndex_.size());

        auto& userIndices = listOffsetToUserIndex_[counts.first];
        userIndices.resize(newNumVecs);
      } else {
        // indices are not stored on the GPU or CPU side
        FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
      }

      // This is used by the multi-pass query to decide how much scratch
      // space to allocate for intermediate results
      maxListLength_ = std::max(maxListLength_, newNumVecs);
    }

    // Update all pointers and sizes on the device for lists that we
    // appended to
    {
      std::vector<int> listIdsV(assignCounts.size());
      int i = 0;
      for (auto& counts : assignCounts) {
        listIdsV[i++] = counts.first;
      }

      updateDeviceListInfo_(listIdsV, stream);
    }
  }

  // If we're maintaining the indices on the CPU side, update our
  // map. We already resized our map above.
  if (indicesOptions_ == INDICES_CPU) {
    // We need to maintain the indices on the CPU side
    HostTensor<long, 1, true> hostIndices(indices, stream);

    for (int i = 0; i < hostIndices.getSize(0); ++i) {
      int listId = listIdsHost[i];

      // Add vector could be invalid (contains NaNs etc)
      if (listId < 0) {
        continue;
      }

      int offset = listOffsetHost[i];

      FAISS_ASSERT(listId < listOffsetToUserIndex_.size());
      auto& userIndices = listOffsetToUserIndex_[listId];

      FAISS_ASSERT(offset < userIndices.size());
      userIndices[offset] = hostIndices[i];
    }
  }

  // Copy the offsets to the GPU
  DeviceTensor<int, 1, true> listOffset(
    resources_, makeTempAlloc(AllocType::Other, stream), listOffsetHost);

  // Actually encode and append the vectors
  appendVectors_(vecs, indices, listIds, listOffset, stream);

  // We added this number
  return numAdded;
}


} } // namespace
