
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

namespace faiss { namespace gpu {

/// Returns the current thread-local GPU device
int getCurrentDevice();

/// Sets the current thread-local GPU device
void setCurrentDevice(int device);

/// Returns the number of available GPU devices
int getNumDevices();

/// Synchronizes the CPU against all devices (equivalent to
/// cudaDeviceSynchronize for each device)
void synchronizeAllDevices();

/// Returns a cached cudaDeviceProp for the given device
cudaDeviceProp& getDeviceProperties(int device);

/// Returns the maximum number of threads available for the given GPU
/// device
int getMaxThreads(int device);

/// Equivalent to getMaxThreads(getCurrentDevice())
int getMaxThreadsCurrentDevice();

/// Returns the maximum smem available for the given GPU device
size_t getMaxSharedMemPerBlock(int device);

/// Equivalent to getMaxSharedMemPerBlock(getCurrentDevice())
size_t getMaxSharedMemPerBlockCurrentDevice();

/// For a given pointer, returns whether or not it is located on
/// a device (deviceId >= 0) or the host (-1).
int getDeviceForAddress(const void* p);

/// RAII object to set the current device, and restore the previous
/// device upon destruction
class DeviceScope {
 public:
  explicit DeviceScope(int device);
  ~DeviceScope();

 private:
  int prevDevice_;
};

/// RAII object to manage a cublasHandle_t
class CublasHandleScope {
 public:
  CublasHandleScope();
  ~CublasHandleScope();

  cublasHandle_t get() { return blasHandle_; }

 private:
  cublasHandle_t blasHandle_;
};

// RAII object to manage a cudaEvent_t
class CudaEvent {
 public:
  /// Creates an event and records it in this stream
  explicit CudaEvent(cudaStream_t stream);
  CudaEvent(const CudaEvent& event) = delete;
  CudaEvent(CudaEvent&& event) noexcept;
  ~CudaEvent();

  inline cudaEvent_t get() { return event_; }

  /// Wait on this event in this stream
  void streamWaitOnEvent(cudaStream_t stream);

  /// Have the CPU wait for the completion of this event
  void cpuWaitOnEvent();

  CudaEvent& operator=(CudaEvent&& event) noexcept;
  CudaEvent& operator=(CudaEvent& event) = delete;

 private:
  cudaEvent_t event_;
};

/// Wrapper to test return status of CUDA functions
#if DEBUG
#define CUDA_VERIFY(X)                          \
  do {                                          \
    auto err = (X);                             \
    FAISS_ASSERT(err == cudaSuccess);           \
  } while (0)
#else
#define CUDA_VERIFY(X) do { (X); } while (0)
#endif

/// Call for a collection of streams to wait on
template <typename L1, typename L2>
void streamWaitBase(const L1& listWaiting, const L2& listWaitOn) {
  // For all the streams we are waiting on, create an event
  std::vector<cudaEvent_t> events;
  for (auto& stream : listWaitOn) {
    cudaEvent_t event;
    CUDA_VERIFY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_VERIFY(cudaEventRecord(event, stream));
    events.push_back(event);
  }

  // For all the streams that are waiting, issue a wait
  for (auto& stream : listWaiting) {
    for (auto& event : events) {
      CUDA_VERIFY(cudaStreamWaitEvent(stream, event, 0));
    }
  }

  for (auto& event : events) {
    CUDA_VERIFY(cudaEventDestroy(event));
  }
}

/// These versions allow usage of initializer_list as arguments, since
/// otherwise {...} doesn't have a type
template <typename L1>
void streamWait(const L1& a,
                const std::initializer_list<cudaStream_t>& b) {
  streamWaitBase(a, b);
}

template <typename L2>
void streamWait(const std::initializer_list<cudaStream_t>& a,
                const L2& b) {
  streamWaitBase(a, b);
}

inline void streamWait(const std::initializer_list<cudaStream_t>& a,
                       const std::initializer_list<cudaStream_t>& b) {
  streamWaitBase(a, b);
}

} } // namespace
