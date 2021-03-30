/*
 * Copyright (c) 2020 MANAAI.
 *
 * This file is part of trt_onnx2
 * (see http://manaai.cn).
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#pragma once

#include <assert.h>

#include <algorithm>
#include <type_traits>
#include <vector>

#include "cuda_runtime.h"  // NOLINT
#include "logger.h"

namespace trtonnx2 {
namespace cuda {

#define ABORT_ON_CUDA_FAILURE(status)               \
  do {                                              \
    auto return_code = (status);                    \
    if (return_code != cudaSuccess) {               \
      gLogError << cudaGetErrorString(return_code); \
      abort();                                      \
    }                                               \
  } while (0)

#define RETURN_FALSE_ON_CUDA_FAILURE(status)        \
  do {                                              \
    auto return_code = (status);                    \
    if (return_code != cudaSuccess) {               \
      gLogError << cudaGetErrorString(return_code); \
      return false;                                 \
    }                                               \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// CUDA: number of blocks for threads.
constexpr int kThreadsPerBlock = sizeof(uint64_t) * 8;
constexpr int kTileSizeX = 16;
constexpr int kTileSizeY = 16;

constexpr int kBlockSize = 512;
inline int DivideUp(int a, int b) { return (a + b - 1) / b; }

// Used for kernel
dim3 cudaGridSize(uint n);
inline void *safeCudaMalloc(size_t memSize) {
  void *deviceMem;
  ABORT_ON_CUDA_FAILURE(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr) {
    std::cerr << "Out of memory" << std::endl;
    exit(1);
  }
  return deviceMem;
}

//// Utility kernels
template <typename T>
void CalculateArgMax(T const *const logits, size_t size, int num_classes,
                     size_t batch_offset, cudaStream_t stream,
                     unsigned char *argmax);
template <typename T>
void ReshapeToCHW(T const *const hwc_image, const int width, const int height,
                  const int channels, cudaStream_t stream, T *chw_image);

inline int GetBlocks(const int N, const int nthreads = kThreadsPerBlock) {
  return (N + nthreads - 1) / nthreads;
}

// Bankers' Rounding Algorthm, or known as "Round to even" algorithm
template <typename Dtype>
__host__ __device__ inline Dtype RoundGPU(const Dtype val) {
  if (abs(val - round(val)) == 0.5) {
    return round(val / 2.0) * 2.0;
  }
  return round(val);
}

template <typename Dtype>
__device__ Dtype Clip(const Dtype value, const Dtype lower_bound,
                      const Dtype upper_bound) {
  return max(lower_bound, min(value, upper_bound));
}

template <typename Dtype>
void Read(const char **buffer, Dtype *val) {
  *val = *reinterpret_cast<const Dtype *>(*buffer);
  *buffer += sizeof(Dtype);
}

template <typename Dtype>
void Write(char **buffer, const Dtype &val) {
  *reinterpret_cast<Dtype *>(*buffer) = val;
  *buffer += sizeof(Dtype);
}

__host__ __device__ inline float CastToFloat(const int value) {
  return static_cast<float>(value);
}

/// Memcpy
template <typename T, std::enable_if_t<!std::is_void<T>::value, int> = 0>
static inline void CudaMemcpy(T *dest, const T *src, size_t size,
                              cudaMemcpyKind kind) {
  ABORT_ON_CUDA_FAILURE(cudaMemcpy(dest, src, sizeof(T) * size, kind));
}

template <typename T, std::enable_if_t<std::is_void<T>::value, int> = 0>
static inline void CudaMemcpy(T *dest, const T *src, size_t size,
                              cudaMemcpyKind kind) {
  ABORT_ON_CUDA_FAILURE(cudaMemcpy(dest, src, size, kind));
}

template <typename T>
static inline void CudaMemcpyHostToDevice(T *dest, const T *src, size_t size) {
  CudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}
template <typename T>
static inline void CudaMemcpyDeviceToHost(T *dest, const T *src, size_t size) {
  CudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}
template <typename T>
static inline void CudaMemcpyDeviceToDevice(T *dest, const T *src,
                                            size_t size) {
  CudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
}

/// Memcpy asynchronously
template <typename T, std::enable_if_t<!std::is_void<T>::value, int> = 0>
static inline void CudaMemcpy(T *dest, const T *src, size_t size,
                              cudaMemcpyKind kind, cudaStream_t stream) {
  ABORT_ON_CUDA_FAILURE(
      cudaMemcpyAsync(dest, src, sizeof(T) * size, kind, stream));
}

/// Memcpy asynchronously
template <typename T, std::enable_if_t<std::is_void<T>::value, int> = 0>
static inline void CudaMemcpy(T *dest, const T *src, size_t size,
                              cudaMemcpyKind kind, cudaStream_t stream) {
  ABORT_ON_CUDA_FAILURE(cudaMemcpyAsync(dest, src, size, kind, stream));
}

template <typename T>
static inline void CudaMemcpyHostToDevice(T *dest, const T *src, size_t size,
                                          cudaStream_t stream) {
  CudaMemcpy(dest, src, size, cudaMemcpyHostToDevice, stream);
}
template <typename T>
static inline void CudaMemcpyDeviceToHost(T *dest, const T *src, size_t size,
                                          cudaStream_t stream) {
  CudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost, stream);
}

template <typename T>
static inline void CudaMemcpyDeviceToDevice(T *dest, const T *src, size_t size,
                                            cudaStream_t stream) {
  CudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice, stream);
}

/// Memset
template <typename T, std::enable_if_t<!std::is_void<T>::value, int> = 0>
static inline void CudaMemset(T *dest, int value, size_t size,
                              cudaStream_t stream) {
  ABORT_ON_CUDA_FAILURE(cudaMemsetAsync(dest, value, size * sizeof(T), stream));
}

template <typename T, std::enable_if_t<std::is_void<T>::value, int> = 0>
static inline void CudaMemset(T *dest, int value, size_t size,
                              cudaStream_t stream) {
  ABORT_ON_CUDA_FAILURE(cudaMemsetAsync(dest, value, size, stream));
}

template <typename T, std::enable_if_t<!std::is_void<T>::value, int> = 0>
static inline void CudaMemset(T *dest, int value, size_t size) {
  ABORT_ON_CUDA_FAILURE(cudaMemset(dest, value, size * sizeof(T)));
}

template <typename T, std::enable_if_t<std::is_void<T>::value, int> = 0>
static inline void CudaMemset(T *dest, int value, size_t size) {
  ABORT_ON_CUDA_FAILURE(cudaMemset(dest, value, size));
}

/// Malloc
template <typename T, std::enable_if_t<!std::is_void<T>::value, int> = 0>
static inline void CudaMalloc(T **dest, size_t size) {
  ABORT_ON_CUDA_FAILURE(
      cudaMalloc(reinterpret_cast<void **>(dest), size * sizeof(T)));
}

template <typename T, std::enable_if_t<std::is_void<T>::value, int> = 0>
static inline void CudaMalloc(T **dest, size_t size) {
  ABORT_ON_CUDA_FAILURE(cudaMalloc(reinterpret_cast<void **>(dest), size));
}

static inline void CudaMalloc(void **dest, size_t size) {
  ABORT_ON_CUDA_FAILURE(cudaMalloc(dest, size));
}

template <typename T, std::enable_if_t<!std::is_void<T>::value, int> = 0>
static inline void CudaHostRegister(T *dest, size_t size) {
  ABORT_ON_CUDA_FAILURE(
      cudaHostRegister(dest, sizeof(T) * size, cudaHostRegisterDefault));
}

template <typename T, std::enable_if_t<std::is_void<T>::value, int> = 0>
static inline void CudaHostRegister(T *dest, size_t size) {
  ABORT_ON_CUDA_FAILURE(cudaHostRegister(dest, size, cudaHostRegisterDefault));
}

static inline void CudaStreamSynchronize(cudaStream_t stream) {
  ABORT_ON_CUDA_FAILURE(cudaStreamSynchronize(stream));
}

static inline void CudaStreamSynchronize(
    const std::vector<cudaStream_t> &streams) {
  for (auto &stream : streams) {
    CudaStreamSynchronize(stream);
  }
}

template <typename T>
static inline void CudaFree(T *var) {
  if (var) ABORT_ON_CUDA_FAILURE(cudaFree(var));
  var = nullptr;
}

static inline void CudaFree(void *var) {
  if (var) ABORT_ON_CUDA_FAILURE(cudaFree(var));
  var = nullptr;
}

static inline void CudaSetDevice(const int id) {
  ABORT_ON_CUDA_FAILURE(cudaSetDevice(id));
}

static inline cudaStream_t CudaStreamCreate(const int device_id) {
  CudaSetDevice(device_id);
  cudaStream_t stream;
  ABORT_ON_CUDA_FAILURE(cudaStreamCreate(&stream));
  return stream;
}

static inline std::vector<cudaStream_t> CudaStreamsCreate(const int device_id,
                                                          const int n) {
  std::vector<cudaStream_t> streams;
  for (int i = 0; i < n; ++i) streams.push_back(CudaStreamCreate(device_id));
  return streams;
}

static inline cudaEvent_t CreateCudaEvent(const int device_id) {
  CudaSetDevice(device_id);
  cudaEvent_t event;
  ABORT_ON_CUDA_FAILURE(cudaEventCreate(&event));
  return event;
}

static inline void CudaGetDevice(int *device) {
  ABORT_ON_CUDA_FAILURE(cudaGetDevice(device));
}

template <typename T>
static inline void CudaFreeHost(T *var) {
  if (var) ABORT_ON_CUDA_FAILURE(cudaFreeHost(var));
  var = nullptr;
}

static inline void CudaEventRecord(const cudaEvent_t &event,
                                   const cudaStream_t &stream = NULL) {
  ABORT_ON_CUDA_FAILURE(cudaEventRecord(event, stream));
}

static inline void CudaEventSynchronize(const cudaEvent_t &event) {
  ABORT_ON_CUDA_FAILURE(cudaEventSynchronize(event));
}

inline void CudaMallocHost(void **ptr, const size_t size) {
  cudaMallocHost(ptr, size);
}

}  // namespace cuda
}  // namespace trtonnx2
