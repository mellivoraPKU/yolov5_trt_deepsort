
#ifndef TENSORRT_BASICOPS_H
#define TENSORRT_BASICOPS_H

#ifndef WARNING
#define WARNING
#define __FILENAME__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__)
#define LOGGER std::cout << "[" << __func__ << "() " << __FILENAME__ << ":" << __LINE__ << "] "
#endif

#include <iostream>
#include <cmath>
#include "NvInfer.h"

#ifndef BLOCK
#define BLOCK 512
#endif

dim3 cudaGridSize(uint n);


#define CUDA_1D_KERNEL_LOOP(i, n)                                  \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
        i += blockDim.x * gridDim.x)

#ifndef CUDA_CHECK
#define CUDA_CHECK(condition)                               \
  /* Code block avoids redefinition of cudaError_t error */ \
  do {                                                      \
    cudaError_t error = (condition);                        \
    if (error != cudaSuccess) {                             \
      std::cout << __FILENAME__ << ":" << __LINE__ << " ";  \
      std::cout << cudaGetErrorString(error) << std::endl;  \
      abort();                                              \
    }                                                       \
  } while (0)
#endif

namespace onnx2trt {
    template<typename T>
    __host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
        return (a + b - 1) / b;
    }

    // a convenient interface to print device memory
    template<typename T>
    int devicePrint(const T *deviceValues, int length, const std::string &info, int step);

    template<typename T>
    int hostPrint(const T *values, int length, const std::string &info, int step);

    // namespace ops {
    //     // refer to detectron2.layers.ops.mask_head_slice
    //     cudaError_t maskHeadSlice(cudaStream_t stream, const void *logits, const void *pred, void *output,
    //                               int numMasks, int numClasses, int MaskHeight, int MaskWidth, nvinfer1::DataType type);

    //     // refer to detectron2.layers.ops.cuda_cast
    //     cudaError_t cudaCast(cudaStream_t stream, const void *input, void *output,
    //                          nvinfer1::Dims dims, nvinfer1::DataType type, nvinfer1::DataType castType);

    //     cudaError_t fmod(cudaStream_t stream, const int *inputs, int *outputs, int length, int denominator);
    // }
}

#endif //TENSORRT_BASICOPS_H
