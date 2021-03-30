
#include <cassert>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <vector>

#include "NvInferPlugin.h"
#include "basicOps.cuh"



dim3 cudaGridSize(uint n)
{
    uint k = (n - 1) /BLOCK + 1;
    uint x = k ;
    uint y = 1 ;
    if (x > 65535 )
    {
        x = ceil(sqrt(x));
        y = (n - 1 )/(x*BLOCK) + 1;
    }
    dim3 d = {x,y,1} ;
    return d;
}


std::ostream &operator<<(std::ostream &output, const half &value) {
    output << static_cast<float>(value);
    return output;
}

namespace onnx2trt {
    // a convenient interface to print device memory
    template<typename T>
    int devicePrint(const T *deviceValues, int length, const std::string &info, int step) {
        T *values = (T *)malloc(sizeof(T) * length);
        cudaMemcpy(values, deviceValues, sizeof(T) * length, cudaMemcpyDeviceToHost);
        std::cout << info << ": ";
        for (int i = 0; i < length; i++) {
            if (step != 1) {
                if (!(i % step)) {
                    std::cout << std::endl;
                }
            }
            std::cout << values[i] << " ";
        }
        std::cout << std::endl;
        free(values);
        return 0;
    }
    template int devicePrint<float>(const float *, int, const std::string &, int);
    template int devicePrint<half>(const half *, int, const std::string &, int);
    template int devicePrint<int>(const int *, int, const std::string &, int);

    template<typename T>
    int hostPrint(const T *values, int length, const std::string &info, int step) {
        std::cout << info << ": ";
        for (int i = 0; i < length; i++) {
            if (step != 1) {
                if (!(i % step)) {
                    std::cout << std::endl;
                }
            }
            std::cout << values[i] << " ";
        }
        std::cout << std::endl;
        return 0;
    }
    template int hostPrint<float>(const float *, int, const std::string &, int);
    template int hostPrint<half>(const half *, int, const std::string &, int);
    template int hostPrint<int>(const int *, int, const std::string &, int);

    // namespace ops {
    //     // refer to detectron2.layers.ops.mask_head_slice
    //     // perform slice operation on prediction masks with given class indices
    //     // inputs:
    //     // logits, (num_det, class_num, mask_height, mask_width)
    //     // pred, (num_det,)
    //     // outputs:
    //     // output, (num_det, mask_height, mask_width)
    //     template<typename T, typename I>
    //     __global__ void sliceKernel(const int64_t nThreads, const T *logits, const I *pred, T *output,
    //                                 int numMasks, int numClasses, int MaskHeight, int MaskWidth) {
    //         CUDA_1D_KERNEL_LOOP(index, nThreads) {
    //             int pw = index % MaskWidth;
    //             int ph = (index / MaskWidth) % MaskHeight;
    //             int n = index / MaskWidth / MaskHeight;

    //             int classIndex = (int)pred[n];
    //             const T *offset = logits +
    //                     (((int64_t)n * (int64_t)numClasses + classIndex) * (int64_t)MaskHeight + ph) * (int64_t)MaskWidth + pw;
    //             output[index] = *offset;
    //         }
    //     }

    //     cudaError_t maskHeadSlice(cudaStream_t stream, const void *logits, const void *pred, void *output,
    //                               int numMasks, int numClasses, int MaskHeight, int MaskWidth, nvinfer1::DataType type) {
    //         auto outputSize = (int64_t)numMasks * (int64_t)MaskHeight * (int64_t)MaskWidth;
    //     #ifdef VERBOSE
    //         std::vector<std::string> types {
    //             "kFLOAT", "kHALF", "kINT8", "kINT32"
    //         };
    //         LOGGER << "type: " << types[int(type)] << std::endl;
    //         std::cout << "output: " << numMasks << ", 1, " << MaskHeight << ", " << MaskWidth << std::endl;
    //         std::cout << "blockNum: " << outputSize << ", " << onnx2trt::ATenCeilDiv(outputSize, 256L) << ", " << 4096L << std::endl;
    //     #endif
    //         dim3 grid(std::min(onnx2trt::ATenCeilDiv(outputSize, (int64_t)256), (int64_t)4096));
    //         dim3 block(256);
    //         assert(type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF);
    //         if (type == nvinfer1::DataType::kFLOAT) {
    //             sliceKernel<<<grid, block, 0, stream>>>(
    //                 outputSize, static_cast<const float *>(logits), static_cast<const int *>(pred),
    //                 static_cast<float *>(output), numMasks, numClasses, MaskHeight, MaskWidth);
    //         } else {
    //             sliceKernel<<<grid, block, 0, stream>>>(
    //                 outputSize, static_cast<const half *>(logits), static_cast<const int *>(pred),
    //                 static_cast<half *>(output), numMasks, numClasses, MaskHeight, MaskWidth);
    //         }
    //         return cudaGetLastError();
    //     }

    //     template<typename T>
    //     struct toInt {
    //         const T epsilon_;

    //         explicit toInt(T epsilon) : epsilon_(epsilon) {}

    //         __device__ int operator()(const T &input) const {
    //             return (int)(input + epsilon_);
    //         }
    //     };

    //     template<typename D, typename T>
    //     struct to {
    //         __device__ T operator()(const D &input) const {
    //             return static_cast<T>(input);
    //         }
    //     };

    //     cudaError_t cudaCast(cudaStream_t stream, const void *input, void *output,
    //         nvinfer1::Dims dims, nvinfer1::DataType type, nvinfer1::DataType castType) {
    //     #ifdef VERBOSE
    //         std::vector<std::string> types {
    //             "kFLOAT", "kHALF", "kINT8", "kINT32"
    //         };
    //         LOGGER << "type: " << types[int(type)] << ", castType: " << types[int(castType)]
    //                << ", dimension: " << dims.nbDims << ", shape (";
    //     #endif
    //         int64_t outputSize = 1;
    //         for (int i = 0; i < dims.nbDims; i++) {
    //             outputSize *= (int64_t)dims.d[i];
    //     #ifdef VERBOSE
    //             std::cout << dims.d[i] << ", ";
    //     #endif
    //         }
    //     #ifdef VERBOSE
    //         std::cout << ")" << std::endl;
    //         LOGGER << "blockNum: " << outputSize << " " << onnx2trt::ATenCeilDiv(outputSize, 256L) << " " << 4096L << std::endl;
    //     #endif
    //         if (type == castType) {
    //             return cudaErrorInvalidValue;
    //         }
    //         assert(type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF);
    //         switch (castType) {
    //             case nvinfer1::DataType::kINT8:
    //                 return cudaErrorInvalidValue;
    //             case nvinfer1::DataType::kINT32:
    //                 if (type == nvinfer1::DataType::kFLOAT) {
    //                     thrust::device_ptr<const float> input_ptr = thrust::device_pointer_cast(static_cast<const float *>(input));
    //                     thrust::device_ptr<int> output_ptr = thrust::device_pointer_cast(static_cast<int *>(output));
    //                     thrust::transform(input_ptr, input_ptr + outputSize, output_ptr, toInt<float>(0.1));
    //                 } else {
    //                     thrust::device_ptr<const half> input_ptr = thrust::device_pointer_cast(static_cast<const half *>(input));
    //                     thrust::device_ptr<int> output_ptr = thrust::device_pointer_cast(static_cast<int *>(output));
    //                     thrust::transform(input_ptr, input_ptr + outputSize, output_ptr, toInt<half>(0.1));
    //                 }
    //                 break;
    //             case nvinfer1::DataType::kHALF:
    //                 if (type == nvinfer1::DataType::kFLOAT) {
    //                     thrust::device_ptr<const float> input_ptr = thrust::device_pointer_cast(static_cast<const float *>(input));
    //                     thrust::device_ptr<half> output_ptr = thrust::device_pointer_cast(static_cast<half *>(output));
    //                     thrust::transform(input_ptr, input_ptr + outputSize, output_ptr, to<float, half>());
    //                 }
    //             case nvinfer1::DataType::kFLOAT:
    //                 if (type == nvinfer1::DataType::kHALF) {
    //                     thrust::device_ptr<const half> input_ptr = thrust::device_pointer_cast(static_cast<const half *>(input));
    //                     thrust::device_ptr<float> output_ptr = thrust::device_pointer_cast(static_cast<float *>(output));
    //                     thrust::transform(input_ptr, input_ptr + outputSize, output_ptr, to<half, float>());
    //                 }
    //         }
    //         return cudaGetLastError();
    //     }

    //     struct modulus {
    //         const int denominator;

    //         modulus(int _denominator) : denominator(_denominator) {}

    //         __device__ int operator()(const int &input) const {
    //             return input % denominator;
    //         }
    //     };

    //     cudaError_t fmod(cudaStream_t stream, const int *inputs, int *outputs, int length, int denominator) {
    //         thrust::device_ptr<const int> inputs_ptr = thrust::device_pointer_cast(inputs);
    //         thrust::device_ptr<int> outputs_ptr = thrust::device_pointer_cast(outputs);
    //         thrust::transform(inputs_ptr, inputs_ptr + length, outputs_ptr, modulus(denominator));
    //         return cudaGetLastError();
    //     }
    // }
}
