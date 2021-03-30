/******************************************************************************
 * Copyright 2019 AutoX. All Rights Reserved.
 *****************************************************************************/
#include "cuda_common.h"

namespace trtonnx2 {
namespace cuda {
namespace kernel {

template <typename T>
__global__ void CalculateArgMax(T const *const input, size_t size,
                                unsigned char num_classes, size_t batch_offset,
                                unsigned char *argmax) {
  const size_t local_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (local_index >= size) return;
  const size_t global_index = local_index + batch_offset;
  argmax[local_index] = 0;
  T maxval = input[global_index];
  for (unsigned char cls = 1; cls < num_classes; ++cls) {
    const size_t offset = cls * size + global_index;
    if (input[offset] > maxval) {
      maxval = input[offset];
      argmax[local_index] = cls;
    }
  }
}

template <typename T>
__global__ void ReshapeToCHW(T const *const hwc_image, const int width,
                             const int height, const int channels,
                             T *chw_image) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= height || col >= width) return;

  for (int chan = 0; chan < channels; ++chan) {
    const int input_offset = (row * width + col) * channels + chan;
    const int output_offset = (chan * height + row) * width + col;
    chw_image[output_offset] = hwc_image[input_offset];
  }
}

}  // namespace kernel


dim3 cudaGridSize(uint n) {
  uint k = (n - 1) / kBlockSize + 1;
  uint x = k;
  uint y = 1;
  if (x > 65535) {
    x = ceil(sqrt(x));
    y = (n - 1) / (x * kBlockSize) + 1;
  }
  dim3 d = {x, y, 1};
  return d;
}

template <typename T>
void CalculateArgMax(T const *const logits, size_t size, int num_classes,
                     size_t batch_offset, cudaStream_t stream,
                     unsigned char *argmax) {
  constexpr int kNumThreadsPerBlock = 512;
  const size_t num_blocks = DivideUp(size, kNumThreadsPerBlock);
  kernel::CalculateArgMax<<<num_blocks, kNumThreadsPerBlock, 0, stream>>>(
      logits, size, num_classes, batch_offset, argmax);
}
/// Instantiations
template void CalculateArgMax<float>(float const *const logits, size_t size,
                                     int num_classes, size_t batch_offset,
                                     cudaStream_t stream,
                                     unsigned char *argmax);

template <typename T>
void ReshapeToCHW(T const *const hwc_image, int width, int height, int channels,
                  cudaStream_t stream, T *chw_image) {
  const dim3 block_size(32, 32);
  const dim3 grid_size(DivideUp(width, block_size.x),
                       DivideUp(height, block_size.y));
  kernel::ReshapeToCHW<<<grid_size, block_size, 0, stream>>>(
      hwc_image, width, height, channels, chw_image);
}

/// Instantiations
template void ReshapeToCHW(float const *const hwc_image, int width, int height,
                           int channels, cudaStream_t stream, float *chw_image);

}  // namespace cuda
}  // namespace trtonnx2
