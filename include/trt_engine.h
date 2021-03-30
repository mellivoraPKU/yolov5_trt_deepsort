#ifndef __TRT_NET_H_
#define __TRT_NET_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>

#include <cudnn.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "./trt_utils.h"


using namespace std;

namespace tensorrt {

class TensorRTEngine {
 public:
  explicit TensorRTEngine(const std::string &engineFile);

  ~TensorRTEngine() {
    // Release the stream and the buffers
//    cudaStreamSynchronize(mTrtCudaStream);
//    cudaStreamDestroy(mTrtCudaStream);
//    for (auto &item : mTrtCudaBuffer) cudaFree(item);
//    if (!mTrtRunTime) mTrtRunTime->destroy();
//    if (!mTrtContext) mTrtContext->destroy();
//    if (!mTrtEngine) mTrtEngine->destroy();
    std::cout << "engine shutdown.\n";
  };

  inline size_t getInputSize() {
    return std::accumulate(mTrtBindBufferSize.begin(),
                           mTrtBindBufferSize.begin() + mTrtInputCount, 0);
  };
  inline size_t getOutputSize() {
    return std::accumulate(mTrtBindBufferSize.begin() + mTrtInputCount,
                           mTrtBindBufferSize.end(), 0);
  };
  inline int getBatchSize() { return mTrtBatchSize; };

  // get engine
  nvinfer1::ICudaEngine *getEngine() {
    return mTrtEngine;
  }

 public:
  int input_width;
  int input_height;
  int input_channel;

 private:
//   void InitEngine();

  nvinfer1::IExecutionContext *mTrtContext;
  nvinfer1::ICudaEngine *mTrtEngine;
  nvinfer1::IRuntime *mTrtRunTime;
  cudaStream_t mTrtCudaStream;
  int mTrtRunMode;

  std::vector<void *> mTrtCudaBuffer;
  std::vector<int64_t> mTrtBindBufferSize;
  int mTrtInputCount;
  int mTrtIterationTime;
  int mTrtBatchSize;

  std::stringstream _gieModelStream;
};
}  // namespace tensorrt

#endif  //__TRT_NET_H_
