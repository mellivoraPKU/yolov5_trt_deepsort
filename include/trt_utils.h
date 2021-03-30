/**
 *
 *
 * Some common tensorrt functions
 *
 * */

#ifndef __TRT_UTILS_H_
#define __TRT_UTILS_H_

#include <cudnn.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace std;
using namespace nvinfer1;

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                 \
  {                                                                         \
    cudaError_t error_code = callstr;                                       \
    if (error_code != cudaSuccess) {                                        \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" \
                << __LINE__;                                                \
      assert(0);                                                            \
    }                                                                       \
  }
#endif

void CheckEngine(ICudaEngine *&engine) {
  cout << "=> checking engine....\n";
  int bds = engine->getNbBindings();
  cout << "=> engine maxBatchSize: " << engine->getMaxBatchSize() << std::endl;
  cout << "=> engine NbBindings: " << bds << std::endl;
  for (int kI = 0; kI < bds; ++kI) {
    cout << "    => BindingName at: " << kI << "=" << engine->getBindingName(kI)
         << " Dims=" << engine->getBindingDimensions(kI).nbDims << " shape: ";
    for (int kJ = 0; kJ < engine->getBindingDimensions(kI).nbDims; ++kJ) {
      cout << engine->getBindingDimensions(kI).d[kJ] << ",";
    }
    cout << endl;
  }
}

// safeMalloc
inline void* safeCudaMalloc(size_t memSize)
{
  void* deviceMem;
  CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr)
  {
    std::cerr << "Out of memory" << std::endl;
    exit(1);
  }
  return deviceMem;
}

// size shortcut
constexpr long long int operator "" _GB(long long unsigned int val) {
  return val * (1 << 30);
}
constexpr long long int operator "" _MB(long long unsigned int val) {
  return val * (1 << 20);
}
constexpr long long int operator "" _KB(long long unsigned int val) {
  return val * (1 << 10);
}


// quick calculate sizes
inline int64_t volume(const nvinfer1::Dims& d)
{
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
  switch (t)
  {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}


// log using in nvinfer
namespace tensorrt {
class Logger : public nvinfer1::ILogger {
 public:
  Logger() : Logger(Severity::kWARNING) {}
  Logger(Severity severity) : reportableSeverity(severity) {}

  void log(Severity severity, const char *msg) override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }
  Severity reportableSeverity{Severity::kWARNING};
};
}  // namespace trt

#endif