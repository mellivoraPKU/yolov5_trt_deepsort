/*
 * Copyright (c) 2020 FJ. All rights reserved.
 * Written by Fagang Jin
 */

#include "../include/trt_engine.h"

namespace tensorrt {

static tensorrt::Logger gLogger;

TensorRTEngine::TensorRTEngine(const std::string& engineFile)
    : mTrtContext(nullptr),
      mTrtEngine(nullptr),
      mTrtRunTime(nullptr),
      mTrtInputCount(0),
      mTrtIterationTime(0) {
  initLibNvInferPlugins(&gLogger, "");

  _gieModelStream.seekg(0, _gieModelStream.beg);
  std::ifstream serialize_iutput_stream(engineFile,
                                        std::ios::in | std::ios::binary);
  if (!serialize_iutput_stream) {
    std::cerr << "cannot find serialize file" << std::endl;
  }
  serialize_iutput_stream.seekg(0);

  _gieModelStream << serialize_iutput_stream.rdbuf();
  _gieModelStream.seekg(0, std::ios::end);
  const int modelSize = _gieModelStream.tellg();
  _gieModelStream.seekg(0, std::ios::beg);
  void* modelMem = malloc(modelSize);
  _gieModelStream.read((char*)modelMem, modelSize);

  IBuilder* builder = createInferBuilder(gLogger);
  builder->destroy();
  // todo: release runtime avoid memory leak
  IRuntime* runtime = createInferRuntime(gLogger);
  mTrtEngine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);

  // log engine information here
  //  CheckEngine(mTrtEngine);
  //  cout << "check engine done.\n";
  std::free(modelMem);
  assert(mTrtEngine != nullptr);
}

// void TensorRTEngine::InitEngine() {
//   mTrtBatchSize = mTrtEngine->getMaxBatchSize();
//   mTrtContext = mTrtEngine->createExecutionContext();
//   assert(mTrtContext != nullptr);
//   mTrtContext->setProfiler(&mTrtProfiler);

//   // Input and output buffer pointers that we pass to the engine - the engine
//   // requires exactly IEngine::getNbBindings()
//   int nbBindings = mTrtEngine->getNbBindings();
//   cout << "we get the engine bindings: " << nbBindings << endl;

//   mTrtCudaBuffer.resize(nbBindings);
//   mTrtBindBufferSize.resize(nbBindings);
//   for (int i = 0; i < nbBindings; ++i) {
//     Dims dims = mTrtEngine->getBindingDimensions(i);
//     DataType dtype = mTrtEngine->getBindingDataType(i);
//     int64_t totalSize = volume(dims) * mTrtBatchSize * getElementSize(dtype);
//     mTrtBindBufferSize[i] = totalSize;
//     mTrtCudaBuffer[i] = safeCudaMalloc(totalSize);
//     if (mTrtEngine->bindingIsInput(i)) mTrtInputCount++;
//   }
//   CUDA_CHECK(cudaStreamCreate(&mTrtCudaStream));
// }

}  // namespace tensorrt
