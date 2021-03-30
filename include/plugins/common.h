/*
 * Copyright (c) 2019,. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

#include <dirent.h>

#include <fstream>
#include <numeric>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#ifndef WARNING
#define WARNING
#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__)
#define LOGGER                                                             \
  std::cout << "[" << __func__ << "() " << __FILENAME__ << ":" << __LINE__ \
            << "] "
#endif

#define LOG_SUPPORTS_FORMAT_COMBINATION()                                      \
  std::vector<std::string> formats = {"kLINEAR", "kCHW2",  "kHWC8",            \
                                      "kCHW4",   "kCHW16", "kCHW32"};          \
  std::vector<std::string> types{"kFLOAT", "kHALF", "kINT8", "kINT32"};        \
  std::cout << "pos_" << pos << ": format " << formats[int(inOut[pos].format)] \
            << ", type " << types[int(inOut[pos].type)] << ", shape (";        \
  for (int j = 0; j < inOut[pos].dims.nbDims; j++) {                           \
    std::cout << inOut[pos].dims.d[j] << ", ";                                 \
  }                                                                            \
  std::cout << ")" << std::endl;

#define LOG_INPUT_DIMENSIONS()                         \
  for (int i = 0; i < nbInputs; i++) {                 \
    std::cout << "input_shape_" << i << ": (";         \
    for (int j = 0; j < inputs[i].nbDims; j++) {       \
      int constant = -1;                               \
      if (inputs[i].d[j]->isConstant()) {              \
        constant = inputs[i].d[j]->getConstantValue(); \
      }                                                \
      std::cout << constant << ", ";                   \
    }                                                  \
    std::cout << ")" << std::endl;                     \
  }

#define LOG_ENQUEUE(in_num, out_num)                                          \
  std::vector<std::string> formats = {"kLINEAR", "kCHW2",  "kHWC8",           \
                                      "kCHW4",   "kCHW16", "kCHW32"};         \
  std::vector<std::string> types{"kFLOAT", "kHALF", "kINT8", "kINT32"};       \
  int nbInputs = in_num;                                                      \
  for (int i = 0; i < nbInputs; i++) {                                        \
    std::cout << "inputDesc_" << i << ": " << inputDesc[i].dims.nbDims << " " \
              << types[int(inputDesc[i].type)] << " "                         \
              << formats[int(inputDesc[i].format)] << " "                     \
              << inputDesc[i].scale << " Dimension: (";                       \
    for (int j = 0; j < inputDesc[i].dims.nbDims; j++) {                      \
      std::cout << inputDesc[i].dims.d[j] << ", ";                            \
    }                                                                         \
    std::cout << ")" << std::endl;                                            \
  }                                                                           \
  int nbOutputs = out_num;                                                    \
  for (int i = 0; i < nbOutputs; i++) {                                       \
    std::cout << "outputDesc_" << i << ": " << outputDesc[i].dims.nbDims      \
              << " " << types[int(outputDesc[i].type)] << " "                 \
              << formats[int(outputDesc[i].format)] << " "                    \
              << outputDesc[i].scale << " Dimension: (";                      \
    for (int j = 0; j < outputDesc[i].dims.nbDims; j++) {                     \
      std::cout << outputDesc[i].dims.d[j] << ", ";                           \
    }                                                                         \
    std::cout << ")" << std::endl;                                            \
  }



namespace nvinfer1 {
namespace plugin {
class BasePluginCreator : public IPluginCreator {
 public:
  void setPluginNamespace(const char* libNamespace) override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const override { return mNamespace.c_str(); }

 protected:
  std::string mNamespace;
};
}  // namespace plugin
}  // namespace nvinfer1

#endif  // TENSORRT_COMMON_H
