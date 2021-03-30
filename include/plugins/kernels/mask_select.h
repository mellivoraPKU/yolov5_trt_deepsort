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

#include <iostream>

#include "common.h"
#include "cuda_common.h"
#include "thor/structures.h"

namespace trtonnx2 {

using std::cout;
using std::endl;
using thor::InstanceSegmentation;
// using thor::dl::InstanceSegmentation;
using trtonnx2::cuda::cudaGridSize;
using trtonnx2::cuda::kBlockSize;

// select mask from N masks by score
void panopticfcn_forward_gpu(const float* pred_inst, const int* classes,
                             const float* scores, float* output,
                             const int candidates_n, const int h, const int w,
                             const float threshold);
void panopticfcn_forward_gpu(const int* classes, const float* scores,
                             float* output, const int candidates_n, const int h,
                             const int w, const float threshold);
void panopticfcn_forward_gpu(const int* classes, const float* scores,
                             int* resCount, InstanceSegmentation* insts,
                             const int candidates_n, const int h, const int w,
                             const float threshold);

}  // namespace trtonnx2
