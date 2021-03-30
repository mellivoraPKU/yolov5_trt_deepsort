
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
 #include "plugins/kernels/mask_select.h"




namespace trtonnx2 {

    __global__ void panopticfcn_forward_kernel(const float* pred_inst, const int* classes,
        const float* scores, float* output,  const int candidates_n, const int h, const int w, const float threshold) {
        int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
        if (idx >= candidates_n) return;
    
        float objProb = scores[idx];
        
        if (objProb > threshold) {
            int cls = classes[idx];
            // printf("%f %d %d | \n", objProb, idx, classes[idx]);
            int resCount = (int) atomicAdd(output, 1);
            //printf("%d",resCount);
            char *data = (char *) output + sizeof(float) + resCount * sizeof(InstanceSegmentation);
            InstanceSegmentation *insg = (InstanceSegmentation *) (data);

            insg->score = objProb;
            insg->cid = cls;
            insg->idx = idx;
        }   
    }

    __global__ void panopticfcn_forward_kernel2(const int* classes,
        const float* scores, int* resCount, InstanceSegmentation* insts,  const int candidates_n, const int h, const int w, const float threshold) {
        // int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
        int idx = threadIdx.x + blockIdx.x;
        if (idx >= candidates_n) return;
    
        float objProb = scores[idx];
        // printf("%f %d %d | \n", objProb, idx, classes[idx]);

        if (objProb > threshold) {
            int cls = classes[idx];
            int resCountIdx = (int) atomicAdd(resCount, 1);
                    // printf("%f %d %d | \n", objProb, idx, classes[idx]);

            insts[resCountIdx].score = objProb;
            insts[resCountIdx].cid = cls;
            insts[resCountIdx].idx = idx;
        }   
    }
    
    
    void panopticfcn_forward_gpu(const float* pred_inst, const int* classes,
        const float* scores, float* output, const int candidates_n, const int h, const int w, const float threshold){
            dim3 gridSize = cudaGridSize(candidates_n);
            cout << "candidates n: " << candidates_n << " gridSize: " << gridSize.x << " " << gridSize.y << " thresh: " << \
            threshold << endl;
        panopticfcn_forward_kernel<<<cudaGridSize(candidates_n), kBlockSize>>>(pred_inst, classes, scores, output,
            candidates_n, h, w, threshold);
    }

    void panopticfcn_forward_gpu(const int* classes,
        const float* scores, float* output, const int candidates_n, const int h, const int w, const float threshold){
            dim3 gridSize = cudaGridSize(candidates_n);
            cout << "candidates n: " << candidates_n << " gridSize: " << gridSize.x << " " << gridSize.y << " thresh: " << \
            threshold << endl;

        InstanceSegmentation* insts;
        cudaMalloc(&insts, sizeof(InstanceSegmentation)*candidates_n);
        int* resCount;
        cudaMalloc(&resCount, sizeof(int));

        panopticfcn_forward_kernel2<<<cudaGridSize(candidates_n), kBlockSize>>>(classes, scores, resCount, insts,
            candidates_n, h, w, threshold);
        // cudaStreamSynchronize();
        cudaDeviceSynchronize();
    }

    void panopticfcn_forward_gpu(const int* classes, const float* scores,
        int* resCount, InstanceSegmentation* insts,
        const int candidates_n, const int h, const int w,
        const float threshold) {
        
        int ts = trtonnx2::cuda::DivideUp(candidates_n, kBlockSize);
        dim3 threadSize(ts);
        // panopticfcn_forward_kernel2<<<cudaGridSize(candidates_n), kBlockSize>>>(classes, scores, resCount, insts,
        //     candidates_n, h, w, threshold);
        panopticfcn_forward_kernel2<<<ts, kBlockSize>>>(classes, scores, resCount, insts,
            candidates_n, h, w, threshold);
        cudaDeviceSynchronize();
        }
}