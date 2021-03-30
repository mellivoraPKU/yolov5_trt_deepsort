//
// Created by fagangjin on 25/3/2020.
//

#include <cuda_runtime_api.h>
#include "./fcosPostprocess.h"
#include "basicOps.cuh"

namespace fcos {

__device__
void sigmoid(int n, float *in, float *out) {
  for (int i = 0; i < n; i++) {
    out[i] = 1. / (1 + exp(-in[i]));
  }
}

__device__ float Logist(float data){ return 1./(1. + exp(-data)); }


__device__
void maskOnLogits(float threshold, int n, float *in, float *out) {
  for (int i = 0; i < n; i++) {
    out[i] = in[i] > threshold ? 1.f : 0.f;
  }
}

__global__
void FCOSforward_kernel(float *logits,
    float *centerness,
                        float *bbox_reg,
                        float *output,
                        const int w,
                        const int h,
                        const int channel,
                        const int stride,
                        float scale_x,
                        float scale_y,
                        float pre_nms_thresh,
                        float nms_thresh,
                        bool thresh_with_ctr,
                        int pre_nms_top_n,
                        int post_nms_top_n) {

//  int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= w * h) return;

  float max=-1;
  int max_index =0;

  for (int cls = 0; cls < channel; ++cls) {
    int objIndex = w * h * cls + idx;
    float objProb = Logist(logits[objIndex]);
    float ctrness = Logist(centerness[idx]);
    float objProbMasked = objProb * ctrness;

//    printf("%f  %f ", logits[objIndex], objProb);
    // objProb is too small to detect?
    if (objProb > 0.5) {
      printf("objProb: %f, logits: %f, centerness: %f\n", objProbMasked, objProb, ctrness);
    }

//    if (ctrness > 0.4) {
//      printf("centerness: %f, idx: %d, cls: %d\n", ctrness, idx, cls);
//    }

    if (objProbMasked > pre_nms_thresh) {
      float x1 = ((idx % w) * stride + stride / 2 - bbox_reg[idx + 0 * w * h] * stride) ;
      float y1 = ((idx / w) * stride + stride / 2 - bbox_reg[idx + 1 * w * h] * stride) ;
      float x2 = ((idx % w) * stride + stride / 2 + bbox_reg[idx + 2 * w * h] * stride) ;
      float y2 = ((idx / w) * stride + stride / 2 + bbox_reg[idx + 3 * w * h] * stride) ;

      bool valid = (x1>=0 && x2-x1 < w && x2-x1 >0 && y1 >=0 && y2-y1 <h && y2-y1>0 );
      printf("box reg: %f %f %f %f\n", bbox_reg[idx + 0 * w * h], bbox_reg[idx + 1 * w * h], bbox_reg[idx + 2 * w * h], bbox_reg[idx + 3 * w * h]);
      printf("%f %f %f %f %d\n", x1, y1, x2, y2, valid);

      float val = valid ? objProb: -1;
      max_index = (val > max) ? idx : max_index;
      max = (val > max ) ?  val: max;

      if (max_index == objIndex) {
        // get box
        int resCount = (int) atomicAdd(output, 1);
        //printf("%d",resCount);
        char *data = (char *) output + sizeof(float) + resCount * sizeof(thor::Detection);
        thor::Detection *det = (thor::Detection *) (data);

        det->bbox.x1 = x1 * scale_x;
        det->bbox.y1 = y1 * scale_y;
        det->bbox.x2 = x2 * scale_x;
        det->bbox.y2 = y2 * scale_y;
        det->classId = cls;
        det->prob = objProb;
        printf("idx: %d, objIndex: %d, prob: %f\n", idx, objIndex, objProb);
        printf("box: x1: %f, y1: %f, x2: %f, y2: %f \n", x1, y1, x2, y2);
      }


    }
  }

}

// since buffer directly from engine is void, so there is void*
void FCOS_forward_gpu(float *buffer1,
                      float *buffer2,
                      float *buffer3,
                      float *buffer4,
                      float *buffer5,
                      float *buffer6,
                      float *buffer7,
                      float *buffer8,
                      float *buffer9,
                      float *buffer10,
                      float *buffer11,
                      float *buffer12,
                      vector<thor::Detection> *output,
                      vector<int> strides,
                      const int n_classes,
                      const int input_w,
                      const int input_h,
                      float scale_x,
                      float scale_y,
                      float pre_nms_thresh,
                      float nms_thresh,
                      bool thresh_with_ctr,
                      int pre_nms_top_n,
                      int post_nms_top_n) {

  int channel, height, width, stride;
  channel = n_classes;

  auto p3_logits = buffer1;
  auto p3_centerness = buffer2;
  auto p3_bbox_reg = buffer3;
  auto p3_topfeats = buffer4;


  for (int l = 0; l < 3; l++) {
    stride = strides[l];
    height = input_h / stride;
    width = input_w / stride;


    float * cudaOutputBuffer;
//    cudaOutputBuffer = safeCudaMalloc(100 * channel*width*height* sizeof(float));
//    CUDA_CHECK(cudaMemset(cudaOutputBuffer, 0, sizeof(float)));
    cudaMallocManaged(&cudaOutputBuffer, 1000*width*height* sizeof(float));


    std::vector<thor::Detection> one_batch_result;
    one_batch_result.clear();

    switch (l) {
      case 0: {
        uint num = height * width;
        FCOSforward_kernel << < height, width >> > (
            buffer1, buffer2, buffer3,
                cudaOutputBuffer, width, height, channel, stride,
                scale_x,
                scale_y,
                pre_nms_thresh,
                nms_thresh,
                thresh_with_ctr,
                pre_nms_top_n,
                post_nms_top_n
        );
        cudaDeviceSynchronize();
        // make it back to
        int num_det = static_cast<int>(cudaOutputBuffer[0]);
        printf("\ncudaOutputBuffer num_det: %d\n", num_det);
        one_batch_result.resize(num_det);
        memcpy(one_batch_result.data(), &cudaOutputBuffer[1], num_det * sizeof(thor::Detection));
        printf("l: %d, one batch res: %d\n", l, one_batch_result.size());
        output->insert(output->end(), one_batch_result.begin(), one_batch_result.end());
        break;
      }
      case 1: {
        FCOSforward_kernel << < height, width >> > (
            buffer5, buffer6, buffer7,
                cudaOutputBuffer, width, height, channel, stride,
                scale_x,
                scale_y,
                pre_nms_thresh,
                nms_thresh,
                thresh_with_ctr,
                pre_nms_top_n,
                post_nms_top_n
        );
        cudaDeviceSynchronize();
        // make it back to
        int num_det = static_cast<int>(cudaOutputBuffer[0]);
        printf("\ncudaOutputBuffer num_det: %d\n", num_det);
        std::vector<thor::Detection> one_batch_result;
        one_batch_result.resize(num_det);
        memcpy(one_batch_result.data(), &cudaOutputBuffer[1], num_det * sizeof(thor::Detection));
        printf("l: %d, one batch res: %d\n", l, one_batch_result.size());
        output->insert(output->end(), one_batch_result.begin(), one_batch_result.end());
        break;
      }
      case 2: {
        FCOSforward_kernel << < height, width >> > (
            buffer9, buffer10, buffer11,
                cudaOutputBuffer, width, height, channel, stride,
                scale_x,
                scale_y,
                pre_nms_thresh,
                nms_thresh,
                thresh_with_ctr,
                pre_nms_top_n,
                post_nms_top_n
        );
        cudaDeviceSynchronize();
        // make it back to
        int num_det = static_cast<int>(cudaOutputBuffer[0]);
        printf("\ncudaOutputBuffer num_det: %d\n", num_det);
        std::vector<thor::Detection> one_batch_result;
        one_batch_result.resize(num_det);
        memcpy(one_batch_result.data(), &cudaOutputBuffer[1], num_det * sizeof(thor::Detection));
        printf("l: %d, one batch res: %d\n", l, one_batch_result.size());
        output->insert(output->end(), one_batch_result.begin(), one_batch_result.end());
        break;
      }
    }
  }

  // concat for final result?

}

}


