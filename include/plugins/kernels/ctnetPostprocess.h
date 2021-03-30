//
// Created by fagangjin on 25/3/2020.
//

#ifndef CTDET_TRT_CTDETLAYER_H
#define CTDET_TRT_CTDETLAYER_H

#include "thor/structures.h"

using namespace thor;

namespace ctnet {

void CTdetforward_gpu(const float *hm, const float *reg, const float *wh,
                      float *output, const int w, const int h,
                      const int classes, const int kernerl_size,
                      const float visthresh);
void CTfaceforward_gpu(const float *hm, const float *wh, const float *reg,
                       const float *landmarks, float *output, const int w,
                       const int h, const int classes, const int kernerl_size,
                       const float visthresh);
}  // namespace ctnet

#endif  // CTDET_TRT_CTDETLAYER_H
