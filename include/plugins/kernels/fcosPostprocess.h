//
// Created by fagangjin on 25/3/2020.
//

#ifndef FCOS_POSTPROCESS_H
#define FCOS_POSTPROCESS_H

#include <host_defines.h>
#include "thor/structures.h"

#include <thor/proto/det.pb.h>

namespace fcos {

void FCOS_forward_gpu(
    float *buffer1, float *buffer2, float *buffer3, float *buffer4,
    float *buffer5, float *buffer6, float *buffer7, float *buffer8,
    float *buffer9, float *buffer10, float *buffer11, float *buffer12,
    vector<thor::Detection> *output,
    vector<int> strides,
    const int n_classes,
    const int input_w,
    const int input_h,
    float scale_x, float scale_y,
    float pre_nms_thresh = 0.5f, float nms_thresh = 0.6f,
    bool thresh_with_ctr = false, int pre_nms_top_n = 1000,
    int post_nms_top_n = 100
);

}



#endif //FCOS_POSTPROCESS_H
