//
// Created by Dongqi Xu on 4/3/2020.
//
#ifndef __TRT_PROFILER_H_
#define __TRT_PROFILER_H_

#include "NvInfer.h"
#include "NvOnnxParser.h"

class Profiler : public nvinfer1::IProfiler {
public:
    Profiler(float threshold)
        : engine_time_(0.0)
        , selected_time_(0.0)
        , threshold_(threshold) {}

    void reportEngineTime() {
        std::cout << std::fixed << std::setprecision(4) << "Engine time: " << engine_time_
                  << " ms, selected time: " << selected_time_ << " ms" << std::endl;
    }

    void reportLayerTime (const char *layerName, float ms) {
        engine_time_ += ms;
        if (ms > threshold_) {
            std::cout << std::fixed << std::setprecision(4) << ms << ": " << layerName << std::endl;
            selected_time_ += ms;
        }
    }

private:
    float engine_time_;
    float selected_time_;
    float threshold_;
};

#endif // __TRT_PROFILER_H_
