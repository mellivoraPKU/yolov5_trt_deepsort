//
// Created by xuyufeng1 on 2021/3/25.
//

#ifndef TRACKCOUNTER_TRACK_DEEPSORT_H
#define TRACKCOUNTER_TRACK_DEEPSORT_H
#include<opencv2/opencv.hpp>
#include "KalmanFilter/tracker.h"
#include <vector>
#include <map>
#include <set>

#define NN_BUDGET 100
#define MAX_COSINE_DISTANCE 0.2

class track_deepsort {
public:
    track_deepsort();
    ~track_deepsort() = default;
    void run(DETECTIONS& detections);
    cv::Mat display(cv::Mat frame);

private:
    std::vector<Track> nextReleaseTrack;
    std::map<int, std::set<std::string>> timestamplist;
    tracker mytracker;
   // std::map<int, std::vector<cv::Point>> track_map_vec;

    int missed_num;
};


#endif //TRACKCOUNTER_TRACK_DEEPSORT_H
