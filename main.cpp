/*
 * Copyright (c) 2020 MANAAI.
 *
 * This file is part of onnx_deploy
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
#include <assert.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <opencv2/opencv.hpp>

#include <thor/vis.h>
#include "thor/timer.h"
#include "thor/logging.h"
#include "thor/os.h"
#include "thor/structures.h"
#include "thor/image.h"

#include "trt_engine.h"
#include "trt_utils.h"
#include "alg_track.h"
#include <string>
#include"DeepSORT/track_deepsort.h"

/**
 *
 *
 * Inference on YoloV5 from Ultralytics inc.
 * We provided in house onnx export repo which able to
 * export ONNX model this file using
 *
 * fist convert that ONNX file into trt engine
 * then edit the CLASSES variable for inference.
 *
 *
 *
 */
#define DEEPSORT
using namespace thor::log;
using namespace nvinfer1;
#define CHECK(_c) CUDA_CHECK(_c)
// our yolov4 using 544 as input
//static const int INPUT_H = 480;
//static const int INPUT_W = 800;
static const int INPUT_H = 640;
static const int INPUT_W = 640;
//static const int INPUT_H = 768;
//static const int INPUT_W = 1280;

// BGR order
static const float kMeans[3] = {0., 0., 0.};
static const float kStds[3] = {255., 255., 255.};
//static const int kOutputCandidates = 23625;
static const int kOutputCandidates = 20250;
// model config
//const static vector<string> CLASSES = {"car", "Van", "Truck", "Bus", "Tanker", "Tricycle", "Heterotype", "Trame", "bicycle", "Eletric Bicycle", "Others"}; //car 1
//const static vector<string> CLASSES = {"car", "Van", "Truck", "Bus", "Tanker", "Tricycle", "Heterotype", "Trame", "Bicycle", "Eletric Bicycle","Cyclist","Pedestrain"};
const static vector<string> CLASSES = {"Pedestrain"};

cv::Mat preprocess_img(cv::Mat &img) {
  int w, h, x, y;
  float r_w = INPUT_W / (img.cols * 1.0);
  float r_h = INPUT_H / (img.rows * 1.0);
  if (r_h > r_w) {
    w = INPUT_W;
    h = r_w * img.rows;
    x = 0;
    y = (INPUT_H - h) / 2;
  } else {
    w = r_h * img.cols;
    h = INPUT_H;
    x = (INPUT_W - w) / 2;
    y = 0;
  }
  cv::Mat res;

  cv::resize(img, res, cv::Point(int(w), int(h)), cv::INTER_CUBIC);
  int top, bottom, left, right;
  top = int(round(y - 0.1));
  bottom = int(round(y + 0.1));
  left = int(round(x - 0.1));
  right = int(round(x + 0.1));
  cv::copyMakeBorder(res, res, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
 // cv::imwrite("121.jpg",res);
  return res;
}

bool CompareBBox(const thor::Detection &a, const thor::Detection &b) {
  return a.prob > b.prob;
}

std::vector<thor::Detection> nms(std::vector<thor::Detection> &bboxes,
                                 float threshold) {
  std::vector<thor::Detection> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), CompareBBox);
  int32_t select_idx = 0;
  int32_t num_bbox = static_cast<int32_t>(bboxes.size());
  std::vector<int32_t> mask_merged(num_bbox, 0);
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }
    bboxes_nms.push_back(bboxes[select_idx]);
    mask_merged[select_idx] = 1;
    thor::BoxSimple select_bbox = bboxes[select_idx].bbox;
    float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) *
        (select_bbox.y2 - select_bbox.y1 + 1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int32_t i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) continue;

      thor::BoxSimple &bbox_i = bboxes[i].bbox;
      float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
      float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
      float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;  //<- float 型不加1
      float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
      if (w <= 0 || h <= 0) continue;

      float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) *
          (bbox_i.y2 - bbox_i.y1 + 1));
      float area_intersect = w * h;
      if (static_cast<float>(area_intersect) /
          (area1 + area2 - area_intersect) >
          threshold) {
        mask_merged[i] = 1;
      }
    }
  }
  return bboxes_nms;
}

vector<thor::Detection> doPostProcess(float *out,
                                      const cv::Size ori_size,
                                      float nms_threshold,
                                      bool resize_shortest = false) {
  // out: 4+1+1+classes = 7
  // bbox, conf, idx, prob
  vector<thor::Detection> dets;
  int ori_h = ori_size.height;
  int ori_w = ori_size.width;
  float r_w = INPUT_W / (ori_w * 1.0);
  float r_h = INPUT_H / (ori_h * 1.0);

  float gain = min(r_w, r_h);  // gain  = old / new
  float pad_x = (INPUT_W - ori_w * gain) / 2;
  float pad_y = (INPUT_H - ori_h * gain) / 2;

  for (int i = 0; i < kOutputCandidates; ++i) {
    // first column is background
    float conf_i = out[i * (6 + CLASSES.size()) + 4];
    int label_i = (int) out[i * (6 + CLASSES.size()) + 5];
    float prob_i = out[i * (6 + CLASSES.size()) + 6 + label_i];
    conf_i *= prob_i;
    if (conf_i >= 0.4) {

      // model outputs in cx,cy,w,h format
      float cx = out[i * (6 + CLASSES.size()) + 0] - pad_x;
      float cy = out[i * (6 + CLASSES.size()) + 1] - pad_y;
      float w = out[i * (6 + CLASSES.size()) + 2];
      float h = out[i * (6 + CLASSES.size()) + 3];
      float x1 = (cx - w / 2.f) / gain;
      float y1 = (cy - h / 2.) / gain;
      float x2 = (cx + w / 2.f) / gain;
      float y2 = (cy + h / 2.) / gain;

      // convert box coordinates back
      thor::Detection det;
      // remap the box size
      det.bbox.x1 = x1;
      det.bbox.y1 = y1;
      det.bbox.x2 = x2;
      det.bbox.y2 = y2;
      det.classId = label_i;
      det.prob = conf_i;
      dets.emplace_back(det);
    }
  }
  // do nms here
  // todo: move this to cuda kernel (then it will not need copy data to CPU again.
  dets = nms(dets, nms_threshold);
  return dets;
}

vector<thor::Detection> doInference(IExecutionContext &context, float *input, const cv::Size ori_size,
                                    int batchSize = 1, float nms_threshold = 0.4) {
  const ICudaEngine &engine = context.getEngine();
  int nbBindings = engine.getNbBindings();
  assert(engine.getNbBindings() == 2);

  void *buffers[nbBindings];
  std::vector<int64_t> bufferSize;
  bufferSize.resize(nbBindings);

  for (int kI = 0; kI < nbBindings; ++kI) {
    nvinfer1::Dims dims = engine.getBindingDimensions(kI);
    nvinfer1::DataType dt = engine.getBindingDataType(kI);
    int64_t totalSize = volume(dims) * 1 * getElementSize(dt);
    bufferSize[kI] = totalSize;
//        LOG(INFO) << "binding " << kI << " nodeName: " << engine.getBindingName(kI) << " total size: " << totalSize;
    CHECK(cudaMalloc(&buffers[kI], totalSize));
  }

  auto out = new float[bufferSize[1] / sizeof(float)];

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  CHECK(cudaMemcpyAsync(buffers[0], input, bufferSize[0], cudaMemcpyHostToDevice, stream));
  context.executeV2(&buffers[0]);

  CHECK(cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  // release the stream and the buffers
  cudaStreamDestroy(stream);
  for (int kJ = 0; kJ < nbBindings; ++kJ) {
    CHECK(cudaFree(buffers[kJ]));
  }
  vector<thor::Detection> dets = doPostProcess(out, ori_size, nms_threshold);
  return dets;
}
void get_detections(DETECTBOX box, float confidence, int type, DETECTIONS &d, int timestamp) {
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);
    tmpRow.type = type;
    tmpRow.timestamp = timestamp;
    tmpRow.shelter = false;
    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}

int run(int argc, char **argv) {
 // string engine_f = argv[1];
  string engine_f ="/home/py/code/mana/onnx-tensorrt/build/person.trt";
 // string data_f = argv[2];
  string data_f ="/home/py/Downloads/person.mp4";
  string mode = "fp32";
  string batch_f;
  track_deepsort deepsort;

//  if (argc >= 4) {
//    mode = argv[3];
//    batch_f = argv[4];
//  }

  LOG(INFO) << "loading from engine file: " << engine_f;
  MAX_algorithm::alg_track m_alg_track;
  m_alg_track.Alg_Track_Init();
  tensorrt::TensorRTEngine trtEngine(engine_f);
  ICudaEngine *engine = trtEngine.getEngine();
  CheckEngine(engine);
  IExecutionContext *context = engine->createExecutionContext();
  assert(context != nullptr);
  LOG(INFO) << "inference directly from engine.";

  if (thor::os::suffix(data_f) == "mp4") {
    cv::VideoCapture cap(data_f);
    if (!cap.isOpened()) {
      std::cout << "Error: video-stream can't be opened! \n";
      return 1;
    }

    cv::Mat frame;
//    float *data;
    static float data[1 * 3 * INPUT_H * INPUT_W];

    int ori_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int ori_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    thor::Timer timer(20);
    timer.on();
    int frame_id=0;
    LOG(INFO) << "start to inference on video file: " << data_f;
    while (1) {

      cap >> frame;
      frame_id++;
      if (frame.empty()) {
        cv::destroyAllWindows();
        cap.release();
        // destroy the engine
        context->destroy();
        engine->destroy();
        LOG(INFO) << "shut down!";
        break;
      }

      cv::Mat pr_img = preprocess_img(frame);
      for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
        data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
        data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
      }

      timer.lap();
      vector<thor::Detection> dets = doInference(*context, data, frame.size());
      double cost = timer.lap();
      LOG(INFO) << "fps: " << 1 / cost << ", cost: " << cost;
#ifdef DEEPSORT
        DETECTIONS detections;
        DETECTION_ROW OBJ;
        vector<MAX_algorithm::Obj_Info> temp_ObjVec;
        int timestamp =0;
        for (int i =0; i<dets.size();i++)
        {
           float x =dets[i].bbox.x1;
           float y =dets[i].bbox.y1;
           float w =abs(dets[i].bbox.x1 -dets[i].bbox.x2);
           float h =abs(dets[i].bbox.y1-dets[i].bbox.y2);
           int class_id =dets[i].classId;
           float conf  =dets[i].prob;
           // temp_ObjVec.push_back(obj);
           get_detections(DETECTBOX(x, y, w, h), conf, class_id, detections, timestamp);
        }
        timestamp++;
        timer.lap();
        deepsort.run(detections);
        double cost2 = timer.lap();
        LOG(INFO)  << "tracker cost: " << cost2;



#else
      vector<MAX_algorithm::Obj_Info> temp_ObjVec;
      for (int i =0; i<dets.size();i++)
      {
          MAX_algorithm::Obj_Info obj;
          obj.obj_box.x =dets[i].bbox.x1;
          obj.obj_box.y =dets[i].bbox.y1;
          obj.obj_box.width =abs(dets[i].bbox.x1 -dets[i].bbox.x2);
          obj.obj_box.height =abs(dets[i].bbox.y1-dets[i].bbox.y2);
          obj.obj_class =dets[i].classId;
          obj.obj_prob =dets[i].prob;
          temp_ObjVec.push_back(obj);
      }

      vector<MAX_algorithm::Obj_Info_Track> tTrackResult;

      m_alg_track.Alg_Track_Deploy(&frame,&temp_ObjVec,frame_id,&tTrackResult);
      for (int i =0; i <tTrackResult.size();i++)
      {
          string id =std::to_string(tTrackResult[i].obj_code);
          cv::Point point(tTrackResult[i].obj_info.obj_box.x,tTrackResult[i].obj_info.obj_box.y);
          cv::putText(frame, id, point, cv::FONT_HERSHEY_COMPLEX, 2.0,
                      cv::Scalar(0, 255, 0), 2);
      }
#endif

      // also we need log FPS here
      char fps_str[128];
      sprintf(fps_str, "FPS: %.3f", 1 / cost);
      cv::putText(frame, fps_str, cv::Point(40, 80), cv::FONT_HERSHEY_COMPLEX, 2.0,
                  cv::Scalar(0, 255, 0), 2);
//      std::free(data);

      // visualize the final detections
//            cv::Mat res = thor::vis::VisualizeDetections(frame, dets, CLASSES, &colors, 2,
//                    0.38, true);
      cv::Mat res = thor::vis::VisualizeDetections(frame, dets, CLASSES, nullptr);
      res=deepsort.display(res);
      cv::imshow("YoloV5 TensorRT", res);
//      cv::imshow("YoloV5 TensorRT rstg", pr_img);
      cv::waitKey(1);
    }

  } else if (thor::os::isdir(data_f)) {
    cv::String path(thor::os::join(data_f, "*.jpg")); //select only jpg
    vector<cv::String> fn;
    cv::glob(path, fn);
    LOG(INFO) << "found all " << fn.size() << " images under: " << data_f;
    for (auto &f: fn) {
      static float data[1 * 3 * INPUT_H * INPUT_W];
      cv::Mat frame = cv::imread(f);
      cv::Mat pr_img = preprocess_img(frame);
      for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
        data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
        data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
      }
      vector<thor::Detection> dets = doInference(*context, data, frame.size());

      // visualize the final detections
      cv::Mat res = thor::vis::VisualizeDetections(frame, dets, CLASSES, nullptr, 1,
                                                   0.38, true);

      cv::imshow("YoloV5 TensorRT", res);
      cv::waitKey(0);
    }

  } else {
    // on image
    cv::Mat frame = cv::imread(data_f);
    static float data[1 * 3 * INPUT_H * INPUT_W];
    cv::Mat pr_img = preprocess_img(frame);
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
      data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
      data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
      data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
    }
    vector<thor::Detection> dets = doInference(*context, data, frame.size());
    LOG(INFO) << "infer done " << dets.size();

    // visualize the final detections
    cv::Mat res = thor::vis::VisualizeDetections(frame, dets, CLASSES, nullptr, 1, 0.38);
    LOG(INFO) << "visual";
    cv::imshow("YoloV5 TensorRT", res);
    cv::waitKey(0);
  }

  return 0;
}

int main(int argc, char **argv) {
  // trt_file data_file
  run(argc, argv);
  return 0;
}


/*
 *  vector<Obj_Info_Track> tTrackResult;  //body(class==0)
    vector<Obj_Info_Track> tHeadResult;   //head(class==1)

    if(eType == Arbitration_Follow || eType == Arbitration_Burst)
    {
        vector<Obj_Info> temp_ObjVec;   //body(class==0)
        for(int i = 0; i < detectResult->obj_num; i++)
        {
            if(detectResult->obj_info[i].obj_class == 0)
            {
                temp_ObjVec.push_back(detectResult->obj_info[i]);
            }
            else
            {//head results
                Obj_Info_Track  temp_HeadVec;
                temp_HeadVec.obj_info = detectResult->obj_info[i];
                tHeadResult.push_back(temp_HeadVec);
            }
        }
        followAritration_handle->Aritration_GetTrackResult(&srcMat, &temp_ObjVec, Frame_ID, &tTrackResult);
        if(detectResult->obj_num>0 && tTrackResult.size()>0)
        {
            tTrackResult[0].iClassIndex =  detectResult->iClassIndex;
            tTrackResult[0].fClassProb  =  detectResult->fClassProb;
        }
    }
 * */
