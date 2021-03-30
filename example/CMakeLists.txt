
find_package(OpenCV  REQUIRED)

add_executable(demo_yolov5 yolov5/demo_yolov5_face_pose.cc)
target_link_libraries(demo_yolov5
        ${CUDA_LIBRARIES}
        ${CUDNN_LIBRARY}
        ${TENSORRT_LIBRARY}
        trt_engine_lib
        nvonnxparser_plugin
        ${OpenCV_LIBS}
        thor)



