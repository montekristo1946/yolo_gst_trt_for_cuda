//
// Created by user on 19.10.2024.
//

#ifndef YOLOGSTFORCUDA_YOLOTOOLSGPU_H
#define YOLOGSTFORCUDA_YOLOTOOLSGPU_H



#include <vector>

#include "NvInfer.h"
#include <stdio.h>
#include <cmath>
#include <map>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include "CudaUtility.h"

using namespace std;


  // x, y, width, height, confidence, class
struct Detection {
public:
    Detection() {};
    Detection(vector<float> &bbox, float conf, float class_id) {
        BBox = bbox;
        Conf = conf;
        ClassId = class_id;
    }

    vector<float> BBox;     //center_x center_y w h
    float Conf;
    float ClassId;
};

class YoloToolsGPU  {
public:
    YoloToolsGPU(const int shapeBboxinOutLayer,
                 const float confidenceThreshold,
                 const int maxCountDetection,
                 const int countLabel,
                 const float nmsThresh,
                 cudaStream_t *stream);

    vector<Detection> GetDetctionsBbox(float *srcMlResult);
    ~YoloToolsGPU();

private:

    void СudaDecode(float *srcMlResult,
                    float *outGpuArrBbox, //8400 * 4 * num_classes
                    const int shapeBboxinOutLayer,//8400
                    const float confidenceThreshold, //0.1
                    const int maxCountDetection,
                    const int countLabel, //num_classes
                    cudaStream_t stream);


    map<double, vector<Detection>> GetMapDetect(float *bboxInGpu);

    static bool CmpDetection(const Detection &a, const Detection &b);

    float IOU(vector<float> lbox, vector<float> rbox);

    vector<Detection> Nms(map<double, vector<Detection>> mapDetect, float nmsThresh);

    vector<float>* _outGpuArrBboxCPU = nullptr;
    float *_outGpuArrBbox = nullptr;

    int _shapeBboxinOutLayer;
    float _confidenceThreshold;
    int _maxCountDetection;
    int _countLabel;
    float _nmsThresh;
    cudaStream_t *_stream;

    int _getSizeAllocateMemoryFromDetections ;

};

#endif
