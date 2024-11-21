

#include "../include/YoloToolsGPU.h"



struct AffineMatrix {
    float value[6];
};

const int bbox_element = sizeof(AffineMatrix) / sizeof(float);

static __global__ void
decode_kernel(float *srcMlResult,
              float *outGpuArrBbox, //8400 * 4 * num_classes
              const int shapeBboxinOutLayer,//8400
              const float confidenceThreshold, //0.1
              const int maxCountDetection,
              const int countLabel) {
    const int countElementInBox = 4;

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= shapeBboxinOutLayer) {
        return;
    }

    float *bestConf = nullptr;
    float labelBbox = -1;
    for (int i = 0; i < countLabel; ++i) {
        float *conf = srcMlResult + (position + (shapeBboxinOutLayer * (countElementInBox + i)));
        if (bestConf == nullptr || bestConf[0] < conf[0]) {
            bestConf = conf;
            labelBbox = i;
        }
    }
    if (bestConf[0] < confidenceThreshold)
        return;


    int index = atomicAdd(outGpuArrBbox, 1);
    if (index >= maxCountDetection) {
        return;
    }

    float *center_x = srcMlResult + position;
    float *center_y = srcMlResult + (position + (shapeBboxinOutLayer * 1));
    float *width = srcMlResult + (position + (shapeBboxinOutLayer * 2));
    float *height = srcMlResult + (position + (shapeBboxinOutLayer * 3));

//    printf("index %i,  center_x %f outGpuArrBbox: %f\n", index, center_x[0], outGpuArrBbox[0]);

    float *pout_item = outGpuArrBbox + 1 + index * bbox_element;
    *pout_item++ = center_x[0];
    *pout_item++ = center_y[0];
    *pout_item++ = width[0];
    *pout_item++ = height[0];
    *pout_item++ = bestConf[0];
    *pout_item++ = labelBbox;

}


void YoloToolsGPU::СudaDecode(float *srcMlResult,
                              float *outGpuArrBbox, //8400 * 4 * num_classes
                              const int shapeBboxinOutLayer,//8400
                              const float confidenceThreshold, //0.1
                              const int maxCountDetection,
                              const int countLabel, //num_classes
                              cudaStream_t stream) {
    int block = 256;
    int grid = std::ceil(shapeBboxinOutLayer / (float) block);
    decode_kernel<<<grid, block, 0, stream>>>(
            (float *) srcMlResult, outGpuArrBbox, shapeBboxinOutLayer, confidenceThreshold, maxCountDetection,
            countLabel);
}

YoloToolsGPU::YoloToolsGPU(const int shapeBboxinOutLayer,
                           const float confidenceThreshold,
                           const int maxCountDetection,
                           const int countLabel,
                           const float nmsThresh,
                           cudaStream_t *stream) {
    _shapeBboxinOutLayer = shapeBboxinOutLayer;
    _confidenceThreshold = confidenceThreshold;
    _maxCountDetection = maxCountDetection;
    _countLabel = countLabel;
    _nmsThresh = nmsThresh;
    _stream = stream;
    const int countSrcElementInBox = 6;
    const int positionFromSizeArr =1;
    _getSizeAllocateMemoryFromDetections = positionFromSizeArr + _maxCountDetection * countSrcElementInBox;

    CUDA_FAILED(cudaMallocAsync(&_outGpuArrBbox, sizeof(float) * _getSizeAllocateMemoryFromDetections, *_stream));

    _outGpuArrBboxCPU = new vector<float>(_getSizeAllocateMemoryFromDetections);
}

map<double, vector<Detection>> YoloToolsGPU::GetMapDetect(float *bboxInGpu) {
    map<double, vector<Detection>> mapDetect;

    auto countBbox = bboxInGpu[0];

    for (int i = 0; i < countBbox; i++) {
        float *x = bboxInGpu + 1 + i * bbox_element;
        float *y = bboxInGpu + 1 + i * bbox_element + 1;
        float *w = bboxInGpu + 1 + i * bbox_element + 2;
        float *h = bboxInGpu + 1 + i * bbox_element + 3;
        float *lastConf = bboxInGpu + 1 + i * bbox_element + 4;
        float *idLabel = bboxInGpu + 1 + i * bbox_element + 5;
        vector<float> bboxArr = {x[0], y[0], w[0], h[0]};
        Detection det(bboxArr, lastConf[0], idLabel[0]);

        if (mapDetect.count(det.ClassId) == 0)
            mapDetect.emplace(det.ClassId, vector<Detection>());

        mapDetect[det.ClassId].push_back(det);
    }
    return mapDetect;
}

bool YoloToolsGPU::CmpDetection(const Detection &a, const Detection &b) {
    return a.Conf > b.Conf;
}

float YoloToolsGPU::IOU(vector<float> lbox, vector<float> rbox) {
    float interBox[] = {
            (max)(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), //left
            (min)(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
            (max)(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
            (min)(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

vector<Detection> YoloToolsGPU::Nms(map<double, vector<Detection>> mapDetect, float nmsThresh) {
    vector<Detection> res;

    if (mapDetect.size() == 0)
        return res;

    for (auto it = mapDetect.begin(); it != mapDetect.end(); it++) {
        auto &dets = it->second;
        sort(dets.begin(), dets.end(), CmpDetection);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto &item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (IOU(item.BBox, dets[n].BBox) > nmsThresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }

    return res;
}

YoloToolsGPU::~YoloToolsGPU() {

    if (_outGpuArrBboxCPU != nullptr) {
        delete _outGpuArrBboxCPU;
    }
    if (_outGpuArrBbox != nullptr) {
        cudaFree(_outGpuArrBbox);
    }
}

vector<Detection> YoloToolsGPU::GetDetctionsBbox(float *srcMlResult) {
    //работает только с batchSize = 1 (т.е. с одним изображением)
   CUDA_FAILED(cudaMemsetAsync(_outGpuArrBbox, 0,sizeof(float) * _getSizeAllocateMemoryFromDetections, *_stream));

    memset(_outGpuArrBboxCPU->data(), 0,  _getSizeAllocateMemoryFromDetections);

    СudaDecode((float *) srcMlResult,
               _outGpuArrBbox,
               _shapeBboxinOutLayer,
               _confidenceThreshold,
               _maxCountDetection,
               _countLabel,
               *_stream);

   CUDA_FAILED(cudaMemcpyAsync(_outGpuArrBboxCPU->data(), _outGpuArrBbox,
                                                sizeof(float) * _getSizeAllocateMemoryFromDetections,
                                                cudaMemcpyDeviceToHost,
                                                *_stream));


    auto mapDetect = GetMapDetect(_outGpuArrBboxCPU->data());
    auto exportDetctionBbox = Nms(mapDetect, _nmsThresh);

    return exportDetctionBbox;

}
