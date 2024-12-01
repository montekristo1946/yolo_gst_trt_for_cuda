#ifndef TENSORRTTOOLS_TRTEngine_H
#define TENSORRTTOOLS_TRTEngine_H


#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <experimental/filesystem>
#include <spdlog/pattern_formatter.h>
#include <NvidiaLogger.h>
#include "IDispose.h"
#include "YoloToolsGPU.h"
#include <filesystem>
#include "TRTEngineConfig.hpp"

using namespace nvinfer1;

class TRTEngine : public IDispose {
public:
    TRTEngine( cudaStream_t *stream) {

        if (!stream)
            throw std::runtime_error("[TRTEngine::Ctr] Null reference exception");
        _stream = stream;
        _logger->info("[TRTEngine::Ctr]  Init  TRTEngine ok");
    }

    bool InitTRTEngine(const string engineName,
                       const int deviseId,
                       const float confThresh,
                       const float nmsThresh,
                       const int maxNumOutputBbox);

    ~TRTEngine() {
        _logger->info("[TRTEngine::~TRTEngine] Call Free cuda");
        Dispose();
    }

    const int GetFullSizeOutputLayer();

    const int GetFullSizeInputLayer();

    vector<int32_t> InputShape() const { return _inputShape; }

    vector<int32_t> OutputShape() const { return _outputShape; }

    bool DoInferenceNMSAsync(vector<Detection>& outPutRectangles);

    vector<void *> _buffers; //спрятать этих ребят в приват

private:

    bool Dispose();
    cudaStream_t *_stream;
    size_t GetSizeByDim(const vector<int32_t> &dims);


    IRuntime *_runtime;
    ICudaEngine *_engine;
    IExecutionContext *_context;


    vector<int32_t> _inputShape;
    vector<int32_t> _outputShape;
    bool _isDispose = false;

    bool SetShape(vector<int32_t> &exportShape, Dims importShape);

    ICudaEngine *LoadWeightInFile(const string engineName, IRuntime *runtime);

    YoloToolsGPU *_yoloToolsGPU;

     bool InitYoloToolsGPU(const float confThresh, const float nmsThresh, const int maxNumOutputBbox);
    shared_ptr<logger> _logger = get("MainLogger");
};


#endif //TENSORRTTOOLS_TRTEngine_H
