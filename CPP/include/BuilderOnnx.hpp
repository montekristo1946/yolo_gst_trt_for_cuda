#ifndef TENSORRTTOOLSWEDGE_BUILDERONNX_H
#define TENSORRTTOOLSWEDGE_BUILDERONNX_H

#include <iostream>

#include "Common.h"
#include "common/MainLogger.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include "NvidiaLogger.h"

using namespace std;
using namespace nvinfer1;
using namespace spdlog;

struct TRTDestroy {
    template<class T>
    void operator()(T *obj) const {
        if (obj) {
            delete obj;
        }
    }
};

template<class T>
using TRTUniquePtr = unique_ptr<T, TRTDestroy>;


class BuilderOnnx {

public:
    BuilderOnnx(unsigned int device, LayerSize shape,string nameInputLayer) {
        _idGpu = device;
        _inputH = shape.Height;
        _inputW = shape.Width;
        _channel = shape.Channel;
        _maxBatch = shape.BatchSize;
        _optimalBatch = shape.BatchSize;
        _minimalBatch = shape.BatchSize;
        _nameInputLayer = nameInputLayer;
        auto resetDevice = cudaSetDevice(_idGpu);
        if(resetDevice !=0)
        {
            throw invalid_argument( "[BuilderOnnx::Ctor]  Fail Set ID Gpu: "+ to_string(_idGpu) );
        }
    }

    bool ApiToFileModel(string &modelInputPath, string &modelOutputPath, bool setHalfModel = true);

private:
    int _inputH;
    int _inputW;
    int _channel;
    int _idGpu;
    string _nameInputLayer;
    int _maxBatch;
    int _optimalBatch ;
    int _minimalBatch ;

    shared_ptr<logger> _logger = get("MainLogger");
};


#endif //TENSORRTTOOLSWEDGE_BUILDERONNX_H
