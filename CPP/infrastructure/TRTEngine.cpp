


#include "TRTEngine.hpp"

#include "CudaUtility.h"


// calculate size of tensor
size_t TRTEngine::GetSizeByDim(const vector<int32_t> &dims) {
    size_t size = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        size *= dims[i];
    }
    return size;
}

bool TRTEngine::InitTRTEngine(const string engineName,
                              const int deviseId,
                              const float confThresh,
                              const float nmsThresh,
                              const int maxNumOutputBbox) {
    try {
        _logger->info("[TRTEngine::InitTRTEngine]  EngineName:{}", engineName);
        _logger->info("[TRTEngine::InitTRTEngine]  Set GPU:{}", deviseId);

        auto resetDevice = cudaSetDevice(deviseId);
        if (resetDevice != 0) {
            throw invalid_argument("[TRTEngine::InitTRTEngine]  Fail Set ID Gpu: " + to_string(deviseId));
        }

        NvidiaLogger nvidiaLogger;
        nvidiaLogger.log(ILogger::Severity::kINFO, "[TRTEngine::InitTRTEngine] Init NvidiaLogger ok");

        _runtime = createInferRuntime(nvidiaLogger);

        if (_runtime == nullptr) {
            _logger->error("[TRTEngine::InitTRTEngine]  runtime == Null");
            return false;
        }

        _engine = LoadWeightInFile(engineName, _runtime);

        if (_engine == nullptr) {
            _logger->error("[TRTEngine::InitTRTEngine]  _engine == Null");
            return false;
        }

        _context = _engine->createExecutionContext();
        if (_context == nullptr) {
            _logger->error("[TRTEngine::InitTRTEngine]  _context == Null");
            return false;
        }

        auto countIOTensor = _engine->getNbIOTensors();
        _buffers = vector<void *>(countIOTensor);

        for (int i = 0; i < countIOTensor; ++i) {
            auto const name = _engine->getIOTensorName(i);
            auto sizeBindingDimensions = _engine->getTensorShape(name);

            string printLayer = "";
            for (size_t i = 0; i < sizeBindingDimensions.nbDims; ++i) {
                printLayer.append(to_string(sizeBindingDimensions.d[i]) + ":");
            }

            _logger->info("[TRTEngine::InitTRTEngine] Network Layer:{}; info shape:[{}]", name, printLayer);

            bool const hasRuntimeDim = any_of(sizeBindingDimensions.d,
                                              sizeBindingDimensions.d + sizeBindingDimensions.nbDims,
                                              [](int32_t dim) { return dim == -1; });
            if (hasRuntimeDim) {

                _logger->error("[TRTEngine::InitTRTEngine] dynamic shape not Implemented");
                return false;
            }

            auto tensorIOMode = _engine->getTensorIOMode(name);

            if (tensorIOMode == TensorIOMode::kINPUT) {


                auto resSetShape = SetShape(_inputShape, sizeBindingDimensions);
                if (!resSetShape) {
                    _logger->error("[TRTEngine::InitTRTEngine]  SetShape input error");
                    return false;
                }

                auto resCudaMalloc = cudaMalloc(&_buffers[i], GetSizeByDim(_inputShape) * sizeof(float));
                if (resCudaMalloc != 0) {
                    _logger->error("[TRTEngine::InitTRTEngine]  cudaMalloc input network error {}",
                                   to_string(resCudaMalloc));
                    return false;
                }
            }

            if (tensorIOMode == TensorIOMode::kOUTPUT) {
                auto resSetShape = SetShape(_outputShape, sizeBindingDimensions);
                if (!resSetShape) {
                    _logger->error("[TRTEngine::InitTRTEngine]  SetShape оutput error");
                    return false;
                }

                auto res = cudaMalloc(&_buffers[i], GetSizeByDim(_outputShape) * sizeof(float));
                if (res != 0) {
                    _logger->error("[TRTEngine::InitTRTEngine]  cudaMalloc input network error {}", to_string(res));
                    return false;
                }


            }
            if (tensorIOMode == TensorIOMode::kNONE) {
                _logger->error("[TRTEngine::InitTRTEngine]  Not Implemented Layer kNONE; name {}", name);
                return false;
            }

            auto resSetTensorAddress = _context->setTensorAddress(name, _buffers[i]);
            if (!resSetTensorAddress) {
                _logger->error("[TRTEngine::InitTRTEngine]  _context->setTensorAddress error {}", resSetTensorAddress);
                return false;
            }

        }

        if (_buffers.size() != 2) {
            _logger->error("[TRTEngine::InitTRTEngine]  Not implemented _buffers size, current: {}", _buffers.size());
            return false;
        }

        auto resInitYoloToolsGPU = InitYoloToolsGPU(confThresh, nmsThresh, maxNumOutputBbox);

        if (resInitYoloToolsGPU == false) {
            _logger->error("[TRTEngine::InitTRTEngine] fail init InitYoloToolsGPU");
            return false;
        }

        _logger->info("[TRTEngine::InitTRTEngine]  Init  TRTEngine ok");
        return true;
    }
    catch (exception &e) {
        _logger->error("[TRTEngine::InitTRTEngine]  {}", e.what());
    }
    catch (...) {
        _logger->error("[TRTEngine::InitTRTEngine]  Unknown exception!");
    }
    return false;
}


bool TRTEngine::Dispose() {
    try {
        if (_isDispose)
            return false;

        for (void *buf: _buffers) {
            cudaFree(buf);
        }
        if (_context != nullptr)
            delete _context;

        if (_engine != nullptr)
            delete _engine;

        if (_runtime != nullptr)
            delete _runtime;

        if (_yoloToolsGPU != nullptr) {
            _yoloToolsGPU->~YoloToolsGPU();
        }


        _logger->info("[TRTEngine::Dispose] TRTEngine Dispose ok");

        _isDispose = true;
        return true;
    }
    catch (exception &e) {
        _logger->error("[Dispose]  {}", e.what());
    }
    catch (...) {
        _logger->error("[Dispose]  Unknown exception!");
    }
    return false;
}


const int TRTEngine::GetFullSizeInputLayer() {

    return GetSizeByDim(_inputShape);
}

const int TRTEngine::GetFullSizeOutputLayer() {

    return GetSizeByDim(_outputShape);
}

bool TRTEngine::SetShape(vector<int32_t> &exportShape, Dims importShape) {

    try {
        constexpr int32_t kDEFAULT_VALUE = 1;

        bool const hasRuntimeDim = any_of(importShape.d,
                                          importShape.d + importShape.nbDims,
                                          [](int32_t dim) { return dim == -1; });
        if (hasRuntimeDim) {

            _logger->error("[TRTEngine::InitTRTEngine] dynamic shape! Set {}", kDEFAULT_VALUE);
        }

        exportShape.resize(importShape.nbDims);


        transform(importShape.d, importShape.d + importShape.nbDims, exportShape.begin(),
                  [&](int32_t dimension) {
                      return dimension >= 0 ? dimension : kDEFAULT_VALUE;
                  });

        return true;
    }
    catch (exception &e) {
        _logger->error("[TRTEngine::SetShape]  {}", e.what());
    }
    catch (...) {
        _logger->error("[TRTEngine::SetShape]  Unknown exception!");
    }
    return false;
}

ICudaEngine *TRTEngine::LoadWeightInFile(const string engineName, IRuntime *runtime) {

    ifstream file(engineName, ios::binary);
    unique_ptr<char[]> trtModelStream = nullptr;
    auto size = filesystem::file_size(engineName);
    if (file.good()) {
        file.seekg(0, file.beg);
        trtModelStream = make_unique<char[]>(size);
        file.read(trtModelStream.get(), size);
        file.close();
    }

    if (trtModelStream == nullptr) {
        _logger->error("[TRTEngine::LoadWeightInFile]  trtModelStream == Null");
        return nullptr;
    }

    auto engine = runtime->deserializeCudaEngine(trtModelStream.get(), size);

    return engine;
}

bool TRTEngine::DoInferenceNMSAsync(vector<Detection>& outPutRectangles) {

    try {

        bool resultEnqueue = _context->enqueueV3(*_stream);

        if (!resultEnqueue) {
            _logger->error("[TRTEngine::DoInferenceNMSAsync] Error context->enqueue");
            return false;
        }

        auto resultDetect = _yoloToolsGPU->GetDetctionsBbox((float *)_buffers[1]);
        outPutRectangles = resultDetect;

        CUDA_FAILED( cudaStreamSynchronize(*_stream));

        return true;

    }
    catch (exception &e) {
        _logger->error("[TRTEngine::DoInferenceNMSAsync]  {}", e.what());
    }
    catch (...) {
        _logger->error("[TRTEngine::DoInferenceNMSAsync]  Unknown exception!");
    }
    return false;
}

bool TRTEngine::InitYoloToolsGPU(const float confThresh, const float nmsThresh, const int maxNumOutputBbox) {

    try {

        auto outputShape = OutputShape();
        if (outputShape[0] != 1) {
            _logger->error("[TRTEngine::InitYoloToolsGPU] Not implemented shape {}", outputShape[0]);
            return false;
        }

        int shapeBboxinOutLayer = outputShape[2];
        const int countCoordinates = 4;
        int countLabel = outputShape[1] - countCoordinates;

        _yoloToolsGPU = new YoloToolsGPU(shapeBboxinOutLayer,
                                         confThresh,
                                         maxNumOutputBbox,
                                         countLabel,
                                         nmsThresh,
                                         _stream);

        return true;
    }
    catch (exception &e) {
        _logger->error("[TRTEngine::InitYoloToolsGPU]  {}", e.what());
    }
    catch (...) {
        _logger->error("[TRTEngine::InitYoloToolsGPU]  Unknown exception!");
    }
    return false;


}


