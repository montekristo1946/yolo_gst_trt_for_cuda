#if defined(_WIN32) || defined(_WIN64)
#define MYLIB_EXPORT __declspec(dllexport)
#define MYLIB_IMPORT __declspec(dllimport)
#else
#define MYLIB_EXPORT __attribute__((visibility("default")))
#define MYLIB_IMPORT __attribute__((visibility("default")))
#define MYLIB_HIDDEN __attribute__((visibility("hidden")))

#endif

#include <BufferFrameGpu.h>
#include <GstBufferManager.h>
#include <GstDecoder.h>

#include "MainLogger.hpp"
#include "Common.h"
#include "BuilderOnnx.hpp"
#include <iostream>
#include <TRTEngine.hpp>
#include <TRTEngineConfig.hpp>

#include "AlgorithmsPolygon.h"
#include "CudaStream.h"
#include "DtoToCharp.h"
#include "EnginePipeline.h"
#include "NvJpgEncoder.h"
#include "SettingPipeline.h"


class BufferFrameGpu;
using namespace std;

void OperationalSavingLogs()
{
    shared_ptr<logger> _loger = spdlog::get("MainLogger");
    if (_loger != nullptr)
        _loger->flush();
}

void SlowloggingError(string message)
{
    shared_ptr<logger> _loger = spdlog::get("MainLogger");
    if (_loger != nullptr)
    {
        _loger->error(message);
        _loger->flush();
    }
    else
    {
        cerr << message << endl;
    }
}

extern "C" MYLIB_EXPORT void InitLogger(const char* logPathFile)
{
    if (logPathFile == nullptr || logPathFile == NULL)
    {
        error("[InitLogger] Null input parameters");
        return;
    }

    info("[InitLogger] InitLogger; yolo_gst_for_cuda  Version libExtensionCHarp.so:  0.2");
    info("[InitLogger] LogPath: {}", logPathFile);

    auto logPathFileString = string(logPathFile);
    auto mainLogger = MainLogger(logPathFileString);
    OperationalSavingLogs();
}


extern "C" MYLIB_EXPORT bool ConverterNetworkWeight(const char* pathOnnxModelChar,
                                                    const char* exportPathModelChar,
                                                    const LayerSize* config,
                                                    const unsigned int idGpu,
                                                    const bool setHalfModel = true)
{
    try
    {
        if (pathOnnxModelChar == nullptr || exportPathModelChar == nullptr || config == nullptr)
        {
            error("[ConverterNetworkWeight] Null input parameters");
            return false;
        }
        auto shape = *config;
        string wtsName = string(pathOnnxModelChar);
        string exportSave = string(exportPathModelChar);
        string nameInputLayer = "images";
        auto builder = BuilderOnnx(idGpu, shape, nameInputLayer);
        auto resSAveModel = builder.ApiToFileModel(wtsName, exportSave, setHalfModel);
        return resSAveModel;
    }
    catch (exception& e)
    {
        SlowloggingError("[ConverterNetworkWeight]  " + string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[ConverterNetworkWeight] Unknown exception!");
    }

    return false;
}


extern "C" MYLIB_EXPORT CudaStream* CreateCudaStream()
{
    spdlog::info("[CudaStream] Init CudaStream Version 1.0.0");
    try
    {
        auto cudaStream = new CudaStream();

        OperationalSavingLogs();
        return cudaStream;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CudaStream]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CudaStream] Unknown exception!");
    }

    return nullptr;
}

extern "C" MYLIB_EXPORT TRTEngine* CreateTRTEngine(const TRTEngineConfig* config, CudaStream* cudaStream)
{
    info("[CreateTRTEngine] Init TRTEngine Version 1.0.0");
    if (!config)
    {
        error("[CreateTRTEngine] Null input parameters");
        return nullptr;
    }
    try
    {
        if (!config->EngineName ||
            config->DeviseId < 0 ||
            config->ConfThresh < 0 || config->ConfThresh > 1 ||
            config->NmsThresh < 0 || config->NmsThresh > 1 ||
            config->MaxNumOutputBbox < 0 || config->MaxNumOutputBbox > 1000)
        {
            throw std::runtime_error("[CreateTRTEngine] Invalid parameters");
        }
        auto trtEngine = new TRTEngine(cudaStream->GetStream());
        bool resultInitTRT = trtEngine->InitTRTEngine(config->EngineName,
                                                      config->DeviseId,
                                                      config->ConfThresh,
                                                      config->NmsThresh,
                                                      config->MaxNumOutputBbox);
        if (!resultInitTRT)
            throw std::runtime_error("[CreateTRTEngine] fail InitTRTEngine");

        OperationalSavingLogs();
        return trtEngine;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CreateTRTEngine]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CreateTRTEngine] Unknown exception!");
    }

    return nullptr;
}


extern "C" MYLIB_EXPORT BufferFrameGpu* CreateBufferFrameGpu()
{
    info("[CreateBufferFrameGpu] Init BufferFrameGpu Version 1.0.0");
    try
    {
        unsigned sizeBuffer = 5;
        auto buffer = new BufferFrameGpu(sizeBuffer);
        OperationalSavingLogs();
        return buffer;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CreateBufferFrameGpu]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CreateBufferFrameGpu] Unknown exception!");
    }

    return nullptr;
}

extern "C" MYLIB_EXPORT GstBufferManager* CreateGstBufferManager(BufferFrameGpu* bufferFrameGpu, CudaStream* cudaStream)
{
    info("[CreateGstBufferManager] Init GstBufferManager Version 1.0.0");
    try
    {
        auto bufferManager = new GstBufferManager(bufferFrameGpu, cudaStream->GetStream());
        OperationalSavingLogs();
        return bufferManager;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CreateGstBufferManager]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CreateGstBufferManager] Unknown exception!");
    }

    return nullptr;
}

extern "C" MYLIB_EXPORT GstDecoder* CreateGstDecoder(GstBufferManager* bufferManager)
{
    info("[CreateGstDecoder] Init GstDecoder Version 1.0.0");
    try
    {
        auto gstDecoder = new GstDecoder(bufferManager);
        OperationalSavingLogs();
        return gstDecoder;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CreateGstDecoder]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CreateGstDecoder] Unknown exception!");
    }

    return nullptr;
}

extern "C" MYLIB_EXPORT NvJpgEncoder* CreateNvJpgEncoder(CudaStream* cudaStream)
{
    info("[CreateNvJpgEncoder] Init NvJpgEncoder Version 1.0.0");
    try
    {
        auto encoder = new NvJpgEncoder(cudaStream->GetStream());
        OperationalSavingLogs();
        return encoder;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CreateNvJpgEncoder]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CreateNvJpgEncoder] Unknown exception!");
    }

    return nullptr;
}

extern "C" MYLIB_EXPORT TrackerManager* CreateTrackerManager(const int frameRate,
                                                             const int trackBuffer,
                                                             const float trackThresh,
                                                             const float highThresh,
                                                             const float matchThresh,
                                                             const int maxNumTrackers)
{
    info("[CreateTrackerManager] Init TrackerManager");
    try
    {
        auto tracker = new TrackerManager(
            frameRate,
            trackBuffer,
            trackThresh,
            highThresh,
            matchThresh,
            maxNumTrackers);

        OperationalSavingLogs();
        return tracker;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CreateTrackerManager]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CreateTrackerManager] Unknown exception!");
    }

    return nullptr;
}

extern "C" MYLIB_EXPORT AlgorithmsPolygon* CreateAlgorithmsPolygon()
{
    info("[CreateAlgorithmsPolygon] Init AlgorithmsPolygon");
    try
    {
        auto algorithmsPolygon = new AlgorithmsPolygon();
        OperationalSavingLogs();
        return algorithmsPolygon;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CreateAlgorithmsPolygon]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CreateAlgorithmsPolygon] Unknown exception!");
    }

    return nullptr;
}

extern "C" MYLIB_EXPORT bool AlgorithmsPolygonClear(AlgorithmsPolygon* algorithmsPolygon)
{
    info("[AlgorithmsPolygonClear] call AlgorithmsPolygonClear");
    try
    {
        if ( !algorithmsPolygon)
            throw std::runtime_error("[AlgorithmsPolygonClear] Null parameters");

        algorithmsPolygon->Clear();
        OperationalSavingLogs();
        return true;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[AlgorithmsPolygonClear]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[AlgorithmsPolygonClear] Unknown exception!");
    }

    return false;
}
extern "C" MYLIB_EXPORT bool AlgorithmsPolygonAppend(AlgorithmsPolygon* algorithmsPolygon,PolygonsSettingsExternal * polygonsSettings)
{
    info("[AlgorithmsPolygonAppend] call AlgorithmsPolygonAppend");
    try
    {
        if (!polygonsSettings || !algorithmsPolygon)
            throw std::runtime_error("[AlgorithmsPolygonAppend] Null parameters");

        auto points = vector<Point>();
        for(int i = 0; i < polygonsSettings->CountPoints; i++)
        {
            auto x = polygonsSettings->PolygonsX[i];
            auto y =  polygonsSettings->PolygonsY[i];
            points.emplace_back(x, y);
        }

        auto polygon = Polygons(polygonsSettings->IdPolygon, points);

        algorithmsPolygon->AppendPolygon(polygon);
        OperationalSavingLogs();
        return true;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[AlgorithmsPolygonAppend]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[AlgorithmsPolygonAppend] Unknown exception!");
    }

    return false;
}



extern "C" MYLIB_EXPORT EnginePipeline* CreateEnginPipeline(TRTEngine* trtEngine,
                                                            BufferFrameGpu* bufferFrameGpu,
                                                            CudaStream* cudaStream,
                                                            SettingPipeline* settingPipeline,
                                                            NvJpgEncoder* encoder,
                                                            TrackerManager* trackerManager,
                                                            AlgorithmsPolygon * algorithmsPolygon)
{
    if (!trtEngine || !bufferFrameGpu || !cudaStream || !settingPipeline || !encoder || !trackerManager || !algorithmsPolygon)
    {
        error("[CreateEnginPipeline] Null reference exception");
        return nullptr;
    }

    info("[CreateEnginPipeline] Init CreateEnginPipeline Version 1.0.0");
    info("[CreateEnginPipeline] SettingPipeline: WidthImgMl={}, HeightImgMl={}, CountImgToBackground={}",
         settingPipeline->WidthImgMl,
         settingPipeline->HeightImgMl,
         settingPipeline->CountImgToBackground);

    try
    {
        auto settingPipelineLoc = new SettingPipeline();
        settingPipelineLoc->WidthImgMl = settingPipeline->WidthImgMl,
            settingPipelineLoc->HeightImgMl = settingPipeline->HeightImgMl,
            settingPipelineLoc->CountImgToBackground = settingPipeline->CountImgToBackground;

        auto pipeline = new EnginePipeline(trtEngine,
                                           bufferFrameGpu,
                                           cudaStream->GetStream(),
                                           settingPipelineLoc,
                                           encoder,
                                           trackerManager,
                                           algorithmsPolygon);

        OperationalSavingLogs();
        return pipeline;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CreateThermalPipeline]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CreateThermalPipeline] Unknown exception!");
    }

    return nullptr;
}


extern "C" MYLIB_EXPORT bool StartPipelineGst(GstDecoder* gstDecoder, const char* connectString)
{
    if (!gstDecoder)
    {
        error("[CreateEnginPipeline] GstDecoder is null");
        return false;
    }

    try
    {
        auto res = gstDecoder->StartPipeline(connectString);
        return res;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[CreateThermalPipeline]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[CreateThermalPipeline] Unknown exception!");
    }

    return false;
}


extern "C" MYLIB_EXPORT bool Dispose(IDispose* ptr)
{
    try
    {
        if (!ptr)
        {
            OperationalSavingLogs();
            return false;
        }
        delete ptr;

        OperationalSavingLogs();
        return true;
    }
    catch (exception& e)
    {
        SlowloggingError("[Dispose]  " + string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[Dispose] Unknown exception!");
    }

    return false;
}

extern "C" MYLIB_EXPORT bool DisposeArr(void* ptr)
{
    try
    {
        if (!ptr)
        {
            OperationalSavingLogs();
            return false;
        }
        delete [] ptr;

        OperationalSavingLogs();
        return true;
    }
    catch (exception& e)
    {
        SlowloggingError("[DisposeArr]  " + string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[DisposeArr] Unknown exception!");
    }

    return false;
}

extern "C" MYLIB_EXPORT bool DoInferencePipeline(EnginePipeline* enginePipeline, PipelineOutputData* pipelineOutputData)
{
    try
    {
        if (!enginePipeline || !pipelineOutputData)
        {
            SlowloggingError("[DoInferencePipeline] Bad Input Data ");
            return false;
        }

        vector<RectDetect> resultNms;
        auto res = enginePipeline->GetResultImages(resultNms);

        if (!res)
        {
            return false;
        }

        auto* arr = new RectDetectExternal[resultNms.size()];

        for (int i = 0; i < resultNms.size(); i++)
        {
            auto polygons = resultNms[i];
            auto polygonsId = new int[polygons.PolygonsId.size()];
            copy(polygons.PolygonsId.begin(), polygons.PolygonsId.end(), polygonsId);

            arr[i].X = polygons.X;
            arr[i].Y = polygons.Y;
            arr[i].Width = polygons.Width;
            arr[i].Height = polygons.Height;
            arr[i].IdClass = polygons.IdClass;
            arr[i].Veracity = polygons.Veracity;
            arr[i].TrackId = polygons.TrackId;
            arr[i].TimeStamp = polygons.TimeStamp;
            arr[i].PolygonsId = polygonsId;
            arr[i].PolygonsIdLen = polygons.PolygonsId.size();
        }

        pipelineOutputData->RectanglesLen = resultNms.size();
        pipelineOutputData->Rectangles = arr;

        return true;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[DoInferencePipeline]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[DoInferencePipeline] Unknown exception!");
    }

    return false;
}

extern "C" MYLIB_EXPORT bool GetCurrenImage(EnginePipeline* enginePipeline, ImageFrame* image)
{
    try
    {
        if (!image)
        {
            SlowloggingError("[GetCurrenImage] Bad Input Data ");
            return false;
        }
        std::vector<unsigned char>* encodedImage = enginePipeline->GetFrame();
        auto timeStamp = enginePipeline->GetCurrentTimeStamp();
        auto arrOutput = new unsigned char[encodedImage->size()];
        copy(encodedImage->begin(), encodedImage->end(), arrOutput);
        image->ImageLen = encodedImage->size();
        image->ImagesData = arrOutput;
        image->TimeStamp = timeStamp;
        return true;
    }
    catch (std::exception& e)
    {
        SlowloggingError("[GetCurrenImage]  " + std::string(e.what()));
    }
    catch (...)
    {
        SlowloggingError("[GetCurrenImage] Unknown exception!");
    }

    return false;
}
