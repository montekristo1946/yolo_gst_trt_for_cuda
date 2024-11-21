
#ifndef THERMALPIPLINE_H
#define THERMALPIPLINE_H
#include <BufferFrameGpu.h>
#include <GstBufferManager.h>
#include <GstDecoder.h>
#include <IDispose.h>
#include <MatConverter.h>
#include <TRTEngine.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include "NvJpgEncoder.h"
#include "SettingPipeline.h"


class EnginePipeline: public IDispose
{
public:

    EnginePipeline(
        TRTEngine *trtEngine,
        BufferFrameGpu *bufferFrameGpu,
        GstBufferManager *gstBufferManager,
        GstDecoder *gstDecoder,
        cudaStream_t* streem,
        SettingPipeline* settingPipeline,
        NvJpgEncoder* encoder);


    bool StartPipeline( string connectCamera );


    bool ConverterDetection(vector<Detection>& vector);
    void UpdateCurrentTimeStamp(uint64_t &uint64);
    bool GetResultImages(vector<Detection>& resultNms, uint64_t &timeStamp);
    ~EnginePipeline();
    std::vector<unsigned char>* GetFrame();
    uint64_t GetCurrentTimeStamp() const { return _currentTimeStamp; }

private:
    void UpdateCurrentImg(cuda::GpuMat* gpu_mat);
    void UpdateBackground();
    void UpdateDiffImg();
    cuda::GpuMat* ResizeImages(cuda::GpuMat* imageSrc);
    void LoadImgToTrt();

    MatConverter * _matConverter = nullptr;

    TRTEngine *_trtEngine = nullptr;
    BufferFrameGpu *_bufferFrameGpu = nullptr;
    GstBufferManager *_gstBufferManager = nullptr;
    GstDecoder *_gstDecoder = nullptr;
    cudaStream_t* _streem = 0;
    SettingPipeline* _settingPipeline= nullptr;

    std::vector<uchar> _imagesExport ;

    cuda::GpuMat *_imageBackground = nullptr;
    cuda::GpuMat *_difImage = nullptr;
    cuda::GpuMat * _currentImage = nullptr;
    uint64_t  _currentTimeStamp = 0;

    NvJpgEncoder* _encoder  = nullptr;
    shared_ptr<logger> _logger = get("MainLogger");
};
#endif //THERMALPIPLINE_H
