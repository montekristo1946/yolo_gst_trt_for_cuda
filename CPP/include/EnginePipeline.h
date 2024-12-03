
#ifndef THERMALPIPLINE_H
#define THERMALPIPLINE_H
#include <BufferFrameGpu.h>
#include <GstBufferManager.h>
#include <IDispose.h>
#include <TRTEngine.hpp>


#include "NvJpgEncoder.h"
#include "SettingPipeline.h"


class EnginePipeline: public IDispose
{
public:

    EnginePipeline(
        TRTEngine *trtEngine,
        BufferFrameGpu *bufferFrameGpu,
        cudaStream_t* streem,
        SettingPipeline* settingPipeline,
        NvJpgEncoder* encoder);

    bool GetResultImages(vector<Detection>& resultNms, uint64_t &timeStamp);
    ~EnginePipeline();
    std::vector<unsigned char>* GetFrame();
    uint64_t GetCurrentTimeStamp() const { return _currentTimeStamp; }

private:
    bool ConverterDetection(vector<Detection>& vector);
    void UpdateCurrentTimeStamp(uint64_t &uint64);
    void UpdateCurrentImg(FrameGpu<Npp8u>* frame);
    void UpdateBackground();
    void LoadImgToTrt();

    TRTEngine *_trtEngine = nullptr;
    BufferFrameGpu *_bufferFrameGpu = nullptr;

    cudaStream_t* _streem = 0;
    SettingPipeline* _settingPipeline= nullptr;
    std::vector<uchar> _imagesExport ;
    FrameGpu<Npp32f> *_imageBackground = nullptr;
    FrameGpu<Npp8u>* _currentImage = nullptr;
    uint64_t  _currentTimeStamp = 0;
    NvJpgEncoder* _encoder  = nullptr;
    shared_ptr<logger> _logger = get("MainLogger");
    NppFunction * _nppFunctions = new NppFunction();

};
#endif //THERMALPIPLINE_H
