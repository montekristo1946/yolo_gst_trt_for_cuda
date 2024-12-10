
#ifndef THERMALPIPLINE_H
#define THERMALPIPLINE_H
#include <BufferFrameGpu.h>
#include <GstBufferManager.h>
#include <IDispose.h>
#include <TrackerManager.h>
#include <TRTEngine.hpp>


#include "NvJpgEncoder.h"
#include "SettingPipeline.h"
#include "ByteTrack/BYTETracker.h"


class EnginePipeline: public IDispose
{
public:

    EnginePipeline(
        TRTEngine *trtEngine,
        BufferFrameGpu *bufferFrameGpu,
        cudaStream_t* streem,
        SettingPipeline* settingPipeline,
        NvJpgEncoder* encoder,
        TrackerManager* _trackerManager);

    bool GetResultImages( vector<RectDetect>&  result);
    ~EnginePipeline();
    std::vector<unsigned char>* GetFrame();
    uint64_t GetCurrentTimeStamp() const { return _currentTimeStamp; }

private:
    bool ConverterDetection(vector<RectDetect>& vector);
    void UpdateCurrentTimeStamp(uint64_t &uint64);
    void UpdateCurrentImg(FrameGpu<Npp8u>* frame);
    void UpdateBackground();
    void LoadImgToTrt();

    TRTEngine *_trtEngine = nullptr;
    BufferFrameGpu *_bufferFrameGpu = nullptr;
    cudaStream_t* _streem = 0;
    SettingPipeline* _settingPipeline= nullptr;
    TrackerManager* _trackerManager  = nullptr;

    std::vector<uchar> _imagesExport ;
    FrameGpu<Npp32f> *_imageBackground = nullptr;
    FrameGpu<Npp8u>* _currentImage = nullptr;
    uint64_t  _currentTimeStamp = 0;
    NvJpgEncoder* _encoder  = nullptr;
    shared_ptr<logger> _logger = get("MainLogger");
    NppFunction * _nppFunctions = new NppFunction();

    // byte_track::BYTETracker * _tracker = new byte_track::BYTETracker(75, 75, 0.2, 0.5, 0.7);



};
#endif //THERMALPIPLINE_H
