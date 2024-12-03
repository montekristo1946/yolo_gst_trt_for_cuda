#include "EnginePipeline.h"


EnginePipeline::EnginePipeline(TRTEngine* trtEngine,
                               BufferFrameGpu* bufferFrameGpu,
                               cudaStream_t* streem,
                               SettingPipeline* settingPipeline,
                               NvJpgEncoder* encoder)
{
    if (!trtEngine || !bufferFrameGpu || !streem || !encoder)
        throw std::runtime_error("[ThermalPipeline::ThermalPipeline] Null reference exception");

    _trtEngine = trtEngine;
    _bufferFrameGpu = bufferFrameGpu;

    _streem = streem;
    _settingPipeline = settingPipeline;
    _encoder = encoder;

    auto channel = 1;
    _imageBackground = FrameGpu<
        Npp32f>::CreateNew(_settingPipeline->WidthImgMl, _settingPipeline->HeightImgMl, channel);
}


bool EnginePipeline::ConverterDetection(vector<Detection>& vector)
{
    for (auto& det : vector)
    {
        det.BBox[2] = det.BBox[2] / _settingPipeline->WidthImgMl; //width
        det.BBox[3] = det.BBox[3] / _settingPipeline->HeightImgMl; //heigt
        det.BBox[0] = det.BBox[0] / _settingPipeline->WidthImgMl; //x center
        det.BBox[1] = det.BBox[1] / _settingPipeline->HeightImgMl; //y center
    }
    return true;
}

void EnginePipeline::UpdateCurrentTimeStamp(uint64_t& timeStamp)
{
    _currentTimeStamp = timeStamp;
}

void EnginePipeline::LoadImgToTrt()
{
    if (!_imageBackground || !_currentImage)
        throw std::runtime_error("[ThermalPipeline::LoadImgToTrt] Null reference exception");

    auto currentImgFloat = _nppFunctions->ConvertFrame8u32f(_currentImage);
    auto resDiffImg = _nppFunctions->AbsDiff(_imageBackground, currentImgFloat);

    float* ptrImgGPU = static_cast<float*>(_trtEngine->_buffers[0]);
    auto oneLayerSize = _settingPipeline->WidthImgMl * _settingPipeline->HeightImgMl;

    CUDA_FAILED(
        cudaMemcpy(ptrImgGPU, resDiffImg->ImagePtr(), oneLayerSize* sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_FAILED(
        cudaMemcpy(ptrImgGPU+oneLayerSize, _imageBackground->ImagePtr(), oneLayerSize* sizeof(float),
            cudaMemcpyDeviceToDevice));
    CUDA_FAILED(
        cudaMemcpy(ptrImgGPU +oneLayerSize+oneLayerSize, currentImgFloat->ImagePtr(), oneLayerSize* sizeof(float),
            cudaMemcpyDeviceToDevice));

    FREE_FRAME_GPU(currentImgFloat);
    FREE_FRAME_GPU(resDiffImg);
}


void EnginePipeline::UpdateBackground()
{
    if (!_currentImage)
        throw std::runtime_error("[ThermalPipeline::UpdateBackground] Null reference exception");

    float alpha = 1.0 / _settingPipeline->CountImgToBackground; // 1/25 fps

    _imageBackground = _nppFunctions->AddWeighted(_imageBackground, _currentImage, alpha);
}


bool EnginePipeline::GetResultImages(vector<Detection>& resultNms, uint64_t& timeStamp)
{
    try
    {
        FrameGpu<Npp8u>* frame = nullptr;
        auto result = _bufferFrameGpu->Dequeue(&frame);

        if (!result || !frame)
            return false;

        unique_ptr<FrameGpu<Npp8u>> frameUnPrt(frame);

        //TODO: добавить проверку по времени...
        timeStamp = frameUnPrt->Timestamp();

        UpdateCurrentImg(frameUnPrt.get());
        UpdateCurrentTimeStamp(timeStamp);
        UpdateBackground();
        LoadImgToTrt();
        auto resulDoInferenceAsync = _trtEngine->DoInferenceNMSAsync(resultNms);

        if (!resulDoInferenceAsync)
            return false;

        auto resConvertRect = ConverterDetection(resultNms);

        if (!resConvertRect)
            return false;

        return true;
    }
    catch (exception& e)
    {
        _logger->error("[EnginePipeline::GetResultImages]  {}", e.what());
    }
    catch (...)
    {
        _logger->error("[EnginePipeline::GetResultImages]  Unknown exception!");
    }

    return false;
}

EnginePipeline::~EnginePipeline()
{
    FREE_FRAME_GPU(_imageBackground);
    FREE_FRAME_GPU(_currentImage);

    if (_settingPipeline)
        delete _settingPipeline;

    if (_nppFunctions)
        delete _nppFunctions;

    _logger->info("[~ThermalPipeline] Call");
}

std::vector<unsigned char>* EnginePipeline::GetFrame()
{
    return _encoder->Encode(_currentImage);
}

void EnginePipeline::UpdateCurrentImg(FrameGpu<Npp8u>* imageSrc)
{
    if (!imageSrc)
        return;

    FREE_FRAME_GPU(_currentImage);
    auto frameGpuResize = _nppFunctions->ResizeGrayScale(imageSrc, _settingPipeline->WidthImgMl,
                                                         _settingPipeline->HeightImgMl);
    _currentImage = frameGpuResize;
}
