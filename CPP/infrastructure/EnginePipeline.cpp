#include "EnginePipeline.h"


EnginePipeline::EnginePipeline(TRTEngine* trtEngine, BufferFrameGpu* bufferFrameGpu,
                               GstBufferManager* gstBufferManager, GstDecoder* gstDecoder, cudaStream_t* streem,
                               SettingPipeline* settingPipeline,NvJpgEncoder* encoder)
{
    if (!trtEngine || !bufferFrameGpu || !gstBufferManager || !gstDecoder || !streem || !encoder)
        throw std::runtime_error("[ThermalPipeline::ThermalPipeline] Null reference exception");

    _trtEngine = trtEngine;
    _bufferFrameGpu = bufferFrameGpu;
    _gstBufferManager = gstBufferManager;
    _gstDecoder = gstDecoder;
    _streem = streem;
    _settingPipeline = settingPipeline;
    _matConverter = new MatConverter();
    _encoder = encoder;
}

bool EnginePipeline::StartPipeline(string connectCamera)
{
    try
    {
        _gstDecoder->InitPipeline(connectCamera);
        _gstDecoder->Open();

        return true;
    }
    catch (exception& e)
    {
        _logger->error("[ThermalPipeline::StartPipeline]  {}", e.what());
    }
    catch (...)
    {
        _logger->error("[ThermalPipeline::StartPipeline]  Unknown exception!");
    }

    return false;
}

bool EnginePipeline::ConverterDetection( vector<Detection>& vector)
{
    for(auto& det : vector)
    {
        det.BBox[2] = det.BBox[2] / _settingPipeline->WidthImgMl; //width
        det.BBox[3] = det.BBox[3] / _settingPipeline->HeightImgMl; //heigt
        det.BBox[0] = det.BBox[0] / _settingPipeline->WidthImgMl ; //x center
        det.BBox[1] = det.BBox[1] / _settingPipeline->HeightImgMl ; //y center
    }
    return true;
}

void EnginePipeline::UpdateCurrentTimeStamp(uint64_t& timeStamp)
{
    _currentTimeStamp = timeStamp;
}

void EnginePipeline::LoadImgToTrt()
{
    if(!_difImage || !_imageBackground || !_currentImage)
        throw std::runtime_error("[ThermalPipeline::LoadImgToTrt] Null reference exception");

    float* ptrImgGPU = static_cast<float*>(_trtEngine->_buffers[0]);
    auto oneLayerSize = _settingPipeline->WidthImgMl * _settingPipeline->HeightImgMl;
    CUDA_FAILED(_matConverter->GrayToFloat32ContinueArr(_difImage, ptrImgGPU, _streem));
    CUDA_FAILED(_matConverter->GrayToFloat32ContinueArr(_imageBackground, ptrImgGPU+oneLayerSize, _streem));
    CUDA_FAILED(_matConverter->GrayToFloat32ContinueArr(_currentImage, ptrImgGPU+oneLayerSize+oneLayerSize, _streem));
}

void EnginePipeline::UpdateBackground()
{
    if(!_currentImage)
        throw std::runtime_error("[ThermalPipeline::UpdateBackground] Null reference exception");

    float alpha = 1.0 / _settingPipeline->CountImgToBackground; // 1/25 fps
    float beta = 1.0 - alpha;
    double gamma = 0.0;
    cuda::GpuMat* newBaground = new cuda::GpuMat();
    cuda::addWeighted(*_currentImage, alpha, *_imageBackground, beta, gamma, *newBaground, -1, (cuda::Stream&)_streem);
    FREE_MATGPU(_imageBackground);
    _imageBackground = newBaground;
}

void EnginePipeline::UpdateDiffImg()
{
    if(!_currentImage)
        throw std::runtime_error("[ThermalPipeline::UpdateBackground] Null reference exception");
    FREE_MATGPU(_difImage);
    _difImage = new cuda::GpuMat(_currentImage->rows, _currentImage->cols, _currentImage->type());
    cuda::absdiff(*_currentImage, *_imageBackground, *_difImage, (cuda::Stream&)_streem);
}

cuda::GpuMat* EnginePipeline::ResizeImages(cuda::GpuMat* imageSrc)
{

    cuda::GpuMat* imageResize = new cuda::GpuMat();
    cuda::resize(*imageSrc, *imageResize,
                 Size(_settingPipeline->WidthImgMl, _settingPipeline->HeightImgMl),
                 0, 0, INTER_LINEAR,
                 (cuda::Stream&)_streem);
    return imageResize;
}

bool EnginePipeline::GetResultImages( vector<Detection>& resultNms,uint64_t &timeStamp)
{
    try
    {
        FrameGpu* frame;
        auto result = _bufferFrameGpu->Dequeue(&frame);


        if (!result || !frame)
            return false;

        unique_ptr<FrameGpu> frameUnPrt(frame);

        auto imageSrc = frameUnPrt->GetImages();


        //TODO: добавить проверку по времени...
        timeStamp = frameUnPrt->GetTimestamp();

        // unique_ptr<cuda::GpuMat> imageResizeUnPrt(ResizeImages(imageSrc));

        UpdateCurrentImg(imageSrc);
        UpdateCurrentTimeStamp(timeStamp);

        if (!_imageBackground)
        {
            _imageBackground = new cuda::GpuMat(_currentImage->rows, _currentImage->cols, _currentImage->type());
            _currentImage->copyTo(*_imageBackground);
            return false;
        }

        UpdateBackground();

        UpdateDiffImg();

        LoadImgToTrt();

        auto resulDoInferenceAsync = _trtEngine->DoInferenceNMSAsync(resultNms);

        if (!resulDoInferenceAsync)
            return false;



        auto resConvertRect = ConverterDetection(resultNms);
        if (!resConvertRect)
            return false;

        // imageResizeUnPrt->download(imageHost);
        // std::copy(difImageHost.data, difImageHost.end, std::back_inserter(_imagesExport));

        // printf("resultNms size %d\n", difImageHost.isContinuous());
        // imagesExport = _imagesExport;


        // printf("resultNms size %i\n", resultNms.size());
        //
        // Mat difImageHost;
        // _difImage->download(difImageHost);
        // imshow("difImageHost", difImageHost);
        //
        // Mat imageBackgroundHost;
        // _imageBackground->download(imageBackgroundHost);
        // imshow("imageBackgroundHost", imageBackgroundHost);
        // waitKey(1);
        return true;
    }
    catch (exception& e)
    {
        _logger->error("[ThermalPipeline::GetResultImages]  {}", e.what());
    }
    catch (...)
    {
        _logger->error("[ThermalPipeline::GetResultImages]  Unknown exception!");
    }

    return false;
}

EnginePipeline::~EnginePipeline()
{
    if (_imageBackground)
        FREE_MATGPU(_imageBackground);

    if(_difImage)
        FREE_MATGPU(_difImage);

    if (_currentImage)
        FREE_MATGPU(_currentImage);


    if (!_matConverter)
        delete _matConverter;

    if(!_settingPipeline)
        delete _settingPipeline;

    _logger->info("[~ThermalPipeline] Call");
}

std::vector<unsigned char>* EnginePipeline::GetFrame()
{
    return _encoder->Encode(*_currentImage);
}

void EnginePipeline::UpdateCurrentImg(cuda::GpuMat* imageSrc)
{
    FREE_MATGPU(_currentImage);
    _currentImage = ResizeImages(imageSrc);
}
