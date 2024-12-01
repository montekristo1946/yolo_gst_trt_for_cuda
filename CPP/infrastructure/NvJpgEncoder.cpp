#include "NvJpgEncoder.h"

#include "CudaUtility.h"


NvJpgEncoder::NvJpgEncoder(cudaStream_t* streem)
{
    info("[NvJpgEncoder::NvJpgEncoder] Call constructor");

    if (!streem)
        throw std::runtime_error("[NvJpgEncoder::NvJpgEncoder] Null reference exception()");

    this->_stream = streem;

    Init();
}

NvJpgEncoder::~NvJpgEncoder()
{
    info("[NvJpgEncoder::~NvJpgEncoder] Call destructor");
    if (_nvEncParams)
        CHECK_NVJPEG(nvjpegEncoderParamsDestroy(_nvEncParams));

    if (_nvEncState)
        CHECK_NVJPEG(nvjpegEncoderStateDestroy(_nvEncState));

    if (_nvHandle)
        CHECK_NVJPEG(nvjpegDestroy(_nvHandle));

}

vector<unsigned char>* NvJpgEncoder::Encode(const FrameGpu<Npp8u>* imageSrc)
{
    if (!imageSrc || imageSrc->Channels() != 1)
        throw std::runtime_error("[NvJpgEncoder::Encode] Null reference exception()");

    if(_jpegExport)
    {
        delete _jpegExport;
        _jpegExport = nullptr;
    }

    nvjpegImage_t nvImage;
    auto width = imageSrc->Width();
    auto height = imageSrc->Height();
    auto newChannel = 3;
    auto step = imageSrc->GetStep();

    for (int i = 0; i < newChannel; i++)
    {
        nvImage.channel[i] = imageSrc->ImagePtr();
        nvImage.pitch[i] = step;
    }

    // Compress image
    CHECK_NVJPEG(nvjpegEncodeImage(_nvHandle, _nvEncState, _nvEncParams,
        &nvImage, NVJPEG_INPUT_RGB, width, height, *_stream));

    // get compressed stream size
    size_t length;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(_nvHandle, _nvEncState, NULL, &length, *_stream));
    // get stream itself
    CUDA_FAILED(cudaStreamSynchronize(*_stream));

    _jpegExport = new  std::vector<unsigned char>(length);

    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(_nvHandle, _nvEncState, _jpegExport->data(), &length, 0));

    // write stream to file
    CUDA_FAILED(cudaStreamSynchronize(*_stream));

    return _jpegExport;
}

void NvJpgEncoder::Init()
{
    info("[NvJpgEncoder::Init] call init");

    int quality = 90;
    _subsampling = NVJPEG_CSS_410;

    CHECK_NVJPEG(nvjpegCreateSimple(&_nvHandle));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(_nvHandle, &_nvEncState, *_stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(_nvHandle, &_nvEncParams, *_stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(_nvEncParams, quality, NULL));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(_nvEncParams, _subsampling, NULL));
}
