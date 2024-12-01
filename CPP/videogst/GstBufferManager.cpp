#include "GstBufferManager.h"


GstBufferManager::GstBufferManager(BufferFrameGpu* bufferFrameGpu, cudaStream_t* stream)
{
    if (!bufferFrameGpu || !stream)
        throw std::runtime_error(
            "[GstBufferManager::GstBufferManager] Null reference exception {name: BufferFrameGpu} {name: cudaStream_t}");


    _bufferFrameGpu = bufferFrameGpu;
    _stream = stream;
}


FrameGpu<Npp8u>* GstBufferManager::CreateImage(const NvBufSurfaceParams& nvBufSurfaceParams)
{
    auto width = nvBufSurfaceParams.width;
    auto height = nvBufSurfaceParams.height;
    auto pitch = nvBufSurfaceParams.pitch;
    auto colorFormat = nvBufSurfaceParams.colorFormat;

    if (colorFormat != NVBUF_COLOR_FORMAT_NV12 && colorFormat != NVBUF_COLOR_FORMAT_NV12_709)
    {
        auto name = magic_enum::enum_name(colorFormat);
        throw std::runtime_error("[GstBufferManager::CreateImage] Color format not supported: " + string(name));
    }

    if (width <= 0 || height <= 0 || pitch <= 0)
        throw std::runtime_error("[GstBufferManager::CreateImage] Invalid image size");

    const size_t rgbBufferSize = width * height * sizeof(uchar3) ;
    void* converImage = NULL;

    CUDA_FAILED(cudaMallocAsync(&converImage, rgbBufferSize,*_stream));
    CUDA_FAILED(
        _cudaYUV_NV12.CudaNV12ToRGB(nvBufSurfaceParams.dataPtr, static_cast<uchar3*>(converImage), width, height, pitch,*_stream));

    auto channel = 3;
    auto rgbFrame = new FrameGpu(static_cast<Npp8u*>(converImage), width, height, 1, channel);
    auto imtGray = _nppFunctions->RGBToGray(rgbFrame);

    delete rgbFrame;
    // CUDA_FAILED(cudaFree(converImage));

    // auto imgSrc = cuda::GpuMat(height, width, CV_8UC3, converImage);
    // auto imtGray = new cuda::GpuMat();
    // cuda::cvtColor(imgSrc, *imtGray, COLOR_BGR2GRAY);
    //
    // cudaFree(converImage);
    return imtGray;
}

bool GstBufferManager::Enqueue(GstBuffer* gstBuffer, GstCaps* gstCaps)
{
    if (!gstBuffer || !gstCaps)
        return false;

    if (_isShowFirstDebugMessage)
    {
        ShowFirstDebugMessage(gstCaps);
        _isShowFirstDebugMessage = false;
    }

    uint64_t timestamp = 0;


    GstMapInfo map;

    if (!gst_buffer_map(gstBuffer, &map, GST_MAP_READ))
    {
        warn("[GstBufferManager::Enqueue] -- failed to map gstreamer buffer memory");
        return false;
    }

    NvBufSurface* infoData = (NvBufSurface*)map.data;

    if (!infoData)
    {
        warn("[GstBufferManager::Enqueue] --  info_data had NULL data pointer...");
        return false;
    }

    if (GST_BUFFER_DTS_IS_VALID(gstBuffer) || GST_BUFFER_PTS_IS_VALID(gstBuffer))
    {
        auto timestampTemp = GST_BUFFER_DTS_OR_PTS(gstBuffer);
        timestamp  =  timestampTemp>0 ? timestampTemp : 1;
    }

    auto frame = CreateImage(infoData->surfaceList[0]);
    frame->SetTimestamp(timestamp);
    _bufferFrameGpu->Enqueue(frame);

    gst_buffer_unmap(gstBuffer, &map);

    return true;
}


GstBufferManager::~GstBufferManager()
{
    _logger->info("[~GstBufferManager] Call");
    if (_nppFunctions)
        delete _nppFunctions;
}

void GstBufferManager::ShowFirstDebugMessage(GstCaps* gst_caps)
{
    if (gst_caps == nullptr)
        return;

    info("[GstBufferManager::ShowFirstDebugMessage] gst_caps:  {}", gst_caps_to_string(gst_caps));
}
