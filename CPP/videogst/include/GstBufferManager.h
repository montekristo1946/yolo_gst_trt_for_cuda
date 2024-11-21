
#ifndef YOLOGSTFORCUDA_GSTBUFFERMANAGER_H
#define YOLOGSTFORCUDA_GSTBUFFERMANAGER_H

#include <BufferFrameGpu.h>
#include <gstbuffer.h>
#include <cstdint>
#include <nvbufsurface.h>
#include "GstUtility.h"
#include <opencv2/core.hpp>
#include <spdlog/spdlog.h>

#include "CudaYUV_NV12.h"
#include "common/IDispose.h"
#include <magic_enum.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>>
#include "CudaUtility.h"


class GstBufferManager: public IDispose {

public:
    GstBufferManager(BufferFrameGpu* bufferFrameGpu,cudaStream_t *stream) ;

    bool Enqueue( GstBuffer* buffer, GstCaps* caps );


    ~GstBufferManager();

protected:

    BufferFrameGpu* _bufferFrameGpu;

    bool _isShowFirstDebugMessage = true;

    void ShowFirstDebugMessage(GstCaps* gst_caps);
    cuda::GpuMat* CreateImage(const NvBufSurfaceParams& nvBufSurfaceParams);

    CudaYUV_NV12 _cudaYUV_NV12;

    cudaStream_t *_stream;
    shared_ptr<logger> _logger = get("MainLogger");
};


#endif
