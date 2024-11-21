#ifndef FRAMEGPU_H
#define FRAMEGPU_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;

class FrameGpu
{
public:
    FrameGpu() = default;

    FrameGpu(cuda::GpuMat* images, const uint64_t timestamp)
    {
        _images = images;
        _timestamp = timestamp;
    }


    ~FrameGpu()
    {
        if (_images == nullptr)
            return;

        if ( _images->data != nullptr)
        {
            cudaFree(_images->data);
        }
        delete _images;
        _images = nullptr;
    };

    cuda::GpuMat* GetImages() const { return _images; }
    uint64_t GetTimestamp() const { return _timestamp; }

private:
    cuda::GpuMat* _images;
    uint64_t _timestamp = 0;
};

#endif //FRAMEGPU_H
