#ifndef FRAMEGPU_H
#define FRAMEGPU_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>

#include "CudaUtility.h"

using namespace cv;

class FrameGpu
{
public:
    FrameGpu() = default;

    FrameGpu(unsigned char* image, const int width,const int height,const uint64_t timestamp, const int channel)
    {
        if(!image || width <= 0 || height <= 0 || timestamp <= 0 || channel <= 0)
            throw std::runtime_error("[FrameGpu::Ctr char, int, int, uint64_t, int] Fail input parameters");

        _images = image;
        _width = width;
        _height = height;
        _timestamp = timestamp;
        _channel = channel;
    }




    ~FrameGpu()
    {
        if ( _images)
        {
            CUDA_FAILED(cudaFree(_images));
        }

    };

    uint64_t Timestamp() const { return _timestamp; }
    int Width() const { return _width; }
    int Height() const { return _height; }
    int Channel() const { return _channel; }
    unsigned char* ImagePtr()  const { return _images; }

    unsigned int GetFulSize() const { return _width * _height * _channel; }

private:
    // cuda::GpuMat* _images;
    unsigned char * _images = nullptr;
    uint64_t _timestamp = 0;
    int _width = -1;
    int _height = -1;
    int _channel = -1;

};

#endif //FRAMEGPU_H
