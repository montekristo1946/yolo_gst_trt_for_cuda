#ifndef FRAMEGPU_H
#define FRAMEGPU_H

#include "CudaUtility.h"


template <typename T>
class FrameGpu
{
public:
    FrameGpu() = default;

    FrameGpu(T* image, const int width,const int height,const uint64_t timestamp, const int channel)
    {
        if(!image || width <= 0 || height <= 0 || timestamp <= 0 || channel <= 0)
            throw std::runtime_error("[FrameGpu::Ctr char, int, int, uint64_t, int] Fail input parameters ptr:" +
            std::to_string(!image != false) +
            " width:" +  std::to_string(width) +
            " height:"  + std::to_string(height) +
            " timestamp:" + std::to_string(timestamp) +
            " channel:" + std::to_string(channel));

        _images = image;
        _width = width;
        _height = height;
        _timestamp = timestamp;
        _channels = channel;
    }

    static FrameGpu* CreateNew( const int width, const int height,  const int channel)
    {
        T* destinationImage = nullptr;
        const auto destinationSize = width * height*channel;
        CUDA_FAILED(cudaMalloc(reinterpret_cast<void**>(&destinationImage), destinationSize*sizeof(T)));
        uint64_t timestamp = 1;
        return new FrameGpu(destinationImage, width, height, timestamp,channel);
    }


    ~FrameGpu()
    {
        if ( _images)
        {
            CUDA_FAILED(cudaFree(_images));
        }

    }

    int GetStep() const { return _width *_channels* sizeof(T); }

    uint64_t Timestamp() const { return _timestamp; }
    int Width() const { return _width; }
    int Height() const { return _height; }
    int Channels() const { return _channels; }
    T* ImagePtr()  const { return _images; }

    unsigned int GetFulSize() const { return _width * _height * _channels; }
    void SetTimestamp(uint64_t timestamp)
    {
        _timestamp = timestamp;
    };

private:
    T * _images = nullptr;
    uint64_t _timestamp = 0;
    int _width = -1;
    int _height = -1;
    int _channels = -1;

    FrameGpu(T* image, const int width,const int height ,const int channel)
    {
        if(!image || width <= 0 || height <= 0 || channel <= 0)
            throw std::runtime_error("[FrameGpu::Ctr char, int, int, uint64_t, int] Fail input parameters");

        _images = image;
        _width = width;
        _height = height;
        _timestamp = 0;
        _channels = channel;
    }

};

#endif //FRAMEGPU_H
