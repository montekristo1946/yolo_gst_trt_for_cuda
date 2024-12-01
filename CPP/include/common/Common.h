#ifndef TENSORRTTOOLS_COMMON_H
#define TENSORRTTOOLS_COMMON_H
#include <vector>
#include <iostream>
#include "cuda_runtime_api.h"
#include "FrameGpu.h"

using namespace std;

struct LayerSize
{
public:
    LayerSize()
    {
        BatchSize = -1;
        Channel = -1;
        Width = -1;
        Height = -1;
    }

    LayerSize(int batchSize, int channel, int width, int height)
    {
        BatchSize = batchSize;
        Channel = channel;
        Width = width;
        Height = height;
    }

    int BatchSize;
    int Channel;
    int Width;
    int Height;
};




template <typename T>
inline bool FreeFrameGpu(FrameGpu<T>* images)
{
    if (!images)
        return false;

    delete images;
    images = nullptr;

    return true;
}

#define FREE_FRAME_GPU(x)	FreeFrameGpu((x))

#endif //TENSORRTTOOLS_COMMON_H
