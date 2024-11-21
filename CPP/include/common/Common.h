#ifndef TENSORRTTOOLS_COMMON_H
#define TENSORRTTOOLS_COMMON_H
#include <vector>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "cuda_runtime_api.h"
#include "IDispose.h"

using namespace std;
using namespace cv;

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





inline bool FreeMatGPU(cuda::GpuMat* images)
{
    if (!images)
        return false;

    cudaFree(images->data);
    images = nullptr;
    return true;
}

#define FREE_MATGPU(x)	FreeMatGPU((x))

#endif //TENSORRTTOOLS_COMMON_H
