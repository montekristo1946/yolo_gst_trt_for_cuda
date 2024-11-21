//
// Created by user on 11.11.2024.
//

#include "MatConverter.h"



static __global__ void ConvertToFloat32(uchar* srcImage, float* dstImage, int width, int height,size_t step)
{
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if( x >= width )
        return;

    if( y >= height )
        return;

    uchar* pixel = srcImage + y * step + x;

    dstImage[x+y*width] = static_cast<float>(pixel[0]);
}

cudaError_t MatConverter::GrayToFloat32ContinueArr(cv::cuda::GpuMat* srcDev, float* dstDev,cudaStream_t *stream)
{
    if( !srcDev || !dstDev || !stream )
        return cudaErrorInvalidDevicePointer;

    if(srcDev->type() != CV_8UC1 )
    {
        spdlog::error("Fail format input Img {}",srcDev->type() );
        return cudaErrorInvalidValue;
    }

    auto width = srcDev->cols;
    auto height = srcDev->rows;
    auto step = srcDev->step;

    const dim3 blockDim(32,8,1);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

    ConvertToFloat32<<<gridDim, blockDim, 0, *stream>>>(&srcDev->data[0], dstDev, width, height,step);

    return CUDA(cudaGetLastError());

}
