#include "include/CudaYUV_NV12.h"


#define COLOR_COMPONENT_BIT_SIZE        10
#define COLOR_COMPONENT_MASK            0x3FF
typedef unsigned char uchar;


//-----------------------------------------------------------------------------------
// YUV to RGB colorspace conversion
//-----------------------------------------------------------------------------------
static inline __device__ float clamp(float x) { return fminf(fmaxf(x, 0.0f), 255.0f); }

// YUV2RGB
template <typename T>
static inline __device__ T YUV2RGB(const uint3& yuvi)
{
    const float luma = float(yuvi.x);
    const float u = float(yuvi.y) - 512.0f;
    const float v = float(yuvi.z) - 512.0f;
    const float s = 1.0f / 1024.0f * 255.0f; // TODO clamp for uchar output?

    return make_vec<T>(clamp((luma + 1.402f * v) * s),
                       clamp((luma - 0.344f * u - 0.714f * v) * s),
                       clamp((luma + 1.772f * u) * s), 255);
}

//-----------------------------------------------------------------------------------
// NV12 to RGB
//-----------------------------------------------------------------------------------
template <typename T>
__global__ void NV12ToRGB(uint32_t* srcImage,
                          size_t nSourcePitch,
                          T* dstImage,
                          size_t nDestPitch,
                          uint32_t width,
                          uint32_t height)
{
    int x, y;
    uint32_t yuv101010Pel[2];
    uint32_t processingPitch = nSourcePitch;
    uint8_t* srcImageU8 = (uint8_t*)srcImage;

    // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
    x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
    y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

    // Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
    // if we move to texture we could read 4 luminance values
    yuv101010Pel[0] = (srcImageU8[y * processingPitch + x]) << 2;
    yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]) << 2;

    uint32_t chromaOffset = processingPitch * height;
    int y_chroma = y >> 1;

    if (y & 1) // odd scanline ?
    {
        uint32_t chromaCb;
        uint32_t chromaCr;

        chromaCb = srcImageU8[chromaOffset + y_chroma * processingPitch + x];
        chromaCr = srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1];

        if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
        {
            chromaCb = (chromaCb + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x] + 1) >> 1;
            chromaCr = (chromaCr + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x + 1] + 1) >> 1;
        }

        yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
        yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));
        yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
    }
    else
    {
        yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x] << (
            COLOR_COMPONENT_BIT_SIZE + 2));
        yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((
            COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x] << (
            COLOR_COMPONENT_BIT_SIZE + 2));
        yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((
            COLOR_COMPONENT_BIT_SIZE << 1) + 2));
    }

    // this steps performs the color conversion
    const uint3 yuvi_0 = make_uint3((yuv101010Pel[0] & COLOR_COMPONENT_MASK),
                                    ((yuv101010Pel[0] >> COLOR_COMPONENT_BIT_SIZE) & COLOR_COMPONENT_MASK),
                                    ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK));

    const uint3 yuvi_1 = make_uint3((yuv101010Pel[1] & COLOR_COMPONENT_MASK),
                                    ((yuv101010Pel[1] >> COLOR_COMPONENT_BIT_SIZE) & COLOR_COMPONENT_MASK),
                                    ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK));

    // YUV to RGB transformation conversion
    dstImage[y * width + x] = YUV2RGB<T>(yuvi_0);
    dstImage[y * width + x + 1] = YUV2RGB<T>(yuvi_1);
}

template <typename T>
static cudaError_t launchNV12ToRGB(void* srcDev,
                                   T* dstDev,
                                   size_t width,
                                   size_t height,
                                   size_t srcPitch,
                                   cudaStream_t stream)
{
    if (!srcDev || !dstDev)
        return cudaErrorInvalidDevicePointer;

    if (width == 0 || height == 0)
        return cudaErrorInvalidValue;


    const size_t dstPitch = width * sizeof(T);

    const dim3 blockDim(32, 8, 1);
    const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y), 1);

    NV12ToRGB<T><<<gridDim, blockDim, 0, stream>>>((uint32_t*)srcDev, srcPitch, dstDev, dstPitch, width, height);

    return CUDA(cudaGetLastError());
}


cudaError_t CudaYUV_NV12::CudaNV12ToRGB(void* srcDev, uchar3* destDev, size_t width, size_t height, size_t srcPitch,
                                        cudaStream_t stream)
{
    return launchNV12ToRGB<uchar3>(srcDev, destDev, width, height, srcPitch, stream);
}
