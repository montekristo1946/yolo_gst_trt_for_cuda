//
// Created by user on 09.11.2024.
//

#ifndef CUDAYUV_NV12_CUH
#define CUDAYUV_NV12_CUH

#include "CudaUtility.h"
#include "bits/stdint-uintn.h"
#include "cudaVector.h"

class CudaYUV_NV12 {
public:
    static cudaError_t CudaNV12ToRGB( void* srcDev, uchar3* destDev, size_t width, size_t height, size_t srcPitch,cudaStream_t stream );
};



#endif //CUDAYUV_NV12_CUH
