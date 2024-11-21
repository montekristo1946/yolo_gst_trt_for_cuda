//
// Created by user on 11.11.2024.
//

#ifndef MATCONVERTER_CUH
#define MATCONVERTER_CUH
#include <opencv2/core/cuda.hpp>
#include <spdlog/spdlog.h>
#include "driver_types.h"
#include <CudaUtility.h>


class MatConverter {
public:
    MatConverter(){};
    cudaError_t GrayToFloat32ContinueArr(cv::cuda::GpuMat* arrInput, float* output, cudaStream_t *stream);
};



#endif //MATCONVERTER_CUH
