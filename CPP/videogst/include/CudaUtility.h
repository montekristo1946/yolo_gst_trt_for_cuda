//
// Created by user on 09.11.2024.
//

#ifndef CUDAUTILITY_H
#define CUDAUTILITY_H
#include <nvjpeg.h>
#include <spdlog/spdlog.h>

#include "magic_enum.hpp"


/**
 * cudaCheckError
 * @ingroup cudaError
 */
inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line)
{
    if (retval == cudaSuccess)
    {
        return cudaSuccess;
    }

    const char* errorName = cudaGetErrorName(retval);
    spdlog::error("[cuda] errorName: {}; text: {}; error: {}; ", errorName, txt, cudaGetErrorString(retval));
    spdlog::error("[cuda] {}; {}", file, line);

    throw std::runtime_error("[cudaCheckError] failed call CUDA");
}


/**
 * Execute a CUDA call and print out any errors
 * @return the original cudaError_t result
 * @ingroup cudaError
 */
#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)

/**
 * Evaluates to true on failure
 * @ingroup cudaError
 */
#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)


inline __device__ __host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }


inline nvjpegStatus_t cudaCheckNVJPEG(nvjpegStatus_t retval, const char* txt, const char* file, int line)
{
    if (retval == NVJPEG_STATUS_SUCCESS)
    {
        return NVJPEG_STATUS_SUCCESS;
    }

    auto errorValue = magic_enum::enum_name(retval);
    spdlog::error("[cuda] errorName: {}; text: {}; file: {};  line: {};", errorValue, txt,file, line );


    throw std::runtime_error("[cudaCheckError] failed call CUDA");
}

#define CHECK_NVJPEG(x) cudaCheckNVJPEG((x), #x, __FILE__, __LINE__)

#endif //CUDAUTILITY_H
