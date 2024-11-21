//
// Created by user on 15.11.2024.
//

#ifndef CUDASTREAM_H
#define CUDASTREAM_H


#include "cudaVector.h"
#include "CudaUtility.h"

class CudaStream: public IDispose{
public:
    CudaStream()
    {
        CUDA_FAILED(cudaStreamCreate(&_stream));
    }

    ~CudaStream()
    {
        _logger->info("[~CudaStream] Call");
        cudaStreamDestroy(_stream);
    }

    cudaStream_t * GetStream()
    {
        if(!_stream)
            throw std::runtime_error("[CudaStream::GetStream] Null reference exception()");
        return &_stream;
    }

private:
    cudaStream_t _stream = nullptr;
    shared_ptr<logger> _logger = get("MainLogger");
};



#endif //СUDASTREAM_H
