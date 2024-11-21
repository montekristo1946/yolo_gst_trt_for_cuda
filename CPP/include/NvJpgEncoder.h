//
// Created by user on 12.11.2024.
//

#ifndef NVJPGENCODER_H
#define NVJPGENCODER_H
#include <IDispose.h>
#include <nvjpeg.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

using namespace cv;
using namespace std;

class NvJpgEncoder: public IDispose {
public:
    NvJpgEncoder(cudaStream_t* streem);
    ~NvJpgEncoder();
    vector<unsigned char>* Encode(cuda::GpuMat &imageSrc);

private:
    void Init();

    nvjpegHandle_t _nvHandle;
    nvjpegEncoderState_t _nvEncState;
    nvjpegEncoderParams_t _nvEncParams;
    nvjpegChromaSubsampling_t _subsampling ;

    std::vector<unsigned char> * _jpegExport = nullptr;

    cudaStream_t* _stream = nullptr;
};

#endif //NVJPGENCODER_H
