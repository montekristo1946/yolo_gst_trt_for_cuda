//
// Created by user on 12.11.2024.
//

#ifndef NVJPGENCODER_H
#define NVJPGENCODER_H
#include <IDispose.h>
#include <nppdefs.h>
#include <nvjpeg.h>

#include "FrameGpu.h"

using namespace std;

class NvJpgEncoder: public IDispose {
public:
    NvJpgEncoder(cudaStream_t* streem);
    ~NvJpgEncoder();
    vector<unsigned char>* Encode(const FrameGpu<Npp8u>* imageSrc);

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
