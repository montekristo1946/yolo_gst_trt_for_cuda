//
// Created by user on 28.11.2024.
//

#include "NppFunction.h"

#include <nppi_color_conversion.h>


FrameGpu* NppFunction::ResizeGrayScale( const FrameGpu* imsSrc, const int widthNew, const int heightNew)
{
    if(!imsSrc || widthNew <= 0 || heightNew <= 0 || imsSrc->Channel() !=1)
        throw std::runtime_error("[NppFunction::ResizeGrayScale] Null reference exception");

    auto interpolationMode = NPPI_INTER_LINEAR;
    unsigned char *  retImage= nullptr;
    auto timeStamp = imsSrc->Timestamp();
    auto channel = imsSrc->Channel();

    auto allSizeDst = widthNew*heightNew*imsSrc->Channel();
    CUDA_FAILED(cudaMalloc(&retImage, allSizeDst));


    NppiSize oSrcSize = { imsSrc->Width(), imsSrc->Height() };
    NppiRect oSrcRectROI = { 0, 0, imsSrc->Width(), imsSrc->Height()};

    NppiSize oDstSize = { widthNew, heightNew};
    NppiRect oDstRectROI = { 0, 0, widthNew, heightNew};

    auto pitchSrc = imsSrc->Width(); //only grayscale
    auto pitchDst = widthNew;

    auto status = nppiResize_8u_C1R(
        imsSrc->ImagePtr(),
        pitchSrc,
        oSrcSize,
        oSrcRectROI,
        retImage,
        pitchDst,
        oDstSize,
        oDstRectROI,
        interpolationMode
        );


    if (status != NPP_SUCCESS) {
        auto name = magic_enum::enum_name(status);
        throw runtime_error("[NppFunction::ResizeGrayScale] nppiResize_8u_C1R false" + string(name));
    }

    return  new FrameGpu(retImage, widthNew, heightNew, timeStamp, channel);
}


FrameGpu* NppFunction::RGBToGray(const FrameGpu* imsSrc)
{
    if(!imsSrc ||  imsSrc->Channel() !=3)
        throw std::runtime_error("[NppFunction::RGBToGray] fail input parameters");

    unsigned char *  retImage= nullptr;
    auto timeStamp = imsSrc->Timestamp();
    auto channel = 1;
    auto width = imsSrc->Width();
    auto height = imsSrc->Height();
    auto nSrStep = imsSrc->Width() *imsSrc->Height() * imsSrc->Channel();
    auto nDstStep = width;

    auto allSizeDst = imsSrc->Width()*imsSrc->Height()*channel;
    CUDA_FAILED(cudaMalloc(&retImage, allSizeDst));


    NppiSize oSizeROI = {width, height};
    auto status = nppiRGBToGray_8u_C3C1R(
    imsSrc->ImagePtr(),
        nSrStep,
        retImage,
        nDstStep,
        oSizeROI);

    if (status != NPP_SUCCESS) {
        auto name = magic_enum::enum_name(status);
        throw runtime_error("TestConvertToGray failed nppiRGBToGray_8u_C3C1R "+ string(name) );
    }

    return  new FrameGpu(retImage, width, height, timeStamp, channel);
}
