//
// Created by user on 28.11.2024.
//

#include "NppFunction.h"




FrameGpu<Npp8u>* NppFunction::ResizeGrayScale( const FrameGpu<Npp8u>* imsSrc, const int widthNew, const int heightNew)
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


FrameGpu<Npp8u>* NppFunction::RGBToGray(const FrameGpu<Npp8u>* imsSrc)
{
    if(!imsSrc ||  imsSrc->Channel() !=3)
        throw std::runtime_error("[NppFunction::RGBToGray] fail input parameters");

    unsigned char *  retImage= nullptr;
    auto timeStamp = imsSrc->Timestamp();
    auto channel = 1;
    auto width = imsSrc->Width();
    auto height = imsSrc->Height();
    auto nSrStep = imsSrc->Width() * imsSrc->Channel();
    auto nDstStep = width * channel;

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

FrameGpu<float>* NppFunction::AddWeighted( FrameGpu<Npp32f>* imgBackground, const FrameGpu<Npp8u>* imgSrc,const float alpha )
{
    if(!imgBackground ||  imgBackground->Channel() !=1 || !imgSrc ||  imgSrc->Channel() !=1)
        throw std::runtime_error("[NppFunction::RGBToGray] fail input parameters");

    NppiSize oSizeROI = {imgSrc->Width(), imgSrc->Height()};

    auto status = nppiAddWeighted_8u32f_C1IR(
        imgSrc->ImagePtr(),
        imgSrc->Width()*sizeof(Npp8u),
        imgBackground->ImagePtr(),
        imgBackground->GetStep(),
        oSizeROI,
        alpha);

    if (status != NPP_SUCCESS)
    {
        auto name = magic_enum::enum_name(status);
        throw runtime_error("TestConvertToGray failed nppiAddWeighted_8u32f_C1IR " + string(name));
    }

    imgBackground->SetTimestamp(imgSrc->Timestamp());
    return imgBackground;
}

FrameGpu<Npp32f>* NppFunction::AbsDiff(const FrameGpu<Npp32f>* imgBackground, const FrameGpu<Npp32f>* imageDiff)
{
    if(!imgBackground ||  imgBackground->Channel() !=1 || !imageDiff ||  imageDiff->Channel() !=1)
        throw std::runtime_error("[NppFunction::AbsDiff] fail input parameters");

    Npp32f* imagePtr = nullptr;
    auto allSize = imgBackground->GetFulSize();
    CUDA_FAILED(cudaMalloc((void **)(&imagePtr), allSize*sizeof(Npp32f) ));


    auto status = nppiAbsDiff_32f_C1R(
        imgBackground->ImagePtr(),
        imgBackground->GetStep(),
        imageDiff->ImagePtr(),
        imageDiff->GetStep(),
        imagePtr,
        imgBackground->GetStep(),
        NppiSize{ imgBackground->Width(), imgBackground->Height() }
        );

    if (status != NPP_SUCCESS) {
        auto name = magic_enum::enum_name(status);
        throw runtime_error("TestConvertToGray failed nppiAbsDiff_32f_C1R "+ string(name) );
    }

    return  new FrameGpu(
        imagePtr,
        imgBackground->Width(),
        imgBackground->Height(),
        imgBackground->Timestamp(),
        imgBackground->Channel());
}
