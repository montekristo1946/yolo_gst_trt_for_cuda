//
// Created by user on 28.11.2024.
//

#include "NppFunction.h"




FrameGpu<Npp8u>* NppFunction::ResizeGrayScale(const FrameGpu<Npp8u>* sourceImage, int newWidth, int newHeight)
{
    if (!sourceImage || newWidth <= 0 || newHeight <= 0 || sourceImage->Channel() != 1)
        throw std::invalid_argument("[NppFunction::ResizeGrayScale] Invalid input parameters");

    NppiSize srcSize = { sourceImage->Width(), sourceImage->Height() };
    NppiRect srcRectROI = { 0, 0, sourceImage->Width(), sourceImage->Height() };

    NppiSize dstSize = { newWidth, newHeight };
    NppiRect dstRectROI = { 0, 0, newWidth, newHeight };

    int pitchSrc = sourceImage->Width(); // only grayscale
    int pitchDst = newWidth;

    unsigned char* resizedImage = nullptr;
    CUDA_FAILED(cudaMalloc(&resizedImage, newWidth * newHeight));

    NppStatus status = nppiResize_8u_C1R(
        sourceImage->ImagePtr(),
        pitchSrc,
        srcSize,
        srcRectROI,
        resizedImage,
        pitchDst,
        dstSize,
        dstRectROI,
        NPPI_INTER_LINEAR
    );

    if (status != NPP_SUCCESS) {
        std::string err = "[NppFunction::ResizeGrayScale] nppiResize_8u_C1R failed: " + std::string(magic_enum::enum_name(status));
        cudaFree(resizedImage);
        throw std::runtime_error(err);
    }

    return new FrameGpu(resizedImage, newWidth, newHeight, sourceImage->Timestamp(), sourceImage->Channel());
}


FrameGpu<Npp8u>* NppFunction::RGBToGray(const FrameGpu<Npp8u>* sourceImage)
{
    if (!sourceImage || sourceImage->Channel() != 3)
        throw std::invalid_argument("[NppFunction::RGBToGray] Invalid input parameters");

    Npp8u* destinationImage = nullptr;
    const auto width = sourceImage->Width();
    const auto height = sourceImage->Height();
    const auto sourceStep = sourceImage->Width() * sourceImage->Channel();
    const auto destinationStep = width;

    const auto destinationSize = width * height;
    CUDA_FAILED(cudaMalloc(&destinationImage, destinationSize));

    NppiSize roiSize = {width, height};
    const auto status = nppiRGBToGray_8u_C3C1R(
        sourceImage->ImagePtr(),
        sourceStep,
        destinationImage,
        destinationStep,
        roiSize);

    if (status != NPP_SUCCESS)
        throw std::runtime_error("[NppFunction::RGBToGray] nppiRGBToGray_8u_C3C1R failed");

    return new FrameGpu(destinationImage, width, height, sourceImage->Timestamp(), 1);
}



FrameGpu<float>* NppFunction::AddWeighted(FrameGpu<Npp32f>* backgroundImg, const FrameGpu<Npp8u>* sourceImg, const float alpha)
{
    if (!backgroundImg || backgroundImg->Channel() != 1 || !sourceImg || sourceImg->Channel() != 1)
        throw std::invalid_argument("[NppFunction::AddWeighted] Invalid input parameters");

    NppiSize roiSize = {sourceImg->Width(), sourceImg->Height()};

    auto status = nppiAddWeighted_8u32f_C1IR(
        sourceImg->ImagePtr(),
        sourceImg->GetStep(),
        backgroundImg->ImagePtr(),
        backgroundImg->GetStep(),
        roiSize,
        alpha);

    if (status != NPP_SUCCESS)
    {
        auto statusName = magic_enum::enum_name(status);
        throw std::runtime_error("[NppFunction::AddWeighted] nppiAddWeighted_8u32f_C1IR failed: " + std::string(statusName));
    }

    backgroundImg->SetTimestamp(sourceImg->Timestamp());
    return backgroundImg;
}

FrameGpu<Npp32f>* NppFunction::AbsDiff(const FrameGpu<Npp32f>* imgBackground, const FrameGpu<Npp32f>* imageDiff)
{
    if(!imgBackground ||  imgBackground->Channel() !=1 || !imageDiff ||  imageDiff->Channel() !=1)
        throw std::invalid_argument("[NppFunction::AbsDiff] fail input parameters");

    Npp32f* imagePtr = nullptr;
    auto allSize = imgBackground->GetFulSize();
    CUDA_FAILED(cudaMalloc(reinterpret_cast<void**>(&imagePtr), allSize*sizeof(Npp32f) ));


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
