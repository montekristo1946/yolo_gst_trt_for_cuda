#ifndef NPPFUNCTION_H
#define NPPFUNCTION_H

#include "FrameGpu.h"
#include "IDispose.h"
#include "CudaUtility.h"
#include <nppi_geometry_transforms.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_color_conversion.h>

class NppFunction : public IDispose
{
public:
    NppFunction()
    {
        info("[NppFunction::Ctr]  Init  NppFunction");
    }

    FrameGpu<Npp8u>* ResizeGrayScale( const FrameGpu<Npp8u>* imsSrc, const int widthNew, const int heightNew);
    FrameGpu<Npp8u>* RGBToGray( const  FrameGpu<Npp8u>* imsSrc);
    FrameGpu<Npp32f>* AddWeighted( FrameGpu<Npp32f>* imgBackground, const FrameGpu<Npp8u>* imgSrc,const float alpha = 0.04);
    FrameGpu<Npp32f> *AbsDiff(const FrameGpu<Npp32f>* imgBackground, const FrameGpu<Npp32f>* imageDiff);

};


#endif //NPPFUNCTION_H
