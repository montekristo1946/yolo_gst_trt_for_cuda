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

    static FrameGpu<Npp8u>* ResizeGrayScale( const FrameGpu<Npp8u>* imsSrc, int widthNew, int heightNew);
    static FrameGpu<Npp8u>* RGBToGray( const  FrameGpu<Npp8u>* imsSrc);
    static FrameGpu<Npp32f>* AddWeighted( FrameGpu<Npp32f>* imgBackground, const FrameGpu<Npp8u>* imgSrc, float alpha = 0.04);
    static FrameGpu<Npp32f> *AbsDiff(const FrameGpu<Npp32f>* imgBackground, const FrameGpu<Npp32f>* imageDiff);

};


#endif //NPPFUNCTION_H
