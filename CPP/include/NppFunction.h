#ifndef NPPFUNCTION_H
#define NPPFUNCTION_H


#include "FrameGpu.h"
#include "IDispose.h"
#include "CudaUtility.h"
#include <nppi_geometry_transforms.h>

class NppFunction : public IDispose
{
public:
    NppFunction()
    {
        _logger->info("[NppFunction::Ctr]  Init  NppFunction ok");
    }

    FrameGpu<Npp8u>* ResizeGrayScale( const FrameGpu<Npp8u>* imsSrc, const int widthNew, const int heightNew);
    FrameGpu<Npp8u>* RGBToGray( const  FrameGpu<Npp8u>* imsSrc);
    FrameGpu<float>* AddWeighted( FrameGpu<float>* imgBackground, const FrameGpu<Npp8u>* imgSrc,const float alpha = 0.04);

private:
    shared_ptr<logger> _logger = get("MainLogger");



};


#endif //NPPFUNCTION_H
