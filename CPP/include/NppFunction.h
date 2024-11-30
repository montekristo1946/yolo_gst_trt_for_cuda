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

    FrameGpu* ResizeGrayScale( const FrameGpu* imsSrc, const int widthNew, const int heightNew);
    FrameGpu* RGBToGray( const  FrameGpu* imsSrc);

private:
    shared_ptr<logger> _logger = get("MainLogger");



};


#endif //NPPFUNCTION_H
