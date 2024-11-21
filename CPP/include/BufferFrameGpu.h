

#ifndef BufferFrameGpu_H
#define BufferFrameGpu_H
#include <IDispose.h>
#include <queue>
#include <opencv2/core.hpp>
#include "cuda_runtime_api.h"
#include "common/FrameGpu.h"

using namespace std;
using namespace cv;

class BufferFrameGpu: public IDispose
{
public:
    BufferFrameGpu(unsigned sizeBuffer);

    ~BufferFrameGpu();

    bool Enqueue(FrameGpu* frame);
    bool Dequeue(FrameGpu** frame);

private:
    std::queue<FrameGpu*> _queueFrame;
    std::mutex _mtx;
    unsigned _sizeBuffer;
};


#endif //FRAMEGPU_H

