

#ifndef BufferFrameGpu_H
#define BufferFrameGpu_H
#include <IDispose.h>
#include <queue>
#include "cuda_runtime_api.h"
#include "common/FrameGpu.h"
#include <nppdefs.h>
#include <spdlog/spdlog.h>
using namespace std;


class BufferFrameGpu: public IDispose
{
public:
    BufferFrameGpu(unsigned sizeBuffer);

    ~BufferFrameGpu();

    bool Enqueue(FrameGpu<Npp8u>* frame);
    bool Dequeue(FrameGpu<Npp8u>** frame);

private:
    std::queue<FrameGpu<Npp8u>*> _queueFrame;
    std::mutex _mtx;
    unsigned _sizeBuffer;
};


#endif //FRAMEGPU_H

