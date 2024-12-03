

#ifndef BufferFrameGpu_H
#define BufferFrameGpu_H
#include <IDispose.h>
#include "common/FrameGpu.h"
#include <nppdefs.h>
#include <queue>
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

    unsigned _sizeBuffer;
    queue<FrameGpu<Npp8u>*> _queueFrame;
    std::mutex _mtx;
};


#endif //FRAMEGPU_H

