

#ifndef BufferFrameGpu_H
#define BufferFrameGpu_H
#include <IDispose.h>
#include "common/FrameGpu.h"
#include <nppdefs.h>
#include <spdlog/spdlog.h>
#include "Concurrentqueue.h"

using namespace std;
using namespace moodycamel;

class BufferFrameGpu: public IDispose
{
public:
    BufferFrameGpu(unsigned sizeBuffer);

    ~BufferFrameGpu();

    bool Enqueue(FrameGpu<Npp8u>* frame);
    bool Dequeue(FrameGpu<Npp8u>** frame);

private:

    unsigned _sizeBuffer;
    ConcurrentQueue<FrameGpu<Npp8u> *> _queue ;
};


#endif //FRAMEGPU_H

