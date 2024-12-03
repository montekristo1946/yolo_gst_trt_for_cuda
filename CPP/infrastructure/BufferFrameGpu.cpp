#include "BufferFrameGpu.h"


// Constructor for FrameGpu class
// Initializes the FrameGpu object with a buffer size
BufferFrameGpu::BufferFrameGpu(unsigned sizeBuffer)
{
    if (sizeBuffer <= 0)
        throw runtime_error("[BufferFrameGpu::BufferFrameGpu] sizeBuffer <= 0");

    printf("test F BufferFrameGpu ctor %d \n", sizeBuffer);

    _sizeBuffer = sizeBuffer;
    _queue = ConcurrentQueue<FrameGpu<Npp8u>*>(_sizeBuffer);
}

BufferFrameGpu::~BufferFrameGpu()
{
    FrameGpu<Npp8u>* curentIntqueue;
    while (_queue.try_dequeue(curentIntqueue))
    {
        info("[BufferFrameGpu::~BufferFrameGpu] Delete Images: {timestamp} ", curentIntqueue->Timestamp());
        delete curentIntqueue;
    }
    info("[~BufferFrameGpu] Call");
}


bool BufferFrameGpu::Enqueue(FrameGpu<Npp8u>* frame)
{
    try
    {
        if (_queue.size_approx() >= _sizeBuffer)
        {
            FrameGpu<Npp8u>* frameTmp = nullptr;
            while (_queue.try_dequeue(frameTmp))
            {
                warn("[BufferFrameGpu::Enqueue] Delete Images: {timestamp} ", frameTmp->Timestamp());
                delete frameTmp;
            };
        }

        if (!_queue.try_enqueue(frame))
        {
            error("[BufferFrameGpu::Enqueue] Fail Add enqueue ");
            if(frame)
            {
                delete frame;
                frame = nullptr;
            }
            return false;
        }

        return true;
    }
    catch (...)
    {
        error("[BufferFrameGpu::Enqueue] fail _queueFrame.push(frame);");
    }


    return false;
}


bool BufferFrameGpu::Dequeue(FrameGpu<Npp8u>** frame)
{
    try
    {
        if(!_queue.try_dequeue(*frame))
        {
            return false;
        }

        return true;
    }
    catch (...)
    {
        error("[BufferFrameGpu::Dequeue] fail _queueFrame.front();");
    }

    return false;
}
