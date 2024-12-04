#include "BufferFrameGpu.h"



BufferFrameGpu::BufferFrameGpu(unsigned sizeBuffer)
{
    if (sizeBuffer <= 0)
        throw runtime_error("[BufferFrameGpu::BufferFrameGpu] sizeBuffer <= 0");

    info("[BufferFrameGpu::BufferFrameGpu] Ctor sizeBuffer:{}", sizeBuffer);
    _sizeBuffer = sizeBuffer;

}

BufferFrameGpu::~BufferFrameGpu()
{
    unique_lock lock(_mtx);
    while (!_queueFrame.empty())
    {
        auto *frameTmp = _queueFrame.front();
        _queueFrame.pop();
        warn("[BufferFrameGpu::~BufferFrameGpu] Delete Frame: {}", frameTmp->Timestamp());
        delete frameTmp;
    }
    info("[~BufferFrameGpu] Call");
}


bool BufferFrameGpu::Enqueue(FrameGpu<Npp8u>* frame)
{
    unique_lock lock(_mtx);
    try
    {
        const int countMilisec = 1000000;
        while (_queueFrame.size() >= _sizeBuffer)
        {
            auto *frameTmp = _queueFrame.front();
            _queueFrame.pop();
            warn("[BufferFrameGpu::Enqueue] skip FrameTime: {} ms, sizeBuffer:{}", frameTmp->Timestamp()/countMilisec,_queueFrame.size() );
            delete frameTmp;

        }

        _queueFrame.push(frame);

        return true;
    }
    catch (...)
    {
        error("[BufferFrameGpu::Enqueue] fail _queueFrame.push(frame);");
        lock.unlock();
    }

    return false;

}


bool BufferFrameGpu::Dequeue(FrameGpu<Npp8u>** frame)
{
    unique_lock lock(_mtx);
    try
    {
        // Check if the queue is empty
        if (_queueFrame.empty())
        {
            return false; // Queue is empty
        }

        *frame = _queueFrame.front();
        _queueFrame.pop();

        return true;
    }
    catch (...)
    {
        error("[BufferFrameGpu::Dequeue] fail _queueFrame.front();");
        lock.unlock();
    }

    return false;

}
