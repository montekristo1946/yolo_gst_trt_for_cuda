
#include "BufferFrameGpu.h"




// Constructor for FrameGpu class
// Initializes the FrameGpu object with a buffer size
BufferFrameGpu::BufferFrameGpu(unsigned sizeBuffer)
{
    _sizeBuffer = sizeBuffer;
}

BufferFrameGpu::~BufferFrameGpu()
{
    while (!_queueFrame.empty())
    {
        // Remove and delete the oldest frame to make space
        auto *frameTmp = _queueFrame.front();
        _queueFrame.pop();
        delete frameTmp;
    }
    info("[~BufferFrameGpu] Call");
}



/**
 * @brief Enqueues a FrameGpu object into the buffer queue.
 *
 * This function adds a FrameGpu object to the end of the queue. If the queue
 * size exceeds the buffer size, it will remove and delete the oldest frame
 * to make space. Uses a mutex for thread safety.
 *
 * @param frame Pointer to the FrameGpu object to be enqueued.
 * @return true If the frame was successfully enqueued.
 * @return false If an exception occurred during enqueuing.
 */
bool BufferFrameGpu::Enqueue(FrameGpu<Npp8u>* frame)
{
   unique_lock lock(_mtx); // Acquire lock for thread safety
    try
    {
        while (_queueFrame.size() >= _sizeBuffer)
        {
            // warn("[BufferFrameGpu::Enqueue] skip Frame");
            auto *frameTmp = _queueFrame.front();
            _queueFrame.pop();
            std::unique_ptr<FrameGpu<Npp8u>> frameUn(frameTmp);
            // delete frameTmp;
        }

        _queueFrame.push(frame);

        return true;
    }
    catch (...)
    {
        // Log an error if enqueuing fails
        spdlog::error("[BufferFrameGpu::Enqueue] fail _queueFrame.push(frame);");
        lock.unlock();
    }

    return false;
}

/**
 * @brief Retrieves the oldest FrameGpu object from the buffer queue.
 *
 * This function retrieves the oldest FrameGpu object from the front of the
 * queue. If the queue is empty, it returns false and does not modify the
 * referenced frame object. Uses a mutex for thread safety.
 *
 * @param frame Reference to a FrameGpu pointer to store the dequeued frame.
 * @return true If a frame was successfully dequeued.
 * @return false If an exception occurred during dequeuing or the queue is empty.
 */
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

        // Retrieve the oldest frame from the front of the queue
        *frame = _queueFrame.front();
        _queueFrame.pop();

        return true;
    }
    catch (...)
    {
        // Log an error if dequeuing fails
        spdlog::error("[BufferFrameGpu::Dequeue] fail _queueFrame.front();");
        lock.unlock();
    }

    return false;
}
