//#include <opencv2/core/cuda.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudawarping.hpp>
//#include "NvInfer.h"

#include "GstDecoder.h"
#include "MainLogger.hpp"
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include "BufferFrameGpu.h"
#include "cuda_runtime_api.h"
#include "FrameGpu.h"

using namespace std;
using namespace cv;

std::mutex _mtx;

void TestGstreamer()
{
    shared_ptr<logger> _logger = get("MainLogger");

    std::ostringstream ss2;
    // ss2 << "rtspsrc location=rtsp://login:password@192.168.1.15:554 latency=1000 "
    //     << "! rtph264depay "
    //     << "! nvv4l2decoder "
    //     << "! nvvideoconvert nvbuf-memory-type=3 "
    //     << "! video/x-raw(memory:NVMM) "
    //     << "! appsink name=mysink sync=true";

    ss2 << "filesrc location=/mnt/Disk_D/Document/Teplovisors/Dataset/010/11.09.2024_001.avi "
        << "! avidemux "
        << "! nvv4l2decoder "
        << "! nvvideoconvert nvbuf-memory-type=3 "
        << "! video/x-raw(memory:NVMM)"
        << "! appsink name=mysink sync=true";

    auto mLaunchStr = ss2.str();
    cudaStream_t stream = 0;

    auto buffer = new BufferFrameGpu(10);
    auto bufferManager = new GstBufferManager(buffer,stream);
    auto mPipeline = new GstDecoder(bufferManager);
    mPipeline->InitPipeline(mLaunchStr);
    mPipeline->Open();

    auto i = 200;
    while (i>0)
    {

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto start = chrono::system_clock::now();

        FrameGpu* frame;
        uint64_t timeStamp;
        auto result = buffer->Dequeue(&frame);
        // auto result = mPipeline->Capture(&image, &status);

        auto endCapture = chrono::system_clock::now();

        if (!result)
            continue;

        i = i-1;

        auto image = frame->GetImages();
        timeStamp = frame->GetTimestamp();
        // cuda::GpuMat imgResize;
        // auto imgGpu = cuda::GpuMat(512, 640, CV_16UC1, image, 1536);
        // cuda::resize(imgGpu, imgResize, Size(640, 640));

        Mat srcHost;
        image->download(srcHost);
        delete frame;

        auto endResize = chrono::system_clock::now();

        imshow("srcHost", srcHost);
        waitKey(1);


        std::ostringstream logs;
        logs << "timeStamp: " <<timeStamp
            << " All time: " << chrono::duration_cast<chrono::microseconds>(endResize - start).count()
            << " microseconds;"
            << " GPU MAT: " << chrono::duration_cast<chrono::microseconds>(endResize - endCapture).count()
            << " microseconds;"
            << " capture: " << chrono::duration_cast<chrono::microseconds>(endCapture - start).count()
            << " microseconds;" ;
        _logger->info(logs.str());
    }

    cv::destroyAllWindows();
    buffer->~BufferFrameGpu();
    bufferManager->~GstBufferManager();
    mPipeline->~GstDecoder();
    printf("tetst end \n");
}



void TestBufferFrameGpu(BufferFrameGpu* buffer)
{
    auto thread_id = std::this_thread::get_id();
    for (int i = 0; i < 1000; ++i)
    {
        FrameGpu* frame;

        _mtx.lock();
        cout << thread_id << " test 1 " << endl;
        auto imageHost = Mat(640, 640, CV_8UC1);
        randn(imageHost, 0, 255);
        cuda::GpuMat imgGpu;
        imgGpu.upload(imageHost);

        auto imgClone = new cuda::GpuMat(640, 640, CV_8UC1);
        imgGpu.copyTo(*imgClone);
        cudaFree(imgGpu.data);

        frame = new FrameGpu(imgClone, i);
        cout << thread_id << " test 2 " << endl;
        _mtx.unlock();

        cout << thread_id << " test 3 " << "img: " << i << endl;

        buffer->Enqueue(frame);

        cout << thread_id << " test 4 " << "img: " << i << endl;
        // imshow("image", image);
        // waitKey(0);
    }

    // for (int i = 0; i < 1000; ++i)
    // {
    //     FrameGpu *frame;
    //     buffer->Dequeue(&frame);
    //
    //     Mat srcHost;
    //     auto img = frame->GetImages();
    //     img->download(srcHost);
    //
    //     cout<<" "<< frame->GetTimestamp() <<endl;
    //
    //     imshow("image", srcHost);
    //     waitKey(0);
    //
    //     delete frame;
    //     frame = nullptr;
    // }
}


void TestBufferFrameGpuMultithreading()
{
    auto buffer = new BufferFrameGpu(10);

    std::thread t1(TestBufferFrameGpu, buffer);
    std::thread t2(TestBufferFrameGpu, buffer);

    t1.join();
    t2.join();
}

void TestGstreamerMainCreate()
{
    for (int i = 0; i < 100; ++i)
    {
        printf("process %i ---------------------- \n",i);
        std::this_thread::sleep_for(std::chrono::seconds(4));
        TestGstreamer();
    }
}

int main(int argc, char* argv[])
{
    string logPathFileString = "./Logs/Test_ConverterNetWeight.log";
    auto mainLogger = MainLogger(logPathFileString);

    // TestGstreamer();
    TestGstreamerMainCreate();
    // TestBufferFrameGpu();
    // TestBufferFrameGpuMultithreading();

    // shared_ptr<spdlog::logger> _loger = spdlog::get("MainLogger");
    // if (_loger != nullptr)
    //     _loger->flush();

    return 0;
}
