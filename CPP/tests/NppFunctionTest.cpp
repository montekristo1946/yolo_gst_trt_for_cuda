#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nppdefs.h>
#include <nppi.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <MainLogger.hpp>

#include <nppi_geometry_transforms.h>
#include <opencv2/highgui.hpp>

#include "CudaUtility.h"
#include "FrameGpu.h"
#include "NppFunction.h"

void TestResize( int iter = 100000)
{
    std::cout << " --- TestResize Run  ---" << std::endl;
    auto nppFunction = new NppFunction();

    //create input image
    Mat mat = cv::imread("../examples/img_001.jpg", cv::IMREAD_GRAYSCALE);
    auto channel = 1;
    auto allSizeSrc = mat.cols * mat.rows * channel;
    uint64_t timestamp = 999;
    unsigned char* imageSrcPtr = nullptr;

    for (int i = 0; i < iter; i++)
    {
        CUDA_FAILED(cudaMalloc(&imageSrcPtr, allSizeSrc));
        CUDA_FAILED(cudaMemcpy(imageSrcPtr, mat.data, allSizeSrc, cudaMemcpyHostToDevice));
        auto frameGpuSrc = new FrameGpu(imageSrcPtr, mat.cols, mat.rows, timestamp, channel);

        auto start = chrono::system_clock::now();

        auto frameGpuResize = nppFunction->ResizeGrayScale(frameGpuSrc, 1000, 900);

        // auto mat2 = cv::Mat(900, 1000, CV_8UC1);

        // CUDA_FAILED(  cudaMemcpy(mat2.data, frameGpuResize->ImagePtr(), frameGpuResize->GetFulSize(),cudaMemcpyDeviceToHost));
        // cv::imshow("mat", mat);
        // cv::imshow("mat2", mat2);
        // cv::waitKey(1);

        delete frameGpuResize;
        delete frameGpuSrc;

        auto endCapture = chrono::system_clock::now();
        info("elapsed time: " +
            to_string(chrono::duration_cast<chrono::microseconds>(endCapture - start).count()) +
            " iter: " + " " + to_string(i));

    }

    delete nppFunction;


    std::cout << " --- TestResize OK ---" << std::endl;
}


void TestConvertToGray(int iter = 100000)
{
    std::cout << " --- TestConvertToGray Run  ---" << std::endl;
    auto nppFunction = new NppFunction();

    //create input image
    Mat mat = cv::imread("../examples/img_001.jpg", cv::IMREAD_COLOR);
    auto channel = 3;
    auto allSizeSrc = mat.cols * mat.rows * channel;
    uint64_t timestamp = 999;
    unsigned char* imageSrcPtr = nullptr;

    for (int i = 0; i < iter; i++)
    {
        CUDA_FAILED(cudaMalloc(&imageSrcPtr, allSizeSrc));
        CUDA_FAILED(cudaMemcpy(imageSrcPtr, mat.data, allSizeSrc, cudaMemcpyHostToDevice));
        auto frameGpuSrc = new FrameGpu(imageSrcPtr, mat.cols, mat.rows, timestamp, channel);

        auto start = chrono::system_clock::now();

        auto frameGpuGray = nppFunction->RGBToGray(frameGpuSrc);

        // auto mat2 = cv::Mat(mat.rows, mat.cols, CV_8UC1);
        //
        // CUDA_FAILED(  cudaMemcpy(mat2.data, frameGpuGray->ImagePtr(), frameGpuGray->GetFulSize(),cudaMemcpyDeviceToHost));
        // cv::imshow("mat", mat);
        // cv::imshow("mat2", mat2);
        // cv::waitKey(1);

        delete frameGpuGray;
        delete frameGpuSrc;

        auto endCapture = chrono::system_clock::now();
        info("elapsed time: " +
            to_string(chrono::duration_cast<chrono::microseconds>(endCapture - start).count()) +
            " iter: " + " " + to_string(i));

    }

    delete nppFunction;

    std::cout << " --- TestConvertToGray OK ---" << std::endl;
}

int main(int argc, char* argv[])
{
    auto logPathFileString = "./Logs/NppFunctionTest.log";
    auto mainLogger = MainLogger(logPathFileString);

    // TestResize();
    TestConvertToGray();
    return 0;
}
