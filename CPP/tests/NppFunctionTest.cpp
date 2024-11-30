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

using namespace cv;

void TestResize(int iter = 100000, bool isShow = false)
{
    std::cout << " --- TestResize Run  ---" << std::endl;
    auto nppFunction = new NppFunction();

    //create input image
    Mat mat = imread("../examples/img_001.jpg", cv::IMREAD_GRAYSCALE);
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

        if (isShow)
        {
            auto mat2 = cv::Mat(900, 1000, CV_8UC1);
            CUDA_FAILED(
                cudaMemcpy(mat2.data, frameGpuResize->ImagePtr(), frameGpuResize->GetFulSize(),cudaMemcpyDeviceToHost));
            cv::imshow("mat", mat);
            cv::imshow("mat2", mat2);
            cv::waitKey(1);
        }
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

void TestConvertToGray(int iter = 100000, bool isShow = false)
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

        if (isShow)
        {
            auto mat2 = cv::Mat(mat.rows, mat.cols, CV_8UC1);
            CUDA_FAILED(
                cudaMemcpy(mat2.data, frameGpuGray->ImagePtr(), frameGpuGray->GetFulSize(),cudaMemcpyDeviceToHost));
            cv::imshow("mat", mat);
            cv::imshow("mat2", mat2);
            cv::waitKey(1);
        }

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

void TestAddWeighted(const Mat& matInput, FrameGpu<Npp32f>* frameBackground, NppFunction *nppFunction)
{
    Mat imgSrcGray;
    cvtColor(matInput, imgSrcGray, COLOR_BGR2GRAY);
    auto channel = 1;

    Npp8u* imageSrcPtr = nullptr;
    auto allSizeSrc = imgSrcGray.cols * imgSrcGray.rows;
    CUDA_FAILED(cudaMalloc((void **)(&imageSrcPtr), allSizeSrc*sizeof(Npp8u) ));
    CUDA_FAILED(cudaMemcpy(imageSrcPtr, imgSrcGray.data, allSizeSrc*sizeof(Npp8u), cudaMemcpyHostToDevice));
    FrameGpu<Npp8u>* imgSrc = new FrameGpu(imageSrcPtr, imgSrcGray.cols, imgSrcGray.rows, 888, channel);

    auto start = chrono::system_clock::now();
    frameBackground = nppFunction->AddWeighted(frameBackground, imgSrc);
    delete imgSrc;
    auto endCapture = chrono::system_clock::now();

    info("elapsed time: " +
        to_string(chrono::duration_cast<chrono::microseconds>(endCapture - start).count()));

    auto allSizeDst = imgSrcGray.cols * imgSrcGray.rows * channel;
    auto mat2 = Mat(imgSrcGray.rows, imgSrcGray.cols, CV_32FC1, imgSrcGray.cols* sizeof(float));
    CUDA_FAILED(cudaMemcpy(mat2.data, frameBackground->ImagePtr(), allSizeDst* sizeof(float), cudaMemcpyDeviceToHost));

    Mat dst;
    mat2.convertTo(dst, CV_8UC1);
    imshow("mat", imgSrcGray);
    imshow("dst", dst);


}

void TestAddWeightedOnVideo()
{
    string filename = "/mnt/Disk_D/Document/Teplovisors/Dataset/010/11.09.2024_001.avi";
    VideoCapture capture(filename, cv::CAP_FFMPEG);
    Mat frame;

    if (!capture.isOpened())
        throw "Error when reading steam_avi";

    auto nppFunction = new NppFunction();

    Npp32f* retImage = nullptr;

    auto width = 640;
    auto height = 512;
    auto channel = 1;
    auto allSizeDst = width * height * channel;
    uint64_t timestamp = 999;
    CUDA_FAILED(cudaMalloc((void **)(&retImage), allSizeDst * sizeof(float)));
    auto frameBackground = new FrameGpu(retImage, width, height, timestamp, channel);

    namedWindow("w", 1);
    for (; ;)
    {
        capture >> frame;
        if (frame.empty())
            break;
        TestAddWeighted(frame, frameBackground,nppFunction);
        imshow("w", frame);
        waitKey(1); // waits to display frame
    }
    waitKey(0);
}

FrameGpu<Npp32f>* IntiImgFloat(const Mat& mat)
{
    Npp32f* imagePtr = nullptr;
    auto allSize = mat.cols * mat.rows;
    CUDA_FAILED(cudaMalloc((void **)(&imagePtr), allSize*sizeof(Npp32f) ));
    CUDA_FAILED(cudaMemcpy(imagePtr, mat.data, allSize*sizeof(Npp32f), cudaMemcpyHostToDevice));

    FrameGpu<Npp32f>* imgSrc = new FrameGpu(imagePtr, mat.cols, mat.rows, 888, mat.channels());
    return imgSrc;
}

void TestAbsDiff(int iter= 100000, bool isShow = false)
{
    std::cout << " --- TestAbsDiff Run  ---" << std::endl;
    auto nppFunction = new NppFunction();

    Mat matFonSrc = cv::imread("../examples/img_002.jpg", cv::IMREAD_GRAYSCALE);
    Mat matStc = cv::imread("../examples/img_003.jpg", cv::IMREAD_GRAYSCALE);

    auto width = matFonSrc.cols;
    auto height = matFonSrc.rows;

    Mat matFonFloat;
    matFonSrc.convertTo(matFonFloat, CV_32FC1);
    Mat matStcFloat;
    matStc.convertTo(matStcFloat, CV_32FC1);

    for (int i = 0; i < iter; i++)
    {
        auto* imageFonPtr = IntiImgFloat(matFonFloat);
        auto* imageStcPtr = IntiImgFloat(matStcFloat);

        auto start = chrono::system_clock::now();
        auto resDiff = nppFunction->AbsDiff(imageFonPtr, imageStcPtr);

        if(isShow)
        {
            auto allSizeDst = resDiff->Width() * resDiff->Height();
            auto mat2 = Mat(resDiff->Width(), resDiff->Height(), CV_32FC1, width* sizeof(float));
            CUDA_FAILED(cudaMemcpy(mat2.data, resDiff->ImagePtr(), allSizeDst* sizeof(float), cudaMemcpyDeviceToHost));
            Mat dst;
            mat2.convertTo(dst, CV_8UC1);
            imshow("dst", dst);
            imshow("matFonSrc", matFonSrc);
            imshow("matStc", matStc);
            waitKey(1);
        }

        delete imageFonPtr;
        delete imageStcPtr;
        delete resDiff;

        auto endCapture = chrono::system_clock::now();
        info("elapsed time: " +
            to_string(chrono::duration_cast<chrono::microseconds>(endCapture - start).count()) +
            " iter: " + " " + to_string(i));

    }

    std::cout << " --- TestAbsDiff End ok  ---" << std::endl;
}

int main(int argc, char* argv[])
{
    auto logPathFileString = "./Logs/NppFunctionTest.log";
    auto mainLogger = MainLogger(logPathFileString);

    TestResize(100000, false);
    // TestConvertToGray(100000, true);
    // TestAddWeightedOnVideo();
    // TestAbsDiff(100000, true);
    return 0;
}
