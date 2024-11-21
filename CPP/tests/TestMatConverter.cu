#include "MatConverter.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;


void TestRun()
{
    auto matConverter = new MatConverter();
    auto image = cv::imread("../examples/img_001.jpg", IMREAD_GRAYSCALE);
    cuda::GpuMat imgGpu;
    imgGpu.upload(image);
    cuda::resize(imgGpu, imgGpu, Size(800, 700));

    // printf("image %d \n", image.isContinuous());
    // printf("imgGpu %d \n", imgGpu.isContinuous());

    for (int i = 0; i < 100000; ++i)
    {
        auto start = std::chrono::system_clock::now();
        float* output;
        auto res = cudaMalloc((void**)&output, imgGpu.cols * imgGpu.rows * sizeof(float));
        printf("cudaMalloc %d  ", res);

        cudaStream_t stream = 0;
        res = matConverter->GrayToFloat32ContinueArr(&imgGpu, output, &stream);
        printf("GrayToFloat32ContinueArr %d  ", res);

        auto end = std::chrono::system_clock::now();
        auto timeElapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        printf("elapsed: %ld \n", timeElapsed);

        auto matToconvert = cuda::GpuMat(imgGpu.rows, imgGpu.cols, CV_32FC1, output);

        // Mat srcHost;
        // matToconvert.download(srcHost);
        // srcHost.convertTo(srcHost, CV_8UC1);
        cudaFree(matToconvert.data);

        // imshow("srcHost",srcHost);
        // imshow("image",image);
        // waitKey(1);
    }

}

int main(int argc, char* argv[])
{
    TestRun();
    return 0;
}
