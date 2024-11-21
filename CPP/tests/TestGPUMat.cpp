
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "cuda_runtime_api.h"

using namespace std;
using namespace cv;

void TestLoadToGpuMat()
{
    auto image = imread("../examples/img_001.jpg", IMREAD_ANYCOLOR);
    cuda::GpuMat imgGpu;
    cuda::GpuMat imgResize;
    imgGpu.upload(image);
    cuda::resize(imgGpu,imgResize,Size(640,640));
    cuda:: cvtColor(imgResize, imgResize, COLOR_BGR2RGB);
    Mat srcHost;
    imgResize.download(srcHost);

    cudaFree(imgResize.data);
    cudaFree(imgGpu.data);

    imshow("srcHost",srcHost);
    waitKey();
}

int main(int argc, char *argv[]) {

    TestLoadToGpuMat();

    return 0;
}
