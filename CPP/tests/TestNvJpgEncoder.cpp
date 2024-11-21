#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nvjpeg.h>
#include <ostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>

#include "CudaUtility.h"
#include "NvJpgEncoder.h"
// #include <cuda_runtime.h>

using namespace cv;
using namespace std;

void FullPass()
{
     /*  nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    unsigned char * pBufferChanel1 = NULL;
    unsigned char * pBufferChanel2 = NULL;
    unsigned char * pBufferChanel3 = NULL;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int resize_quality = 90;
    // initialize nvjpeg structures
    CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nv_enc_params, resize_quality, NULL));

    nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_410;
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, subsampling, NULL));

    // Fill nv_image with image data, let's say 640x480 image in RGB format
    // auto image = cv::imread("/mnt/Disk_D/TMP/12.11.2024/bmp/frame000000.bmp", IMREAD_GRAYSCALE);

    cv::Mat image = cv::Mat::zeros(640, 640, CV_8U);
    cv::randu(image, cv::Scalar(0), cv::Scalar(250));

    // imshow("image",image);
    // waitKey();

    printf("   Target isContinuous: %ld, \n",image.isContinuous());
    // Mat bgr[3];
    // split(image, bgr); // split RGB three channels from whole_slide_rgb[idx]



    auto target_width = image.cols;
    auto target_height = image.rows;

    for (int i = 0; i < 100000; i++)
    {
        cv::randu(image, cv::Scalar(0), cv::Scalar(250));
        nvjpegImage_t nv_image;
        auto start = chrono::system_clock::now();
        // for (int i = 0; i < 3; i++)
        // {
        //     CUDA_FAILED(cudaMalloc((void**)&(nv_image.channel[i]), target_width * target_height));
        //     CUDA_FAILED(cudaMemcpy(nv_image.channel[i], image.data, target_width * target_height,cudaMemcpyHostToDevice));
        //     nv_image.pitch[i] = (size_t)target_width;
        // }
        CUDA_FAILED(cudaMallocAsync((void**)&pBufferChanel1, target_width * target_height, stream));
        CUDA_FAILED(cudaMemcpyAsync(pBufferChanel1, image.data, target_width * target_height,cudaMemcpyHostToDevice,stream));
        nv_image.channel[0] = pBufferChanel1;
        nv_image.pitch[0] = (size_t)target_width;

        CUDA_FAILED(cudaMallocAsync((void**)&pBufferChanel2, target_width * target_height,stream));
        CUDA_FAILED(cudaMemcpyAsync(pBufferChanel2, image.data, target_width * target_height,cudaMemcpyHostToDevice,stream));
        nv_image.channel[1] = pBufferChanel2;
        nv_image.pitch[1] = (size_t)target_width;

        CUDA_FAILED(cudaMallocAsync((void**)&pBufferChanel3, target_width * target_height,stream));
        CUDA_FAILED(cudaMemcpyAsync(pBufferChanel3, image.data, target_width * target_height,cudaMemcpyHostToDevice,stream));
        nv_image.channel[2] = pBufferChanel3;
        nv_image.pitch[2] = (size_t)target_width;


        // Compress image
        CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
            &nv_image, NVJPEG_INPUT_RGB, target_width, target_height, stream));

        // get compressed stream size
        size_t length;
        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
        // get stream itself
        CUDA_FAILED(cudaStreamSynchronize(stream));
        std::vector<unsigned char> jpeg(length);
        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg.data(), &length, 0));

        // write stream to file
        CUDA_FAILED(cudaStreamSynchronize(stream));


        CUDA_FAILED(cudaFree(pBufferChanel1));
        CUDA_FAILED(cudaFree(pBufferChanel2));
        CUDA_FAILED(cudaFree(pBufferChanel3));

        // string pathcFile = "/mnt/Disk_D/TMP/12.11.2024/export/frame" + to_string(i) + ".jpg";
        // std::ofstream output_file(pathcFile, std::ios::out | std::ios::binary);
        // output_file.write(reinterpret_cast<const char *>(jpeg.data()), static_cast<int>(length));
        // output_file.close();

        auto endAllProcess = chrono::system_clock::now();

        cout << "iter: " << i
             << " All time: " << chrono::duration_cast<chrono::microseconds>(endAllProcess - start).count()
             << " microseconds"
             << endl;
    }

    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(nv_enc_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(nv_enc_state));
    CHECK_NVJPEG(nvjpegDestroy(nv_handle));*/
}
void TestRun()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto encoder = new NvJpgEncoder(&stream);
    // auto image = imread("../examples/img_001.jpg", IMREAD_GRAYSCALE);

    for (int i = 0; i < 1000000; i++)
    {
        cv::Mat image = cv::Mat::zeros(640, 640, CV_8U);
        cv::randu(image, cv::Scalar(0), cv::Scalar(250));

        cuda::GpuMat imgGpu;
        imgGpu.upload(image);

        std::vector<unsigned char>* encodedImage =  encoder->Encode(imgGpu);

        printf("   Target A size: %ld, \n",encodedImage->size());
        cv::Mat imageConvert = cv::imdecode(cv::_InputArray(encodedImage->data(), encodedImage->size()), cv::IMREAD_COLOR);

        // imshow("original", image);
        // imshow("imageConvert", imageConvert);
        // waitKey(1);
        imgGpu.release();
    }
}

int main()
{

    TestRun();


}
