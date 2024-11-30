

#include "nppi.h"
#include "nppi_geometry_transforms.h"

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <vector>

void write(const cv::Mat& mat1, const std::string& path) {
    auto mat2 = cv::Mat(mat1.rows, mat1.cols, CV_8UC4);
    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat1.cols; j++) {
            auto& bgra = mat2.at<cv::Vec4b>(i, j);
            auto& rgb = mat1.at<cv::Vec3b>(i, j);
            bgra[0] = rgb[2];
            bgra[1] = rgb[1];
            bgra[2] = rgb[0];
            bgra[3] = UCHAR_MAX;
        }
    }
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    cv::imwrite(path, mat2, compression_params);
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    auto mat = cv::Mat(256, 256, CV_8UC3);
    auto mat2 = cv::Mat(mat.rows, mat.cols, CV_8UC3);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            auto& rgb = mat.at<cv::Vec3b>(i, j);
            rgb[0] = (uint8_t)j;
            rgb[1] = (uint8_t)i;
            rgb[2] = (uint8_t)(UCHAR_MAX - j);
        }
    }
    write(mat, "./test.png");
    uint8_t* gpuBuffer1;
    uint8_t* gpuBuffer2;
    size_t mat_size_in_bytes = mat.step[0] * mat.rows;  // https://stackoverflow.com/questions/26441072/finding-the-size-in-bytes-of-cvmat
    size_t mat2_size_in_bytes = mat2.step[0] * mat2.rows;
    cudaMalloc(&gpuBuffer1, mat_size_in_bytes);
    cudaMalloc(&gpuBuffer2, mat2_size_in_bytes);
    cudaMemcpy(gpuBuffer1, mat.data, mat_size_in_bytes, cudaMemcpyHostToDevice);

    NppiSize oSrcSize = { mat.cols, mat.rows };
    NppiRect oSrcRectROI = { 0, 0, mat.cols, mat.rows };
    NppiSize oDstSize = { mat2.cols, mat2.rows };
    NppiRect oDstRectROI = { 0, 0, mat2.cols, mat2.rows };

    auto status = nppiResize_8u_C3R(
        gpuBuffer1, mat.step[0], oSrcSize,
        oSrcRectROI, gpuBuffer2,
        mat2.step[0], oDstSize,
        oDstRectROI,
        NPPI_INTER_NN);

    if (status != NPP_SUCCESS) {
        std::cerr << "Error executing Resize -- code: " << status << std::endl;
    }

    cudaMemcpy(mat2.data, gpuBuffer2, mat2_size_in_bytes, cudaMemcpyDeviceToHost);
    write(mat2, "./test1.png");
}