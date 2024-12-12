
//#include <iostream>
#include <BufferFrameGpu.h>
#include <GstBufferManager.h>
#include <GstDecoder.h>

#include "TRTEngineConfig.hpp"
#include "Helper.hpp"
#include "MainLogger.hpp"
#include "TRTEngine.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "CudaStream.h"
#include "NvInfer.h"
#include "NvJpgEncoder.h"
#include "SettingPipeline.h"
#include "EnginePipeline.h"


// void Test_CreateTRTEngine(string pathWeight)
// {
//     cout << "______ Start Test_CreateTRTEngine OK ______" << endl;
//     auto configTrt = CreateTRTEngineConfig(pathWeight);
//
//     auto trtEngine = CreateTRTEngine(configTrt);
//
//     delete trtEngine;
//
//     cout << "______ End Test_CreateTRTEngine OK ______" << endl;
// }






void LoadTest(string pathWeight)
{
    // cout << "______ Start LoadTest OK ______" << endl;
    //
    //
    // auto openCvHelper = new OpenCvHelper();
    // auto configTrt = CreateTRTEngineConfig(pathWeight);
    // auto trtEngine = CreateTRTEngine(configTrt);
    //
    // auto image = imread("../examples/img_001.jpg", IMREAD_COLOR);
    // cvtColor(image, image, COLOR_BGR2RGB);
    // auto imageArr = vector<Mat>{image};
    //
    // for (int i = 0; i < 10000000; ++i) {
    //     auto start = chrono::system_clock::now();
    //
    //     vector<float> dataToDl(trtEngine->GetFullSizeInputLayer());
    //     auto inputSize = trtEngine->InputShape();
    //     auto resultConvert = openCvHelper->ConvertMatToTRTArray(imageArr, inputSize, dataToDl);
    //     if (resultConvert == false)
    //         throw runtime_error("[Test_CreateTRTEngine] openCvHelper->ConvertMatToTRTArray false");
    //
    //     vector<Detection> resultNms;
    //
    //     auto resulDoInferenceAsync = trtEngine->DoInferenceNMSAsync(dataToDl.data(), resultNms);
    //     if (resulDoInferenceAsync == false)
    //         throw runtime_error("[Test_CreateTRTEngine] trtEngine->DoInferenceNMSAsync false");
    //
    //     if (resultNms.size() != 17)
    //         throw runtime_error("[Test_FullPass] oloTools->NMSOpencv empty");
    //
    //
    //     auto endAllProcess = chrono::system_clock::now();
    //
    //     cout << "iter: " << i
    //          << " All time: " << chrono::duration_cast<chrono::microseconds>(endAllProcess - start).count()
    //          << " microseconds"
    //          << endl;
    // }
    // delete trtEngine;
    // delete openCvHelper;
    // delete configTrt;
    //
    //
    // cout << "______ End LoadTest OK ______" << endl;
}


void Test_FullPassYoloGPU(string pathWeight, bool isDrawingResults)
{
    // cout << "______ Start Test_FullPass OK ______" << endl;
    // auto configTrt = CreateTRTEngineConfig(pathWeight);
    // auto trtEngine = CreateTRTEngine(configTrt);
    //
    // auto openCvHelper = new OpenCvHelper();
    // auto image = imread("../examples/img_001.jpg", IMREAD_COLOR);
    // cvtColor(image, image, COLOR_BGR2RGB);
    // auto imageArr = vector<Mat>{image};
    // vector<float> dataToDl(trtEngine->GetFullSizeInputLayer());
    // auto inputSize = trtEngine->InputShape();
    // auto resultConvert = openCvHelper->ConvertMatToTRTArray(imageArr, inputSize, dataToDl);
    // if (resultConvert == false)
    //     throw runtime_error("[Test_CreateTRTEngine] openCvHelper->ConvertMatToTRTArray false");
    //
    // vector<Detection> resultNms;
    // auto resulDoInferenceAsync = trtEngine->DoInferenceNMSAsync(dataToDl.data(), resultNms);
    // if (resulDoInferenceAsync == false)
    //     throw runtime_error("[Test_CreateTRTEngine] trtEngine->DoInferenceAsync false");
    //
    // if (resultNms.size() != 17)
    //     throw runtime_error("[Test_FullPass] oloTools->NMSOpencv empty");
    //
    // if (isDrawingResults)
    //     DrawingResults(image, resultNms);
    //
    // delete trtEngine;
    // delete openCvHelper;
    // delete configTrt;

    //cout << "______ End Test_FullPass OK ______" << endl;
}
/*
void Test_matGpu_in_trt(string pathWeight)
{
    // cuda::GpuMat test =  cuda::GpuMat(640, 640,CV_8UC1);
    // cuda::GpuMat difImgF;
    // test.convertTo(difImgF,CV_32FC1);

    cout << "______ Start Test_FullPass OK ______" << endl;
    auto configTrt = CreateTRTEngineConfig(pathWeight);
    auto trtEngine = CreateTRTEngine(configTrt);
    auto streem = &trtEngine->_stream;

    auto image = imread("../examples/img_001.jpg", IMREAD_COLOR);
    vector<Mat> channels;
    cvtColor(image, image, COLOR_BGR2RGB);


    split(image, channels);

    Mat blueChannel = channels[0];
    blueChannel.convertTo(blueChannel, CV_32FC1);

    Mat greenChannel = channels[1];
    greenChannel.convertTo(greenChannel, CV_32FC1);

    Mat redChannel = channels[2];
    redChannel.convertTo(redChannel, CV_32FC1);

    cuda::GpuMat imgB, imgG, imgR;
    imgB.upload(blueChannel, (cuda::Stream&)streem);
    imgG.upload(greenChannel, (cuda::Stream&)streem);
    imgR.upload(redChannel, (cuda::Stream&)streem);

    //    auto sizeAllImg = 640 * 640 * 3;
    auto oneLayer = 640 * 640;
    float* ptrImgGPU = static_cast<float*>(trtEngine->_buffers[0]);


    //    auto res = cudaMalloc(reinterpret_cast<void **>(&ptrImgGPU), sizeAllImg * sizeof(float));
    //    if (res != 0) {
    //        throw runtime_error("cudaMalloc false");
    //    }

    if (!imgB.isContinuous())
        throw runtime_error("!imgB.isContinuous()");

    auto res = cudaMemcpy(ptrImgGPU, &imgB.data[0], oneLayer * sizeof(float), cudaMemcpyDeviceToDevice);

    if (res != 0)
    {
        throw runtime_error("cudaMemcpy false");
    }

    res = cudaMemcpy(ptrImgGPU + oneLayer, &imgG.data[0], oneLayer * sizeof(float), cudaMemcpyDeviceToDevice);

    if (res != 0)
    {
        throw runtime_error("cudaMemcpy false");
    }

    res = cudaMemcpy(ptrImgGPU + oneLayer + oneLayer, &imgR.data[0], oneLayer * sizeof(float),
                     cudaMemcpyDeviceToDevice);

    if (res != 0)
    {
        throw runtime_error("cudaMemcpy false");
    }

    vector<Detection> resultNms;
    auto resulDoInferenceAsync = trtEngine->DoInferenceNMSAsync( resultNms);
    if (resulDoInferenceAsync == false)
        throw runtime_error("[Test_CreateTRTEngine] trtEngine->DoInferenceAsync false");

    if (resultNms.size() != 17)
        throw runtime_error("[Test_FullPass] oloTools->NMSOpencv empty");


    // auto imgGpu = cuda::GpuMat(640, 640, CV_32FC1, ptrImgGPU+oneLayer+oneLayer);
    // Mat srcHost;
    // imgGpu.download(srcHost);
    // srcHost.convertTo(srcHost, CV_8UC1);
    //
    // equalizeHist(srcHost,srcHost);
    // imshow("srcHost",srcHost);
    // waitKey();

    delete trtEngine;


    cout << "______ End Test_FullPass OK ______" << endl;
}
*/

string CreateConnectString()
{
    std::ostringstream ss2;
    // ss2 << "filesrc location=/mnt/Disk_D/Document/Teplovisors/Dataset/010/11.09.2024_001.avi "
    //     << "! avidemux "
    //     << "! nvv4l2decoder "
    //     << "! nvvideoconvert nvbuf-memory-type=3 "
    //     << "! video/x-raw(memory:NVMM)"
    //     << "! appsink name=mysink sync=true";

    ss2  << "rtspsrc location=rtsp://admin:1234567Qw@10.225.1.66:554 latency=1000 "
                   <<  "! rtph264depay "
                 <<    "! nvv4l2decoder "
                  <<   "! nvvideoconvert nvbuf-memory-type=3 "
                  <<   "! video/x-raw(memory:NVMM) "
                   <<  "! appsink name=mysink sync=true";

    auto mLaunchStr = ss2.str();
    return mLaunchStr;
}

TRTEngine* CreateTRTEngineLocal(TRTEngineConfig* config, CudaStream* cudaStream)
{
    auto trtEngine = new TRTEngine(cudaStream->GetStream());
    bool resultInitTRT = trtEngine->InitTRTEngine(config->EngineName,
                                                  config->DeviseId,
                                                  config->ConfThresh,
                                                  config->NmsThresh,
                                                  config->MaxNumOutputBbox);
    if (!resultInitTRT)
        throw std::runtime_error("[CreateTRTEngine] fail InitTRTEngine");

    return trtEngine;
}

void TestGstreamer()
{
    throw std::runtime_error("[CreateTRTEngine] not implement TestGstreamer");
    /*
    cout << "______ Start Test_FullPass OK ______" << endl;
    auto streem = new CudaStream();
    auto pathWeight = "../weight/model_001.engine";
    auto configTrt = CreateTRTEngineConfig(pathWeight);
    auto trtEngine = CreateTRTEngineLocal(configTrt,streem);

    auto connectString = CreateConnectString();

    auto settingPipeline = new SettingPipeline();
    auto bufferFrameGpu = new BufferFrameGpu(5);
    auto bufferManager = new GstBufferManager(bufferFrameGpu, streem->GetStream());

    auto gstDecoder = new GstDecoder(bufferManager);
    auto encoder = new NvJpgEncoder(streem->GetStream());

    auto pipeline = new EnginePipeline(trtEngine, bufferFrameGpu, bufferManager, gstDecoder,streem->GetStream(),settingPipeline,encoder);
    pipeline->StartPipeline(connectString);

    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto start = chrono::system_clock::now();

        vector<Detection> resultNms;

        uint64_t timeStamp;
        auto res = pipeline->GetResultImages(resultNms, timeStamp);
        if (!res)
        {
            continue;
        }

        std::vector<unsigned char>* encodedImage = pipeline->GetFrame();

        printf("   Target A size: %ld, %ld\n",encodedImage->size(), timeStamp);

        auto endCapture = chrono::system_clock::now();
        info("elapsed time: " + to_string(chrono::duration_cast<chrono::microseconds>(endCapture - start).count()));

        cv::Mat image = cv::imdecode(cv::_InputArray(encodedImage->data(), encodedImage->size()), cv::IMREAD_COLOR);
        DrawingResults(image, resultNms,timeStamp);
    }

    cout << "______ End Test_FullPass OK ______" << endl;
    */

}

int main(int argc, char* argv[])
{
    auto logPathFileString = "./Logs/Test_ConverterNetWeight.log";
    auto mainLogger = MainLogger(logPathFileString);
    auto modelInput = "../weight/model_001.engine";

    //Test_CreateTRTEngine
    //LoadTest(modelInput);
    //Test_FullPassYoloGPU(modelInput, true);
    // Test_matGpu_in_trt(modelInput);

    TestGstreamer();

    return 0;
}





