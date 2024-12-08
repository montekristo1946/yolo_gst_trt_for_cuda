#include "../infrastructure/ExtensionCharp.cpp"
#include "Helper.hpp"
#include "SettingPipeline.h"

void Test_ConverterNetWeight(const string modelInput, const string modelOutput)
{
    auto pathLog = CreatCharLineToString("./Logs/Test_ConverterNetWeight.log");
    InitLogger(pathLog);

    auto enginePathInput = CreatCharLineToString(modelInput);
    auto pathExportSave = CreatCharLineToString(modelOutput);


    auto config = new LayerSize(1, 3, 640, 640);
    bool setHalfModel = true;
    auto idGpu = 0;
    auto resConverterNetworkWeight = ConverterNetworkWeight(enginePathInput, pathExportSave, config, idGpu,
                                                            setHalfModel);
    delete config;
    if (!resConverterNetworkWeight)
        throw runtime_error("[Test_ConverterNetWeight] ConverterNetworkWeight");

    cout << "______ Test_ConverterNetWeight OK ______" << endl;
}

char* CreateConnectString()
{
    std::ostringstream ss2;
    ss2 << "filesrc location=/mnt/Disk_D/Document/Teplovisors/Dataset/010/11.09.2024_001.avi "
        << "! avidemux "
        << "! nvv4l2decoder "
        << "! nvvideoconvert nvbuf-memory-type=3 "
        << "! video/x-raw(memory:NVMM)"
        << "! appsink name=mysink sync=true";

    // ss2  << "rtspsrc location=rtsp://admin:jkluio789@192.168.1.15:554 latency=1000 "
    //                <<  "! rtph264depay "
    //              <<    "! nvv4l2decoder "
    //               <<   "! nvvideoconvert nvbuf-memory-type=3 "
    //               <<   "! video/x-raw(memory:NVMM) "
    //                <<  "! appsink name=mysink sync=true";

    auto mLaunchStr = ss2.str();
    char* pathLogArrChar = new char[mLaunchStr.length() + 1];
    strcpy(pathLogArrChar, mLaunchStr.c_str());
    return pathLogArrChar;
}

SettingPipeline* CreateSettingPipeline()
{
    auto settingPipeline = new SettingPipeline();
    settingPipeline->HeightImgMl = 640;
    settingPipeline->WidthImgMl = 640;
    settingPipeline->CountImgToBackground = 25;
    return settingPipeline;
}


void DrawingResultsSTrack(const Mat& mat, const vector<byte_track::BYTETracker::STrackPtr>& outputsMl)
{
    cv::Scalar colorText(0, 0, 255); // Green color
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.75;
    // auto rectsSrc = pipelineOutputData->Rectangles;
    // vector<RectDetect> rects = vector<RectDetect>(rectsSrc, rectsSrc + pipelineOutputData->RectanglesLen);
    for (auto &outputs_per_frame : outputsMl)
    {
        const auto &rect = outputs_per_frame->getRect();
        const auto &track_id = outputs_per_frame->getTrackId();


        auto width = rect.width();
        auto height = rect.height();
        auto x =(int) (rect.x() - rect.width() / 2);
        auto y = (int) (rect.y()- rect.height() / 2);

        // auto text = to_string(track_id);
        // auto labelId = outputs_per_frame->getScore();
        auto color = ColorInLabels[1];
        rectangle(mat, Rect(x, y, width, height), color, 1, 8, 0);
        cv::putText(mat, to_string(track_id), cv::Point(x, y), fontFace, fontScale, colorText);

        // auto width = (int)(rect.);
        // auto height = (int)(rect.Height*ImageHeight);
        // auto x = (int)((rect.X - rect.Width / 2) *ImageWidth);
        // auto y = (int)((rect.Y - rect.Height / 2) *ImageHeight);;
        // auto text = to_string(rect.Veracity);
        // auto color = ColorInLabels[(int)rect.IdClass];
        //
        // rectangle(mat, Rect(x, y, width, height), color, 1, 8, 0);
        // cv::putText(mat, to_string(rect.TimeStamp), cv::Point(10, ImageHeight-50), fontFace, fontScale, colorText);
    };

    imshow("Result", mat);
    waitKey(1);
}

void Test_init_pipeline(const char* model_output, bool isShow = true)
{
    printf("______  Test_init_pipeline start______  \n");
    auto pathLog = CreatCharLineToString("./Logs/Test_SunBunny.log");
    InitLogger(pathLog);
    auto cudaStream = CreateCudaStream();
    auto config = CreateTRTEngineConfig(model_output);
    auto trtEngine = CreateTRTEngine(config, cudaStream);
    auto bufferFrameGpu = CreateBufferFrameGpu();
    auto bufferManager = CreateGstBufferManager(bufferFrameGpu, cudaStream);

    auto gstDecoder = CreateGstDecoder(bufferManager);

    auto connectString = CreateConnectString();
    auto resConnect = StartPipelineGst(gstDecoder, connectString);
    if (!resConnect)
        throw runtime_error("[Test_init_pipeline] StartPipelineGst");

    auto encoder = CreateNvJpgEncoder(cudaStream);
    auto settingPipeline = CreateSettingPipeline();
    auto pipeline = CreateEnginPipeline(trtEngine, bufferFrameGpu, cudaStream, settingPipeline, encoder);

    auto maxCountDetectRectangle = 150;
    auto countImg = 10000;
    while (countImg > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto start = chrono::system_clock::now();

        std::vector<byte_track::BYTETracker::STrackPtr> resultNms;

        uint64_t timeStamp;
        auto res = pipeline->GetResultImages(resultNms, timeStamp);

        if (!res)
        {
            continue;
        }

        countImg--;

        ImageFrame* imageFrame = new ImageFrame();
        res = GetCurrenImage(pipeline, imageFrame);

        if (!res)
            throw runtime_error("[Test_init_pipeline] GetCurrenImage");

        auto endCapture = chrono::system_clock::now();
        info("elapsed time: " +
            to_string(chrono::duration_cast<chrono::microseconds>(endCapture - start).count()) +
            " resultNms.size: " + " " + to_string(resultNms.size()));

        if (imageFrame->ImageLen == 0)
            throw runtime_error("[Test_init_pipeline] encodedImage->size() == 0");

        if (isShow)
        {
            Mat image = cv::imdecode(cv::_InputArray(imageFrame->ImagesData, imageFrame->ImageLen), cv::IMREAD_COLOR);
            // DrawingResults(image, pipelineOutputData);
            DrawingResultsSTrack(image, resultNms);
        }
        // delete pipelineOutputData;
        delete imageFrame;
    }

    vector<IDispose*> dispose = {
        gstDecoder,
        trtEngine,
        bufferFrameGpu,
        bufferManager,
        encoder,
        pipeline
    };

    for (auto i = 0; i < dispose.size(); i++)
    {
        Dispose(dispose[i]);
    }
    //TODO: add test trt engine in ExtensionCharp.cpp
    printf("______  Test_init_pipeline OK______  \n");
}

void Test_memory_leak(const char* model_output)
{

    for (int i = 0; i < 100; ++i)
    {
        printf("process %i ---------------------- \n", i);
        Test_init_pipeline(model_output, false);
    }
    printf("______  Test_memory_leak OK______  \n");
}


void Test_reconnect_pipeline(const char* model_output, bool isShow = false)
{
    printf("______  Test_reconnect_pipeline start______  \n");
    auto pathLog = CreatCharLineToString("./Logs/Test_SunBunny.log");
    InitLogger(pathLog);
    auto cudaStream = CreateCudaStream();
    auto config = CreateTRTEngineConfig(model_output);
    auto trtEngine = CreateTRTEngine(config, cudaStream);
    auto bufferFrameGpu = CreateBufferFrameGpu();
    auto bufferManager = CreateGstBufferManager(bufferFrameGpu, cudaStream);
    auto connectString = CreateConnectString();
    auto encoder = CreateNvJpgEncoder(cudaStream);
    auto settingPipeline = CreateSettingPipeline();

    auto pipeline = CreateEnginPipeline(trtEngine, bufferFrameGpu, cudaStream, settingPipeline, encoder);

    auto countIterReconnect = 1000;
    while (countIterReconnect > 0)
    {

        auto gstDecoder = CreateGstDecoder(bufferManager);
        auto resConnect = StartPipelineGst(gstDecoder, connectString);
        if (!resConnect)
            throw runtime_error("[Test_init_pipeline] StartPipelineGst");

        auto countImg = 1000;
        while (countImg > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            auto pipelineOutputData = new PipelineOutputData();
            auto res = DoInferencePipeline(pipeline, pipelineOutputData);
            if (!res)
            {
                delete pipelineOutputData;
                continue;
            }
            countImg--;

            ImageFrame* imageFrame = new ImageFrame();
            res = GetCurrenImage(pipeline, imageFrame);
            if (!res)
                throw runtime_error("[Test_init_pipeline] GetCurrenImage");

            if (imageFrame->ImageLen == 0)
                throw runtime_error("[Test_init_pipeline] encodedImage->size() == 0");

            if (isShow)
            {
                Mat image = cv::imdecode(cv::_InputArray(imageFrame->ImagesData, imageFrame->ImageLen), cv::IMREAD_COLOR);
                DrawingResults(image, pipelineOutputData);
            }

            delete pipelineOutputData;
            delete imageFrame;
        }

        Dispose(gstDecoder);
        countIterReconnect--;
    }


    vector<IDispose*> dispose = {
        trtEngine,
        bufferFrameGpu,
        bufferManager,
        encoder,
        pipeline
    };

    for (auto i = 0; i < dispose.size(); i++)
    {
        Dispose(dispose[i]);
    }
    //TODO: add test trt engine in ExtensionCharp.cpp
    printf("______  Test_reconnect_pipeline OK______  \n");
}

int main(int argc, char* argv[])
{
    auto modelInput = "../weight/model_001.onnx";
    auto modelOutput = "../weight/model_001.engine";

    // Test_ConverterNetWeight(modelInput, modelOutput);
    Test_init_pipeline(modelOutput);
    // Test_memory_leak( modelOutput);
    // Test_reconnect_pipeline(modelOutput,true);
    return 0;
}
