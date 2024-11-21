#ifndef HELPER_TESTS_H
#define HELPER_TESTS_H

#include <experimental/filesystem>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>

#include "DtoToCharp.h"
#include "TRTEngine.hpp"

using namespace std;

const int ImageWidth = 640;
const int ImageHeight = 640;

char *CreatCharLineToString(string line) {
    char *pathLogArrChar = new char[line.length() + 1];
    strcpy(pathLogArrChar, line.c_str());
    return pathLogArrChar;
}


bool LoadCharImgToStream(const string imagePath, char **imgFileStream, unsigned int *sizeImg) {

    ifstream file(imagePath, ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        *sizeImg = file.tellg();
        file.seekg(0, file.beg);
        *imgFileStream = new char[*sizeImg];
        file.read(*imgFileStream, *sizeImg);
        file.close();
    }
    return true;
}

TRTEngineConfig *CreateTRTEngineConfig(string pathWeight){
    auto configTrt = new TRTEngineConfig();
    configTrt->DeviseId = 0;
    configTrt->EngineName = CreatCharLineToString(pathWeight);
    configTrt->ConfThresh = 0.1;
    configTrt->NmsThresh = 0.6;
    configTrt->MaxNumOutputBbox = 1000;

    return configTrt;
}

TRTEngine *CreateTRTEngine(TRTEngineConfig *configTrt){
    // auto trtEngine = new TRTEngine();
    //
    // auto result = trtEngine->InitTRTEngine(configTrt->EngineName,
    //                                        configTrt->DeviseId,
    //                                        configTrt->ConfThresh,
    //                                        configTrt->NmsThresh,
    //                                        configTrt->MaxNumOutputBbox);
    // if (result == false)
    //     throw runtime_error("[Test_CreateTRTEngine] InitTRTEngine false");
    //
    // return trtEngine;
    return nullptr;
}

void WriteOutPutToFile(const std::vector<float> &resultDl, const std::string &filename) {
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for (const auto &value: resultDl) {
            outputFile << value << " ";
        }
        outputFile.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

std::vector<float> ReadFileToVector(const std::string &filename) {
    std::vector<float> result;
    std::ifstream inputFile(filename);
    if (inputFile.is_open()) {
        std::string line;
        while (std::getline(inputFile, line)) {
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                result.push_back(value);
            }
        }
        inputFile.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
    return result;
}

//vector<ImageSrc> CreateImagesProcessing(vector<string> arrPathImages) {
//    vector<ImageSrc> imgInput;
//    for (auto path: arrPathImages) {
//        unsigned int sizeImgInput;
//        char *imgFileStream;
//        auto resLoadCharImgToStream = LoadCharImgToStream(path, &imgFileStream, &sizeImgInput);
//        if (!resLoadCharImgToStream)
//            throw runtime_error("[TestDataBuffer] LoadCharImgToStream");
//
//        ImageSrc imgProcesing;
//        imgProcesing.ImageLen = sizeImgInput;
//        imgProcesing.ImagesData = (unsigned char *) imgFileStream;
//
//        imgInput.push_back(imgProcesing);
//    }
//    return imgInput;
//}
//
//vector<float> LoadArrInFile(string pathFile) {
//    FileStorage storage(pathFile, FileStorage::READ);
//    Mat matrix;
//    storage["matName"] >> matrix;
//    storage.release();
//    float *pf = matrix.ptr<float>(0);
//    vector<float> resultDl{pf, pf + matrix.channels() * matrix.cols * matrix.rows};
//
//    return resultDl;
//}

map<int, Scalar> ColorInLabels = {
    {0, Scalar(255, 0, 0)},
    {1, Scalar(0, 255, 0)},
    {2, Scalar(0, 0, 255)},
    {3, Scalar(255, 255, 0)},
    {4, Scalar(255, 0, 255)}
};


void DrawingResults(Mat mat, PipelineOutputData * pipelineOutputData)
{
    cv::Scalar colorText(0, 0, 255); // Green color
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.75;
    auto rectsSrc = pipelineOutputData->Rectangles;
    vector<RectDetect> rects = vector<RectDetect>(rectsSrc, rectsSrc + pipelineOutputData->RectanglesLen);
    for (auto rect : rects)
    {
        auto width = (int)(rect.Width*ImageWidth);
        auto height = (int)(rect.Height*ImageHeight);
        auto x = (int)((rect.X - rect.Width / 2) *ImageWidth);
        auto y = (int)((rect.Y - rect.Height / 2) *ImageHeight);;
        auto text = to_string(rect.Veracity);
        auto color = ColorInLabels[(int)rect.IdClass];

        rectangle(mat, Rect(x, y, width, height), color, 1, 8, 0);
        cv::putText(mat, to_string(rect.TimeStamp), cv::Point(10, ImageHeight-50), fontFace, fontScale, colorText);
    };

    imshow("Result", mat);
    waitKey(1);
}

#endif //HELPER_TESTS_H