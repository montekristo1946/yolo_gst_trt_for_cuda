#include <vector>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include "YoloToolsGPU.h"
#include <chrono>


using namespace std;


std::vector<float> ReadFileToVector(const std::string &filename) {
    std::vector<float> result;
    std::ifstream inputFile(filename);
    if (inputFile.is_open()) {
        std::copy(std::istream_iterator<float>(inputFile),
                  std::istream_iterator<float>(),
                  std::back_inserter(result));
        inputFile.close();
    } else {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
    return result;
}


void TestNmsGPU() {
    cudaStream_t _stream;
    auto resCudaStreamCreate = cudaStreamCreate(&_stream);
    if (resCudaStreamCreate != 0) {
        runtime_error("[TRTEngine::InitTRTEngine]  Not CudaStreamCreate {}");
    }

    vector<float> resultDl = ReadFileToVector("../examples/img_001_result.txt");

    float *ml_ptr_device = nullptr;
    cudaMallocAsync((void **) &ml_ptr_device, resultDl.size() * sizeof(float), _stream);
    auto resultCudaMemcopyHostToDevice = cudaMemcpy(ml_ptr_device,
                                                    resultDl.data(),
                                                    resultDl.size() * sizeof(float),
                                                    cudaMemcpyHostToDevice);
    if (resultCudaMemcopyHostToDevice != 0) {
        throw runtime_error(" [cudaMalloc] resCudaMalloc false");
    }

    int shapeBboxinOutLayer = 8400;
    float kConfThresh = 0.1;
    float kNmsThresh = 0.6f;
    const int countLabel = 4;
    int kMaxNumOutputBbox = 1000;

    auto _yoloToolsGPU = new YoloToolsGPU(shapeBboxinOutLayer,
                                          kConfThresh,
                                          kMaxNumOutputBbox,
                                          countLabel,
                                          kNmsThresh,
                                          _stream);

    for (int i = 0; i < 10000000; ++i) {
        auto start = chrono::system_clock::now();

        auto rects = _yoloToolsGPU->GetDetctionsBbox((float *) ml_ptr_device);

        auto checkCudaStreamSynchronize = cudaStreamSynchronize(_stream);

        if (checkCudaStreamSynchronize != 0) {
            throw runtime_error("[cudaMalloc] checkCudaStreamSynchronize false");
        }

        if (rects.size() != 17) {
            cout<<rects.size()<<endl;
            throw runtime_error("[GetDetctionsBbox] fail detect");

        }
        auto endAllProcess = chrono::system_clock::now();

        cout << "iter: " << i
             << " Nms time: " << chrono::duration_cast<chrono::microseconds>(endAllProcess - start).count()<< " microseconds"
             << endl;
    }
    cudaFree(ml_ptr_device);
    _yoloToolsGPU->~YoloToolsGPU();
}


int main(int argc, char *argv[]) {
    TestNmsGPU();
    return 0;
}
