#include  <common/MainLogger.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void TestOpenCvImagesLoad() {
    auto image_path = "../examples/img_001.jpg";
    auto frame = imread(image_path, IMREAD_COLOR);

    imshow("img",frame);
    waitKey();
}

void TestOpenCvVideoLoad() {
    string filename = "rtsp://login:password@192.168.1.15:554/";
    VideoCapture capture(filename,cv::CAP_FFMPEG );
    Mat frame;

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    namedWindow( "w", 1);
    for( ; ; )
    {
        capture >> frame;
        if(frame.empty())
            break;
        imshow("w", frame);
        waitKey(20); // waits to display frame
    }
    waitKey(0);}

int main(int argc, char *argv[]) {

    auto mainLogger = MainLogger("../CPP/cmake-build-debug/logFile.log");

    shared_ptr<logger> _logger = spdlog::get("MainLogger");

    // TestOpenCvImagesLoad();
    TestOpenCvVideoLoad();
    _logger->info("Hello World!");
    return 0;
}
