//
// Created by user on 19.11.2024.
//

#ifndef DTOTOCHARP_H
#define DTOTOCHARP_H
#include <cstdint>


struct RectDetect
{
public:
    RectDetect()
    {
        X = -1;
        Y = -1;
        Width = -1;
        Height = -1;
        IdClass =-1;
        TimeStamp = 0;
        Veracity = -1;
        TrackId = -1;
    }

    RectDetect(float x, float y, float width, float height, int idClass, uint64_t timeStamp, float veracity, int trackId)
    {
        X = x;
        Y = y;
        Width = width;
        Height = height;
        IdClass = idClass;
        TimeStamp = timeStamp;
        Veracity = veracity;
        TrackId = trackId;
    }

    float X;
    float Y;
    float Width;
    float Height;
    int IdClass;
    float Veracity;
    uint64_t TimeStamp;
    int TrackId;
};

struct PipelineOutputData {
    PipelineOutputData() {
        Rectangles = nullptr;
        RectanglesLen = 0;
    }

~PipelineOutputData() {
        if (Rectangles)
            delete [] Rectangles;
}
    RectDetect *Rectangles;
    unsigned int RectanglesLen;
};


struct ImageFrame {
    ImageFrame()
    {
        ImagesData = nullptr;
        ImageLen = -1;
        TimeStamp =0;
    }
    unsigned char *ImagesData;
    int ImageLen;
    uint64_t TimeStamp;

    ~ImageFrame() {
        if(ImagesData)
            delete[] ImagesData;
        // if(TimeStamp)
        //     delete TimeStamp;
    }
};



#endif //DTOTOCHARP_H
