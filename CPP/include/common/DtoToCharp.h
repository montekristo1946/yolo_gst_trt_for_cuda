#ifndef DTOTOCHARP_H
#define DTOTOCHARP_H
#include <cstdint>
#include <vector>

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
        PolygonId = -1;
    }

    RectDetect(const RectDetect& rect, int polygonId)
    {
        X = rect.X;
        Y = rect.Y;
        Width = rect.Width;
        Height = rect.Height;
        IdClass = rect.IdClass;
        TimeStamp = rect.TimeStamp;
        Veracity = rect.Veracity;
        TrackId = rect.TrackId;
        PolygonId = polygonId;
    }

    float X;
    float Y;
    float Width;
    float Height;
    int IdClass;
    float Veracity;
    int TrackId;
    uint32_t   TimeStamp;
    int PolygonId;

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
        ImageLen = 0;
        TimeStamp =0;
    }
    unsigned char *ImagesData;
    unsigned int ImageLen;
    uint64_t TimeStamp;

    ~ImageFrame() {
        if(ImagesData)
            delete[] ImagesData;
    }
};

struct Point {
    int Id;
    float X;
    float Y;
};


struct Polygons
{
    int Id;
    std::vector<Point> Points;
};

struct PolygonExternal
{

    int Id;
    Point * PointPtr;
    unsigned int PointLen;

    ~PolygonExternal() {
        if(PointPtr)
            delete[] PointPtr;
    }

};

struct PolygonsSettingsExternal
{
    PolygonExternal * Polygons;
    unsigned int PolygonsLen;
    ~PolygonsSettingsExternal() {
        if(Polygons)
            delete[] Polygons;
    }
};

#endif //DTOTOCHARP_H
