#include <iostream>
#include <MainLogger.hpp>
#include <vector>

#include "AlgorithmsPolygon.h"
using namespace std;

vector<Polygons> CreateExportDataPolygons()
{
    auto points1 = vector<Point>
    {
        {0.4f, 0.2f},
        {0.6f, 0.2f},
        {0.8f, 0.4f},
        {0.8f, 0.6f},
        {0.6f, 0.8f},
        {0.4f, 0.8f},
        {0.2f, 0.6f},
        {0.2f, 0.4f},
    };

    auto points2 = vector<Point>
    {
        {0.4f, 0.2f},
        {0.6f, 0.2f},
        {0.6f, 0.9f},
        {0.4f, 0.9f},
    };

    auto points3 = vector<Point>
    {
        {0.4f, 0.2f},
        {0.6f, 0.2f},
        {0.8f, 0.4f},
        {0.8f, 0.6f},
        {0.6f, 0.8f},
        {0.4f, 0.8f},
        {0.2f, 0.6f},
        {0.2f, 0.4f},
    };

    auto polygons = vector<Polygons>
    {
        {1, points1},
        {2, points2},
        {3, points3},
    };
    return polygons;
}

vector<RectDetect> CreateRectangles()
{
    auto rects = vector<RectDetect>
    {
        {0.3, 0.4, 0.2f, 0.2f, 1, 100, 0.5, 1000},
        {0.2, 0.2, 0.2f, 0.2f, 2, 100, 0.5, 1000},
        {0.3, 0.3, 0.2f, 0.2f, 3, 100, 0.5, 1000},
        {0.5, 0.85, 0.2f, 0.2f, 3, 100, 0.5, 1000},
        {0.75, 0.3, 0.2f, 0.2f, 3, 100, 0.5, 1000},
        {0.7, 0.8, 0.2f, 0.2f, 3, 100, 0.5, 1000},
        {0.35, 0.65, 0.2f, 0.2f, 3, 100, 0.5, 1000},
        {0.5, 0.5, 0.2f, 0.2f, 3, 100, 0.5, 1000},
    };
    return rects;
}

size_t GetCount(const std::vector<int>& idVectors, int idVector)
{
    auto count = std::count_if(idVectors.begin(), idVectors.end(),
        [idVector](const int& other) { return idVector == other; });

    return count;

}

void TestPredict()
{
    auto polygonsSettings = CreateExportDataPolygons();
    auto rectangle = CreateRectangles();

    auto algorithm = AlgorithmsPolygon();
    algorithm.Clear();
    for (const auto& polygon : polygonsSettings)
        algorithm.AppendPolygon(polygon);

    auto rects = vector<RectDetect>();
    auto res = algorithm.Predict(rectangle, rects);

    if (res == false)
        throw std::runtime_error("[TestPredict] Predict false");

    if (rects.size() != rectangle.size())
        throw std::runtime_error("[TestPredict] Predict rects.size() != rectangle.size()");

    if(rects[0].PolygonsId.size() != 2)
        throw std::runtime_error("[TestPredict] fail find polygon = 2");

    if(rects[1].PolygonsId.size() != 1)
        throw std::runtime_error("[TestPredict] fail find polygon = 1");

    if(rects[7].PolygonsId.size() != 3)
        throw std::runtime_error("[TestPredict] fail find polygon = 3");

    if (GetCount(rects[0].PolygonsId, 1) != 1)
        throw std::runtime_error("[TestPredict] fail find polygon = 1");

    if (GetCount(rects[0].PolygonsId, 3) != 1)
        throw std::runtime_error("[TestPredict] fail find polygon = 1");

    if (GetCount(rects[7].PolygonsId, 2) != 1)
        throw std::runtime_error("[TestPredict] fail find polygon = 1");

    auto tolerance = 0.0001;

    if (rects[0].X - rectangle[0].X > tolerance ||
        rects[0].Y - rectangle[0].Y > tolerance ||
        rects[0].Width - rectangle[0].Width > tolerance ||
        rects[0].Height - rectangle[0].Height > tolerance ||
        rects[0].IdClass != rectangle[0].IdClass ||
        rects[0].Veracity - rectangle[0].Veracity > tolerance ||
        rects[0].TrackId - rectangle[0].TrackId > tolerance ||
        rects[0].TimeStamp != rectangle[0].TimeStamp ||
        rects[0].PolygonsId.size() != 2)
        throw std::runtime_error("[TestPredict] fail copy fileds ");

    printf("______  TestPredict OK______  \n");
}

int main()
{
    string logPathFileString = "./Logs/TestPointInPolygon.log";
    auto mainLogger = MainLogger(logPathFileString);

    TestPredict();

    return 0;
}
