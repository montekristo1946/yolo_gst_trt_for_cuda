#include <AlgorithmsPolygon.h>
#include <stdexcept>

AlgorithmsPolygon::AlgorithmsPolygon()
{
}

bool AlgorithmsPolygon::Predict(const vector<RectDetect>& source, vector<RectDetect>& destination)
{
    try
    {
        if(source.empty() || _polygonsSettings.empty())
            return true;

        for (const auto& rect : source)
        {
            auto isFindPolygon = false;

            for (const auto& polygon : _polygonsSettings)
            {
                if(IsPointInPolygon(polygon,rect))
                {
                    auto retRct = RectDetect(rect, polygon.Id);
                    destination.emplace_back(retRct);
                    isFindPolygon = true;
                    break;
                }
            }

            if(!isFindPolygon)
                destination.emplace_back(rect);

        }

        return true;
    }
    catch (exception& e)
    {
        _logger->error("[AlgorithmsPolygon::Predict]  {}", e.what());
    }
    catch (...)
    {
        _logger->error("[AlgorithmsPolygon::Predict]  Unknown exception!");
    }

    return false;
}

bool AlgorithmsPolygon::IsPointInPolygon( const Polygons& polygonComplex,const RectDetect& rect)
{
    auto polygon = polygonComplex.Points;
    auto point = Point(rect.X, rect.Y);
    int n = polygon.size();
    int count = 0;

    for (int i = 0; i < n; i++)
    {
        auto p1 = polygon[i];
        auto p2 = polygon[(i + 1) % n];

        if ((point.Y > min(p1.Y, p2.Y)) && (point.Y <= max(p1.Y, p2.Y))  && (point.X <= max(p1.X, p2.X)))
        {
            auto xIntersect = (point.Y - p1.Y) * (p2.X - p1.X) / (p2.Y - p1.Y) + p1.X;

            if (p1.X == p2.X || point.X <= xIntersect)
            {
                count++;
            }
        }
    }

    return count % 2 == 1;
}


AlgorithmsPolygon::~AlgorithmsPolygon()
{

}

void AlgorithmsPolygon::Clear()
{
    _polygonsSettings.clear();
    _logger->info("[AlgorithmsPolygon::Clear] Clear ok");
}

void AlgorithmsPolygon::AppendPolygon(const Polygons& polygon)
{
    _polygonsSettings.emplace_back(polygon);
    _logger->info("[AlgorithmsPolygon::AppendPolygon] Append ok");
}


