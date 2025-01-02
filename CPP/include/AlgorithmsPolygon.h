#ifndef ALGORITHMSPOLYGON_H
#define ALGORITHMSPOLYGON_H

#include <vector>
#include "DtoToCharp.h"
#include "IDispose.h"


using namespace std;

class AlgorithmsPolygon: public IDispose {
public:
    AlgorithmsPolygon(vector<Polygons>  & polygonsSettings);

    bool Predict(const vector<RectDetect>& source,   vector<RectDetect> & destination);

    ~AlgorithmsPolygon();
private:
    vector<Polygons>   _polygonsSettings;
    shared_ptr<logger> _logger = get("MainLogger");

    bool IsPointInPolygon(const Polygons& polygon,const RectDetect& rect);
};
#endif //ALGORITHMSPOLYGON_H
