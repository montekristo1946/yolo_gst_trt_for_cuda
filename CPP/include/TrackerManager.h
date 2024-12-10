//
// Created by user on 09.12.2024.
//

#ifndef TRACKERMANAGER_H
#define TRACKERMANAGER_H
#include "DtoToCharp.h"
#include "IDispose.h"
#include "YoloToolsGPU.h"
#include "ByteTrack/BYTETracker.h"

using namespace std;
using namespace byte_track;

class TrackerManager : public IDispose
{
public:
    TrackerManager(const int& frameRate = 30,
                   const int& trackBuffer = 30,
                   const float& trackThresh = 0.2,
                   const float& highThresh = 0.5,
                   const float& matchThresh = 0.7,
                   const int& maxNumTrackers = 0);

    bool Predict(const vector<Detection>& source, uint64_t timeStamp,vector<RectDetect> & retRectDetect);

private:
    ~TrackerManager();
    void CraeteTreackers(int cuntTreackers);

    int _frameRate = 0;
    int _trackBuffer = 0;
    float _trackThresh = 0;
    float _highThresh = 0;
    float _matchThresh = 0;
    // BYTETracker* _tracker = new BYTETracker(_frameRate, _trackBuffer, _trackThresh, _highThresh, _matchThresh);
    std::map<int, BYTETracker*>* _trackers = new map<int, BYTETracker*>();

    // vector<BYTETracker>* _trackers = new vector<BYTETracker>();
    const int& _maxNumTrackers;
};
#endif //TRACKERMANAGER_H
