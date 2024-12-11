#include <set>
#include <TrackerManager.h>
#include <unordered_set>


TrackerManager::TrackerManager(
    const int& frameRate,
    const int& trackBuffer,
    const float& trackThresh,
    const float& highThresh,
    const float& matchThresh,
    const int& maxNumTrackers)
    : _frameRate(frameRate)
      , _trackBuffer(trackBuffer)
      , _trackThresh(trackThresh)
      , _highThresh(highThresh)
      , _matchThresh(matchThresh)
      , _maxNumTrackers(maxNumTrackers)
{
    if (maxNumTrackers < 0)
        throw std::invalid_argument("maxNumTrackers should be > 0");

    if (frameRate <= 0)
        throw std::invalid_argument("frameRate should be > 0");


    if (trackBuffer <= 0)
        throw std::invalid_argument("trackBuffer should be > 0");


    if (trackThresh < 0 || trackThresh > 1)
        throw std::invalid_argument("trackThresh should be >= 0 and <= 1");


    if (highThresh < 0 || highThresh > 1)
        throw std::invalid_argument("highThresh should be >= 0 and <= 1");


    if (matchThresh < 0 || matchThresh > 1)
        throw std::invalid_argument("matchThresh should be >= 0 and <= 1");


    CraeteTreackers(_maxNumTrackers);
}

 bool TrackerManager::Predict(const vector<Detection>& source,  uint64_t timeStamp, vector<RectDetect> & retRectDetect)
{
    try
    {
        if(source.empty())
            return false;

        unordered_set<int> uniqueIds;
        for (const auto& rect : source) {
            uniqueIds.insert((int)rect.ClassId);
        }

        for (const auto& idLabel : uniqueIds)
        {
            vector<Detection> currentIdDetection;
            ranges::copy_if(source, std::back_inserter(currentIdDetection),
                            [idLabel](Detection rect) { return rect.ClassId == (float)idLabel; });

            std::vector<Object> objects;
            for (auto& det : currentIdDetection)
            {
                objects.emplace_back(Rect(det.BBox[0],
                    det.BBox[1], det.BBox[2], det.BBox[3]), det.ClassId, det.Conf);
            }

            auto tracker = _trackers->find(idLabel)->second;;
            if(!tracker)
            {
                error("[TrackerManager::Predict]  Tracker not found {}", idLabel);
                continue;
            }

            auto resTracke = tracker->update(objects);

            for ( auto& track : resTracke)
            {
                const auto &rect = track->getRect();
                const auto &trackId = track->getTrackId();

                retRectDetect.emplace_back(RectDetect(
                    rect.x(),
                    rect.y(),
                    rect.width(),
                    rect.height(),
                    idLabel,
                    timeStamp,
                    track->getScore(),
                    trackId));
            }
        }

        return true;
    }
    catch (exception& e)
    {
       error("[TrackerManager::Predict]  {}", e.what());
    }
    catch (...)
    {
       error("[TrackerManager::Predict]  Unknown exception!");
    }
    return false;
}

TrackerManager::~TrackerManager()
{
    for (auto it = _trackers->begin(); it != _trackers->end(); ++it)  {
        BYTETracker* value =  it->second;
        delete value;
    }

    if (_trackers)
        delete _trackers;

    info("[TrackerManager::~TrackerManage] Call destructor");
}

void TrackerManager::CraeteTreackers(int cuntTreackers)
{
    for (int i = 0; i < cuntTreackers; i++)
    {
        _trackers->insert(
            std::make_pair(i, new BYTETracker(_frameRate, _trackBuffer, _trackThresh, _highThresh, _matchThresh)));
    }

}
