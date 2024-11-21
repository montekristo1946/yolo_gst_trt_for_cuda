#ifndef TENSORRTTOOLSWEDGE_IDISPOSE_H
#define TENSORRTTOOLSWEDGE_IDISPOSE_H
#include <spdlog/spdlog.h>

using namespace std;
using namespace spdlog;

class IDispose{
public:
    IDispose(){
    }
    virtual ~IDispose() {
    };

};

#endif //TENSORRTTOOLSWEDGE_IDISPOSE_H
