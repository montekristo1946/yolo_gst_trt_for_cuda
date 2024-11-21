//
// Created by root on 4/6/22.
//

#ifndef TENSORRTTOOLSWEDGE_TRTENGINECONFIG_H
#define TENSORRTTOOLSWEDGE_TRTENGINECONFIG_H
#include "Common.h"

struct TRTEngineConfig {
public:
    const char *EngineName;
    unsigned int DeviseId;
    float ConfThresh = 0.1;
    float NmsThresh = 0.6f;
    unsigned int MaxNumOutputBbox = 1000;
};

#endif //TENSORRTTOOLSWEDGE_TRTENGINECONFIG_H
