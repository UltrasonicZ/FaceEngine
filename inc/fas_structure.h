#pragma once

#include "platform.h"
#include "net.h"

class FasStructure {
public:
    FasStructure();
    ~FasStructure();
    float detect(unsigned char*pInBGRData, int nInCols, int nInRows, float *score);
private:
	ncnn::Net *net;
};