#pragma once

#include "platform.h"
#include "net.h"

class FasStructure {
public:
    FasStructure();
    ~FasStructure();
    int detect(unsigned char*pInBGRData, int nInCols, int nInRows);
private:
	ncnn::Net *net;
};