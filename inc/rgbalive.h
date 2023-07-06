#pragma once
#include "platform.h"
#include "net.h"

class RGBFacefas {
public:
    RGBFacefas();
    ~RGBFacefas();
    float detect(unsigned char* pInBGRData1, int in1cols, int in1rows, unsigned char* pInBGRData2, int in2cols, int in2rows);

private:
    ncnn::Net *net1;
    ncnn::Net *net2;
};

