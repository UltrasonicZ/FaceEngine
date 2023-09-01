#pragma once
#include "platform.h"
#include "net.h"

class FaceOcc
{
public:
	FaceOcc();
	~FaceOcc();
	int detect(unsigned char*pInBGRData, int nInRows, int nInCols);

private:
	ncnn::Net *net;
};

