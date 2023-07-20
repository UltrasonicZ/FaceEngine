#pragma once
#include "platform.h"
#include "net.h"

class FaceOcc
{
public:
	FaceOcc();
	~FaceOcc();
	bool detect(unsigned char*pInBGRData,int nInRows,int nInCols, float* faceok);
private:
	ncnn::Net *net;
};

