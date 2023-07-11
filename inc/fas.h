#pragma once
#include "platform.h"
#include "net.h"

class FaceFas
{
public:
	FaceFas();
	~FaceFas();
	bool detect(unsigned char*pInBGRData,int nInRows,int nInCols, float* faceok);
private:
	ncnn::Net *net;
};

