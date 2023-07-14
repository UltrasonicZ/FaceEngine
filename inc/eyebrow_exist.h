#pragma once
#include "platform.h"
#include "net.h"

#include "detectstrategy.h"

class EyebrowExist : public DetectStrategy {
public:
	EyebrowExist();
	~EyebrowExist();
	virtual int detect(unsigned char*pInBGRData, int nInCols, int nInRows) override;
private:
	ncnn::Net *net;
};

