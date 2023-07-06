#pragma once
#include "platform.h"
#include "net.h"

#include "detectstrategy.h"

class MouthExist : public DetectStrategy {
public:
	MouthExist();
	~MouthExist();
	virtual int detect(unsigned char*pInBGRData, int nInCols, int nInRows) override;
private:
	ncnn::Net *net;
};

