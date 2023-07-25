#pragma once
#include "platform.h"
#include "net.h"

#include "detectstrategy.h"

class ForeheadExist : public DetectStrategy {
public:
	ForeheadExist();
	~ForeheadExist();
	virtual int detect(unsigned char*pInBGRData, int nInCols, int nInRows) override;
private:
	ncnn::Net *net;
};

