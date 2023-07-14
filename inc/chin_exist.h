#pragma once
#include "platform.h"
#include "net.h"

#include "detectstrategy.h"

class ChinExist : public DetectStrategy {
public:
	ChinExist();
	~ChinExist();
	virtual int detect(unsigned char*pInBGRData, int nInCols, int nInRows) override;
private:
	ncnn::Net *net;
};

