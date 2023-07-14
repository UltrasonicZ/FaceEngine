#pragma once
#include "platform.h"
#include "net.h"

#include "detectstrategy.h"

class NoseExist : public DetectStrategy {
public:
	NoseExist();
	~NoseExist();
	virtual int detect(unsigned char*pInBGRData, int nInCols, int nInRows) override;
private:
	ncnn::Net *net;
};

