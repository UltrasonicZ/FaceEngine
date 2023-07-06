#pragma once
#include "platform.h"
#include "net.h"

#include "detectstrategy.h"

class EyeExist : public DetectStrategy {
public:
	EyeExist();
	~EyeExist();
	virtual int detect(unsigned char*pInBGRData, int nInCols, int nInRows) override;
private:
	ncnn::Net *net;
};

