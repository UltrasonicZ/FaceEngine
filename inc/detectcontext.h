#pragma once
#include "detectstrategy.h"

class DetectContext {
private:
    DetectStrategy* strategy_;

public:
    DetectContext(DetectStrategy* strategy) : strategy_(strategy) {}

    void setStrategy(DetectStrategy* strategy) {
        strategy_ = strategy;
    }

    int detect(unsigned char*pInBGRData, int nInCols, int nInRows) const {
        return strategy_->detect(pInBGRData, nInCols, nInRows);
    }
};