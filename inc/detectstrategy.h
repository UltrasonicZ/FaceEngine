#pragma once

// 声明抽象的检测策略接口
class DetectStrategy {
public:
    virtual int detect(unsigned char*pInBGRData, int nInCols, int nInRows) = 0;
};