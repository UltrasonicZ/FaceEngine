#pragma once
//create by ljw
#include "platform.h"
#include "net.h"
#include <vector>
#include<opencv2/opencv.hpp>

struct sFace
{
	int nlabel;
	float fprob;
	float xleft;
	float xright;
	float ytop;
	float ybottom;
};

struct ssdFaceRect
{
	int x, y, w, h;
	float confidence;
};

class ncnnssd
{
public:
	ncnnssd();
	~ncnnssd();
	
public:
	int clipBorder(int i, int lower, int upper);
	bool detect(unsigned char*pInBGRData,int nInRows,int nInCols, std::vector<ssdFaceRect> &rtfaces,
		        int ntargetrows=200, int ntargetcols = 200, float fminscore=0.6);

private:
	ncnn::Net * m_facenet;
	float mean_vals[3];
	float norm_vals[3];
};

