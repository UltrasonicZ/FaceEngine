#include "eye_exist.h"
#include "eye_exist.id.h"
#include "eye_exist.mem.h"
#include <iostream>

EyeExist::EyeExist()
{
	net = NULL;
	net = new ncnn::Net;
	net->load_param(eye_exist_param_bin);
	net->load_model(eye_exist_bin);
}

EyeExist::~EyeExist()
{
	if (net){
		delete net;
		net = NULL;
	}
}

int EyeExist::detect(unsigned char*pInBGRData, int nInCols, int nInRows) {
	float mean[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; 
	float stds[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f}; 
	ncnn::Mat indata = ncnn::Mat::from_pixels_resize(pInBGRData, ncnn::Mat::PIXEL_BGR2RGB, nInCols, nInRows, 64, 64);
	indata.substract_mean_normalize(mean, stds);
	ncnn::Mat out;
	ncnn::Extractor ex = net->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(1);
	ex.input(eye_exist_param_id::BLOB_input, indata);
	ex.extract(eye_exist_param_id::BLOB_output, out);
	if (out.w * out.h * out.c == 2) 
	{ 
		float score = exp(((float*)out.data)[0]) / (exp(((float*)out.data)[0]) + exp(((float*)out.data)[1]));
		std::cout << "eye score : " << score << std::endl;
		if(score >= 0.5)      // 有眼睛
		{
			return 1;
		}
		else if(score <= 0.4)  // 无眼睛
		{
			return 2;
		}
		else
		{
			return 0;
		}
	}
	return -1;
}
