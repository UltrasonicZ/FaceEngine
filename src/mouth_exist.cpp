#include "mouth_exist.h"
#include "mouth_exist.id.h"
#include "mouth_exist.mem.h"
#include <iostream>

MouthExist::MouthExist() {
	net = nullptr;
	net = new ncnn::Net;
	net->load_param(mouth_exist_param_bin);
	net->load_model(mouth_exist_bin);
}

MouthExist::~MouthExist() {
	if (net){
		delete net;
		net = nullptr;
	}
}

int MouthExist::detect(unsigned char*pInBGRData, int nInCols, int nInRows) {
	float mean[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; 
	float stds[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f}; 
	ncnn::Mat indata = ncnn::Mat::from_pixels_resize(pInBGRData, ncnn::Mat::PIXEL_BGR2RGB, nInCols, nInRows, 64, 64);
	indata.substract_mean_normalize(mean, stds);
	ncnn::Mat out;
	ncnn::Extractor ex = net->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(1);
	ex.input(mouth_exist_param_id::BLOB_input, indata);
	ex.extract(mouth_exist_param_id::BLOB_output, out);
	if (out.w * out.h * out.c == 2) { 
		float score = exp(((float*)out.data)[0]) / (exp(((float*)out.data)[0]) + exp(((float*)out.data)[1]));
		if(score >= 0.7)      // 有嘴
		{
			return 1;
		}
		else if(score <= 0.2)  // 无嘴
		{
			return 2;
		}
		else
		{
			return 0;
		}
	}
}
