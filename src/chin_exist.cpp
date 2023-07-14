#include "chin_exist.h"
#include "chin-sim-opt.id.h"
#include "chin-sim-opt.mem.h"
#include <iostream>

ChinExist::ChinExist()
{
	net = NULL;
	net = new ncnn::Net;
	net->load_param(chin_sim_opt_param_bin);
	net->load_model(chin_sim_opt_bin);
}

ChinExist::~ChinExist()
{
	if (net){
		delete net;
		net = NULL;
	}
}

int ChinExist::detect(unsigned char*pInBGRData, int nInCols, int nInRows) {
	float mean[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; 
	float stds[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f}; 
	ncnn::Mat indata = ncnn::Mat::from_pixels_resize(pInBGRData, ncnn::Mat::PIXEL_BGR2RGB, nInCols, nInRows, 64, 64);
	indata.substract_mean_normalize(mean, stds);
	ncnn::Mat out;
	ncnn::Extractor ex = net->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(1);
	ex.input(chin_sim_opt_param_id::BLOB_input, indata);
	ex.extract(chin_sim_opt_param_id::BLOB_output, out);
	if (out.w * out.h * out.c == 2) 
	{ 
		float score = exp(((float*)out.data)[0]) / (exp(((float*)out.data)[0]) + exp(((float*)out.data)[1]));
		std::cout << "chin out.data[0]:" << exp(((float*)out.data)[0]) / (exp(((float*)out.data)[0]) + exp(((float*)out.data)[1])) << std::endl;
		std::cout << "chin out.data[1]:" << exp(((float*)out.data)[1]) / (exp(((float*)out.data)[0]) + exp(((float*)out.data)[1])) << std::endl;
		
		std::cout << "chin score : " << score << std::endl;
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
