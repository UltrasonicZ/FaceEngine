#include "occ_exist.h"
#include "occ_sim_opt.id.h"
#include "occ_sim_opt.mem.h"
#include <iostream>

FaceOcc::FaceOcc()
{
	net = new ncnn::Net;
	net->load_param(occ_sim_opt_param_bin);
	net->load_model(occ_sim_opt_bin);
}

FaceOcc::~FaceOcc()
{
	if (net){
		delete net;
	}
	net = nullptr;
}

int FaceOcc::detect(unsigned char*pInBGRData, int nInCols, int nInRows) {
	float mean[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; 
	float stds[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f}; 
	ncnn::Mat indata = ncnn::Mat::from_pixels_resize(pInBGRData, ncnn::Mat::PIXEL_BGR2RGB, nInCols, nInRows, 64, 64);
	indata.substract_mean_normalize(mean, stds);
	ncnn::Mat out;
	ncnn::Extractor ex = net->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(1);
	ex.input(occ_sim_opt_param_id::BLOB_input, indata);
	ex.extract(occ_sim_opt_param_id::BLOB_output, out);
	// printf("dimension: w:%d, h:%d, c:%d\n", out.w * out.h * out.c);
	// printf("out data : data0: %f, data1: %f\n", ((float*)out.data)[0], ((float*)out.data)[1]);
	// *faceok = exp(((float*)out.data)[0]) / (exp(((float*)out.data)[0]) + exp(((float*)out.data)[1])); 
	// *faceok = ((float*)out.data)[0];
	if (out.w * out.h * out.c == 2) 
	{ 
		float score = exp(((float*)out.data)[0]) / (exp(((float*)out.data)[0]) + exp(((float*)out.data)[1]));
		std::cout << "occ score : " << score << std::endl;
		if(score >= 0.5)      // 有鼻子
		{
			return 1;
		}
		else if(score <= 0.4)  // 无鼻子
		{
			return 2;
		}
		else
		{
			return 0;
		}
	}
	return -1;
	return true;
}
