#include "fas_structure.h"
#include "fas_structure-sim-opt.id.h"
#include "fas_structure-sim-opt.mem.h"
#include <iostream>
FasStructure::FasStructure() {
	net = new ncnn::Net;
	net->load_param(fas_structure_sim_opt_param_bin);
	net->load_model(fas_structure_sim_opt_bin);
}

FasStructure::~FasStructure() {
    if(net) {
        delete net;
        net = nullptr;
    }
}

float FasStructure::detect(unsigned char*pInBGRData, int nInCols, int nInRows, float *score) {
    float mean[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f}; 
	float stds[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f}; 
	ncnn::Mat indata = ncnn::Mat::from_pixels_resize(pInBGRData, ncnn::Mat::PIXEL_BGR, nInCols, nInRows, 60, 60);
	indata.substract_mean_normalize(mean, stds);
	ncnn::Mat out;
	ncnn::Extractor ex = net->create_extractor();
	ex.set_light_mode(true);
	ex.set_num_threads(1);
	ex.input(fas_structure_sim_opt_param_id::BLOB_input, indata);
	ex.extract(fas_structure_sim_opt_param_id::BLOB_output, out);
	if (out.w * out.h * out.c == 2) { 
		float score_3d = exp(((float*)out.data)[1]) / (exp(((float*)out.data)[0]) + exp(((float*)out.data)[1]));
		*score = score_3d;
		if(score_3d >= 0.5)      // 有额头
		{
			return 1;
		}
		else if(score_3d <= 0.4)  // 无额头
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
