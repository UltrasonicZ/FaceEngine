#include "rgbalive.h"
#include "mat.h"
#include "model1.id.h"
#include "model1.mem.h"
#include "model2.id.h"
#include "model2.mem.h"
#include <algorithm>
#include <iostream>

RGBFacefas::RGBFacefas() {
    net1 = nullptr;
    net1 = new ncnn::Net;
    net1->load_param(model1_param_bin);
    net1->load_model(model1_bin);

    net2 = nullptr;
    net2 = new ncnn::Net;
    net2->load_param(model2_param_bin);
    net2->load_model(model2_bin);
}

RGBFacefas::~RGBFacefas() {
    if (net1){
		delete net1;
		net1 = nullptr;
	}
    if (net2){
		delete net2;
		net2 = nullptr;
	}
}

// float RGBFacefas::detect(unsigned char* pInBGRData1, unsigned char* pInBGRData2) {
//     float confidence = 0.f;
//     ncnn::Mat in = ncnn::Mat::from_pixels(pInBGRData1, ncnn::Mat::PIXEL_BGR, 80, 80);
//     // inference
//     ncnn::Extractor extractor1 = net1->create_extractor();
//     extractor1.set_light_mode(true);
//     extractor1.set_num_threads(4);
//     extractor1.input(model1_param_id::BLOB_input_1, in);
//     ncnn::Mat out;
//     extractor1.extract(model1_param_id::BLOB_523, out);
//     confidence += out.row(0)[1];
//     float b = exp(out.channel(0)[1]) / (exp(out.channel(0)[0]) + exp(out.channel(0)[1]) + exp(out.channel(0)[2]));
//     confidence += b;

//     ncnn::Mat in2 = ncnn::Mat::from_pixels(pInBGRData2, ncnn::Mat::PIXEL_BGR, 80, 80);
//     ncnn::Extractor extractor2 = net2->create_extractor();
//     extractor2.set_light_mode(true);
//     extractor2.set_num_threads(4);
//     extractor2.input(model2_param_id::BLOB_input_1, in2);
//     ncnn::Mat out2;
//     extractor2.extract(model2_param_id::BLOB_583, out2);
//     float b2 = exp(out2.channel(0)[1]) / (exp(out2.channel(0)[0]) + exp(out2.channel(0)[1]) + exp(out2.channel(0)[2]));
//     confidence += b2;

//     confidence /= 2;
//     return confidence;
// }

float RGBFacefas::detect(unsigned char* pInBGRData1, int in1cols, int in1rows, unsigned char* pInBGRData2, int in2cols, int in2rows) {
    std::vector<float> cls_scores1;
    ncnn::Mat in1 = ncnn::Mat::from_pixels_resize(pInBGRData1, ncnn::Mat::PIXEL_BGR, in1cols, in1rows, 80,80);
    ncnn::Extractor ex1 = net1->create_extractor();
    ex1.input(model1_param_id::BLOB_input_1, in1);
    ncnn::Mat out1;
    ex1.extract(model1_param_id::BLOB_523, out1);
    ncnn::Layer* softmax1 = ncnn::create_layer("Softmax");
    ncnn::ParamDict pb1;
    softmax1->load_param(pb1);
    softmax1->forward_inplace(out1, net1->opt);
    delete softmax1;
    out1 = out1.reshape(out1.h*out1.w*out1.c);
    
    cls_scores1.resize(out1.w);
    for (int j = 0; j < out1.w; j++)
    {
        cls_scores1[j] = out1[j];
        // printf("cls_scores1[%d]=%f\n",j,cls_scores1[j]);
    }

    std::vector<float> cls_scores2;
    ncnn::Mat in2 = ncnn::Mat::from_pixels_resize(pInBGRData2, ncnn::Mat::PIXEL_BGR, in2cols, in2rows, 80,80);
    ncnn::Extractor ex2 = net2->create_extractor();
    ex2.input(model2_param_id::BLOB_input_1, in2);
    ncnn::Mat out2;
    ex2.extract(model2_param_id::BLOB_583, out2);
 
    ncnn::Layer* softmax2=ncnn::create_layer("Softmax");
    ncnn::ParamDict pb2;
    softmax2->load_param(pb2);
    softmax2->forward_inplace(out2, net2->opt);
    delete softmax2;
    out2 = out2.reshape(out2.h*out2.w*out2.c);
 
    cls_scores2.resize(out2.w);
    for (int j = 0; j < out2.w; j++)
    {
        cls_scores2[j] = out2[j];
        // printf("cls_scores2[%d]=%f\n",j,cls_scores2[j]);
    }
    return (cls_scores1[1] + cls_scores2[1])/2;
}