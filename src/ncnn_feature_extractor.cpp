#include "ncnn_feature_extractor.h"
#include "landmarks_ljw.id.h"
#include "landmarks_ljw.mem.h"
#include "tool.h"
#include <cstdio>

//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/imgcodecs/imgcodecs.hpp"

CNCNNFeatureExtractor2::CNCNNFeatureExtractor2() {
	net_ = nullptr;
}

CNCNNFeatureExtractor2::~CNCNNFeatureExtractor2() {
	if(net_) {
        net_->clear();
		delete net_;
        net_ = nullptr;
	}
}

int CNCNNFeatureExtractor2::init() {
	int ret = 0;

	if(net_) {
		delete net_;
		net_ = nullptr;
	}

	net_ = new ncnn::Net();
	if(!net_) return -1;

    ret = net_->load_param(landmarks_ljw_params_bin);
    ret = net_->load_model(landmarks_ljw_bin);
    return 0;
}

int g_ncnndebug = 1;
float CNCNNFeatureExtractor2::ExtractFeature(cv::Mat const & image, std::vector<float> &ret) {
	ncnn::Extractor extractor = net_->create_extractor();
    std::vector<float> tmp;

	ret.clear();
    if(image.channels() != 3) { return -1;}
    //if(g_ncnndebug==1) { kcv::imwrite("/mnt/sdcard/fosafer/landmark_1.jpg",image); g_ncnndebug=0; }

	int w = image.cols; 
    int h = image.rows; 
	
	float mean[3] = { 0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f }; 
    float stds[3] = { 1 / 255.0f, 1 / 255.0f, 1 / 255.0f }; 
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 160, 160); 
    in.substract_mean_normalize(mean, stds); 

    ncnn::Mat prob, landmarks; 
    #ifdef __LINUX__
    extractor.set_num_threads(1);
    #endif
    extractor.set_light_mode(true);
    extractor.input(landmarks_ljw_params_id::BLOB_input_1, in); 
    extractor.extract(landmarks_ljw_params_id::BLOB_579, prob);
    extractor.extract(landmarks_ljw_params_id::BLOB_565, landmarks);   
    
    int size = landmarks.h * landmarks.w;
    //FLOG("97point prob: %d, %d, %d\n", prob.c, prob.h, prob.w); 
    //FLOG("97point mark: %d, %d, %d\n", landmarks.c, landmarks.h, landmarks.w); 

    for(int c = 0;c < landmarks.c;c++) {
        float *ptr = (float *)landmarks.data + landmarks.cstep * c;
        for(int z = 0;z < size;z++) {
            float value =0;
            if( z%2==0 ) value = ptr[z] * w ;
            if( z%2==1 ) value = ptr[z] * h ;
            //if(value < 0   ) value = 0;
            //if(value >=160 ) value = 159; 
            tmp.push_back(value);
        }
    }

    transformPts(tmp, ret);
    //FLOG("97point size：%d", tmp.size());
    //FLOG("82point size：%d", ret.size());
    //FLOG("97point prob: %f", ((float *)prob.data)[1]);

    return ((float *)prob.data)[1];

    // if(g_ncnndebug)
    // {
    //     cv::Mat debug;
    //     debug.create(image.size().height, image.size().width, image.channels() == 3 ? CV_8UC3 : CV_8UC1);
    //     memcpy(debug.data, image.data, image.size().height* image.size().width * image.channels());
    //     for(int j=0; j<ret.size()/2; j++)
    //         cv::circle(debug, cv::Point(ret[2*j+0], ret[2*j+1]), 1, cv::Scalar(0,255,0),1,8,0);
    //     cv::imwrite("/mnt/sdcard/fosafer/landmark_3.jpg",debug);
    //     g_ncnndebug=0;
    // }
}

void CNCNNFeatureExtractor2::transformPts(std::vector<float> &ret2, std::vector<float> &ret)
{
    if(ret2.size()==194)
    {
        //FLOG("97point alldata: ------------------------------------");
        //for(int i=0; i<97; i++)
        //{
        //    FLOG("97point alldata: %d %f %f", i, ret2[2*i+0], ret2[2*i+1]);
        //}
        //FLOG("97point alldata: ------------------------------------");

        int i = 0;
        ret.push_back(ret2[2*16+0]);    ret.push_back(ret2[2*16+1]);     // 00-00
        for(i=23; i>=17; i-- ) 
        {
            //if(i>18)FLOG("97point data: %f %f",ret2[2*i+0],ret2[2*i+1]);
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 01-07
        }
        ret.push_back(ret2[2*24+0]);    ret.push_back(ret2[2*24+1]);     // 08-08
        for(i=31; i>=25; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 09-15
        }
        ret.push_back(ret2[2*32+0]);    ret.push_back(ret2[2*32+1]);     // 16-16
        for(i=39; i>=33; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 17-23
        }
        ret.push_back(ret2[2*40+0]);    ret.push_back(ret2[2*40+1]);     // 24-24
        for(i=47; i>=41; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 25-31
        }

        ret.push_back(ret2[2*63+0]);    ret.push_back(ret2[2*63+1]);     // 32-32
        ret.push_back(ret2[2*60+0]);    ret.push_back(ret2[2*60+1]);     // 33-33
        ret.push_back(ret2[2*50+0]);    ret.push_back(ret2[2*50+1]);     // 34-34

        for(i=52; i<=54; i++ ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 35-37
        }
        ret.push_back(ret2[2*64+0]);    ret.push_back(ret2[2*64+1]);     // 38-38
        for(i=59; i>=57; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 39-41
        }
        ret.push_back(ret2[2*55+0]);    ret.push_back(ret2[2*55+1]);     // 42-42

        ret.push_back(ret2[2*96+0]);    ret.push_back(ret2[2*96+1]);     // 43-43
        ret.push_back(ret2[2*81+0]);    ret.push_back(ret2[2*81+1]);     // 44-44
        ret.push_back(ret2[2*83+0]);    ret.push_back(ret2[2*83+1]);     // 45-45
        ret.push_back(ret2[2*85+0]);    ret.push_back(ret2[2*85+1]);     // 46-46
        ret.push_back(ret2[2*86+0]);    ret.push_back(ret2[2*86+1]);     // 47-47
        ret.push_back(ret2[2*88+0]);    ret.push_back(ret2[2*88+1]);     // 48-48
        ret.push_back(ret2[2*89+0]);    ret.push_back(ret2[2*89+1]);     // 49-49

        for(i=73; i>=65; i-- ) 
        {
            if(i%2==1) { ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);} // 50-54
        }
        for(i=95; i>=91; i-- ) 
        {
            if(i%2==1) { ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);} // 55-57
        }
        for(i=75; i<=79; i++ ) 
        {
            if(i%2==1) { ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);} // 58-60
        }

        ret.push_back(ret2[2*7+0]);    ret.push_back(ret2[2*7+1]);      // 61-61
        ret.push_back(ret2[2*7+0]);    ret.push_back(ret2[2*7+1]);      // 62-62

        for(i=7; i>=1; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);   // 63-69
        }
        for(i=8; i<=15; i++ ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);   // 70-77
        }

        ret.push_back(ret2[2*15+0]);    ret.push_back(ret2[2*15+1]);   // 78-78
        ret.push_back(ret2[2*15+0]);    ret.push_back(ret2[2*15+1]);   // 79-79
        ret.push_back(ret2[2*48+0]);    ret.push_back(ret2[2*48+1]);   // 80-80
        ret.push_back(ret2[2*49+0]);    ret.push_back(ret2[2*49+1]);   // 81-81

        //FLOG("82point alldata: ------------------------------------");
        //for(int j=0; j<82; j++)
        //{
        //    FLOG("82point alldata: %d %f %f", j, ret[2*j+0], ret[2*j+1]);
        //}
        //FLOG("82point alldata: ------------------------------------");
    }
}