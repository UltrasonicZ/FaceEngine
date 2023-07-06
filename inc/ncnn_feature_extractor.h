#ifndef __NCNN_FEATURE_EXTRACTOR2__
#define __NCNN_FEATURE_EXTRACTOR2__

#include <vector>
#include <string>
#include <net.h>

#include<opencv2/opencv.hpp>

class CNCNNFeatureExtractor2 {
public:
	CNCNNFeatureExtractor2();
	~CNCNNFeatureExtractor2();

	int init();
	float ExtractFeature(cv::Mat const & image, std::vector<float> &ret);
	void  transformPts(std::vector<float> &ret2, std::vector<float> &ret);

private:
	ncnn::Net *net_;
	std::string str_landmark_ncnn_proto_bin_;
    std::string str_landmark_ncnn_weights_;
};

#endif