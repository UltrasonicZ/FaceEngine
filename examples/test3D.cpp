#include "facerecg_interface.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    FACERECOG_ENGINE_HANDLE engine = FOSAFER_FaceRECOG_Initialize();
    // cv::Mat test_image = cv::imread("../data/images/foreigner.png");
    cv::Mat test_image = cv::imread("../data/images/treat.jpg");
	DeepthImage* img = new DeepthImage;

	img->data = test_image.data;
	img->width = test_image.cols;
	img->height = test_image.rows;
	img->channel = test_image.channels();
	memcpy(img->data, test_image.data, test_image.cols * test_image.rows * test_image.channels());
	
	//int ret = FOSAFER_FaceRECOG_RGBDetect(engine, img, 0);
	float face_rect[4] = {0.0, 0.0, test_image.cols, test_image.rows};
	int ret = FOSAFER_FaceRECOG_DeepthDetect(engine, img, face_rect);
	
	ret = FOSAFER_FaceRECOG_Release(engine);

	std::cout << "success" << std::endl;
	return 0;
}
