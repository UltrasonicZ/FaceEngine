#include "facerecg_interface.h"
#include <iostream>
#include<opencv2/opencv.hpp>

int main() {
    FACERECOG_ENGINE_HANDLE engine = FOSAFER_FaceRECOG_Initialize();
    //cv::Mat test_image = cv::imread("../data/images/foreigner.png");
    //cv::Mat test_image = cv::imread("../data/images/gaozhou.jpg");
	cv::VideoCapture cap = cv::VideoCapture(0);
	cv::Mat src;
	
	Image* img = new Image;
	cap.read(src);
	cv::namedWindow("test");
	cv::moveWindow("test", 200, 200);
	while (true) {
		if(!cap.isOpened()) {
			std::cout << "Read Video Failed!" << std::endl;
			return -1;
		}
		cap.read(src);
		cv::resize(src, src, cv::Size(800, 450));
		img->data = src.data;
		img->data = src.data;
		img->width = src.cols;
		img->height = src.rows;
		img->channel = src.channels();
		img->size = src.channels() * src.rows * src.cols;
		img->alive_score = 0.0;
		memcpy(img->data, src.data, src.cols * src.rows * src.channels());
		int ret = FOSAFER_FaceRECOG_Detect(engine, img, 0);
		std::cout << img->face_percent << std::endl;
		cv::putText(src, std::to_string(img->face_percent), cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 255), 2, 8, 0);
		cv::Rect rect;
		rect.x = img->face_rect[0][0];
		rect.y = img->face_rect[0][1];
		rect.width = img->face_rect[0][2];
		rect.height = img->face_rect[0][3];
		
        cv::rectangle(src, rect, cv::Scalar(0, 255, 0), 2);
		cv::imshow("test", src);
		if(cv::waitKey(30) == 'q') {
			ret = FOSAFER_FaceRECOG_Release(engine);
			break;
		}
	}
	
	return 0;
}