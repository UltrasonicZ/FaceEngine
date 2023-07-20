#include "facerecg_interface.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    FACERECOG_ENGINE_HANDLE engine = FOSAFER_FaceRECOG_Initialize();
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
		int ret = FOSAFER_FaceRECOG_RGBDetect(engine, img, 0);
		
		std::string face_percent = "face percent : " + std::to_string(img->face_percent);
		cv::Scalar font_color;
		if (img->face_percent > 0.15) {
			face_percent += " keep further";
			font_color = cv::Scalar(0, 0, 255);
		} else if (img->face_percent < 0.05) {
			face_percent += " keep closer";
			font_color = cv::Scalar(0, 0, 255);
		}
		else {
			font_color = cv::Scalar(0, 255, 0);
		}
		cv::putText(src, face_percent, cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 1, font_color, 2, 8, 0);
		
		std::string alive_score = "alive score : " + std::to_string(img->alive_score);
		if(img->alive_score > 0.3) {
			alive_score += " true";
			font_color = cv::Scalar(0, 255, 0);
		}else {
			alive_score += " false";
			font_color = cv::Scalar(0, 0, 255);
		}
		cv::putText(src, alive_score, cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 1, font_color, 2, 8, 0);
		
		std::string image_brightness = "image_brightness : " + std::to_string(img->image_average_gray_value);
		if(img->image_brightness == 0) {
			image_brightness += " low";
			font_color = cv::Scalar(0, 0, 255);
		}else if (img->image_brightness == 2) {
			image_brightness += " high";
			font_color = cv::Scalar(0, 255, 0);
		}
		else if (img->image_brightness == 1) {
			image_brightness += " medium";
			font_color = cv::Scalar(0, 255, 255);	
		}
		cv::putText(src, image_brightness, cv::Point(0, 90), cv::FONT_HERSHEY_SIMPLEX, 1, font_color, 2, 8, 0);

		std::string face_brightness = "face_brightness : " + std::to_string(img->face_average_gray_value);
		if(img->face_brightness == 0) {
			face_brightness += " low";
			font_color = cv::Scalar(0, 0, 255);
		}else if (img->face_brightness == 2) {
			face_brightness += " high";
			font_color = cv::Scalar(0, 255, 0);
		}
		else if (img->face_brightness == 1) {
			face_brightness += " medium";
			font_color = cv::Scalar(0, 255, 255);	
		}
		cv::putText(src, face_brightness, cv::Point(0, 120), cv::FONT_HERSHEY_SIMPLEX, 1, font_color, 2, 8, 0);

		std::string block = "block : ";
		if(img->detect_lefteye == 2) {
			block += "lefteye,";
		}
		if (img->detect_righteye == 2) {
			block += "righteye,";
		}
		if (img->detect_mouth == 2) {
			block += "mouth,";
		}
		if (img->detect_nose == 2) {
			block += "nose,";
		}
		if (img->detect_lefteyebrow == 2) {
			block += "lefteyebrow,";
		}
		if (img->detect_righteyebrow == 2) {
			block += "righteyebrow,";
		}
		if (img->detect_chin == 2) {
			block += "chin,";
		}
		font_color = cv::Scalar(0, 255, 0);	
		cv::putText(src, block, cv::Point(0, 150), cv::FONT_HERSHEY_SIMPLEX, 1, font_color, 2, 8, 0);

		std::string pitch = "pitch : " + std::to_string(img->pitch);
		cv::putText(src, pitch, cv::Point(0, 180), cv::FONT_HERSHEY_SIMPLEX, 1, font_color, 2, 8, 0);	
		std::string yaw = "yaw : " + std::to_string(img->yaw);
		cv::putText(src, yaw, cv::Point(0, 210), cv::FONT_HERSHEY_SIMPLEX, 1, font_color, 2, 8, 0);
		std::string roll = "roll : " + std::to_string(img->roll);
		cv::putText(src, roll, cv::Point(0, 240), cv::FONT_HERSHEY_SIMPLEX, 1, font_color, 2, 8, 0);
		std::string eye_dist = "eye distance : " + std::to_string(img->eye_dist);
		cv::putText(src, eye_dist, cv::Point(0, 270), cv::FONT_HERSHEY_SIMPLEX, 1, font_color, 2, 8, 0);
		
		cv::Rect rect;
		// rect.x = img->face_rect[0][0];
		// rect.y = img->face_rect[0][1];
		// rect.width = img->face_rect[0][2];
		// rect.height = img->face_rect[0][3];
		
		rect.x = img->face_rect[1][0];
		rect.y = img->face_rect[1][1];
		rect.width = img->face_rect[1][2];
		rect.height = img->face_rect[1][3];
		
        cv::rectangle(src, rect, cv::Scalar(0, 255, 0), 2);
		cv::imshow("test", src);
		if(cv::waitKey(30) == 'q') {
			ret = FOSAFER_FaceRECOG_Release(engine);
			break;
		}
	}
	
	return 0;
}