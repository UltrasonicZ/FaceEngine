#include "facerecg_interface.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
    FACERECOG_ENGINE_HANDLE engine = FOSAFER_FaceRECOG_Initialize();
    // cv::Mat test_image = cv::imread("../data/images/foreigner.png");
    cv::Mat test_image = cv::imread("../data/images/gaozhou.jpg");
	Image* img = new Image;
	
	img->data = test_image.data;
	img->width = test_image.cols;
	img->height = test_image.rows;
	img->channel = test_image.channels();
	img->size = test_image.channels() * test_image.rows * test_image.cols;
	img->alive_score = 0.0;
	memcpy(img->data, test_image.data, test_image.cols * test_image.rows * test_image.channels());
	
	auto start = std::chrono::high_resolution_clock::now();

	int ret = FOSAFER_FaceRECOG_RGBDetect(engine, img, 0);

	DeepthImage *deepimg = new DeepthImage;
	cv::Mat deep_image = cv::imread("../data/images/gaozhou.jpg");	
	deepimg->data = deep_image.data;
	deepimg->width = deep_image.cols;
	deepimg->height = deep_image.rows;
	deepimg->channel = deep_image.channels();
	deepimg->alive_score = 0.0;
	memcpy(deepimg->data, deep_image.data, deep_image.cols * deep_image.rows * deep_image.channels());
	
	// DLL_PUBLIC int APIENTRY FOSAFER_FaceRECOG_DeepthDetect(FACERECOG_ENGINE_HANDLE pHandle, DeepthImage* image, float face_rect[4]);

	ret = FOSAFER_FaceRECOG_DeepthDetect(engine, deepimg, img->face_rect[0]);

	auto end = std::chrono::high_resolution_clock::now();

	std::cout << "ret = " << ret << std::endl;
	std::cout << "face_num = " << img->face_num << std::endl;
	std::cout << "mouth detect : " << img->detect_mouth << std::endl;
	std::cout << "lefteye detect : " << img->detect_lefteye << std::endl;
	std::cout << "righteye detect : " << img->detect_righteye << std::endl;
	std::cout << "lefteyebrow detect : " << img->detect_lefteyebrow << std::endl;
	std::cout << "righteyebrow detect : " << img->detect_righteyebrow << std::endl;
	std::cout << "nose detect : " << img->detect_nose << std::endl;
	std::cout << "chin detect : " << img->detect_chin << std::endl;
	std::cout << "forehead detect : " << img->detect_forehead << std::endl;
	std::cout << "face alive score : " << img->alive_score << std::endl;
	std::cout << "face 3D alive score : " << deepimg->alive_score << std::endl;

	std::chrono::duration<double> duration = end - start;

    // 输出运行时间（单位：秒）
    std::cout << "运行时间：" << duration.count() << " 秒" << std::endl;
	
	for(int i = 0; i < img->face_num; ++i) {
		cv::Rect rect;
		rect.x = img->face_rect[i][0];
		rect.y = img->face_rect[i][1];
		rect.width = img->face_rect[i][2];
		rect.height = img->face_rect[i][3];
		
        //cv::rectangle(test_image, rect, cv::Scalar(0, 255, 0), 2);
    }       

    cv::Rect large_face_rect;
    large_face_rect.x = img->face_rect[0][0];
    large_face_rect.y = img->face_rect[0][1];
    large_face_rect.width = img->face_rect[0][2];
    large_face_rect.height = img->face_rect[0][3];

    // cv::rectangle(test_image, large_face_rect, cv::Scalar(0, 255, 0), 2);
    // cv::imwrite("../data/images/result.jpg", test_image);
	// cv::imshow("result", test_image);
	// cv::waitKey(0);
	
	ret = FOSAFER_FaceRECOG_Release(engine);

	std::cout << "success" << std::endl;
	return 0;
}