#include "facerecg_interface.h"
#include <iostream>
#include <opencv2/opencv.hpp>

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
	int ret = FOSAFER_FaceRECOG_Detect(engine, img, 0);
	std::cout << "ret = " << ret << std::endl;
	std::cout << "face_num = " << img->face_num << std::endl;
	std::cout << "x = " << img->face_rect[0][0] << std::endl;
	std::cout << "y = " << img->face_rect[0][1] << std::endl;
	std::cout << "w = " << img->face_rect[0][2] << std::endl;
	std::cout << "h = " << img->face_rect[0][3] << std::endl;
	
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