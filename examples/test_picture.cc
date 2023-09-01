#include "detectstrategy.h"
#include "chin_exist.h"
#include "eyebrow_exist.h"
#include "eye_exist.h"
#include "forehead_exist.h"
#include "mouth_exist.h"
#include "nose_exist.h"
#include "occ_exist.h"
#include "detectcontext.h"
#include "fas.h"

#include <opencv2/opencv.hpp>
#include <memory>

int main() {
    std::string image_path = "/home/storm/project/facedetection/data/images/nose.jpg";
    cv::Mat image = cv::imread(image_path);
    cv::Mat rgb_image;
    // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // cv::resize(image, image, cv::Size(64, 64));
    // cv::imshow("image", image);
    // cv::waitKey(0);

    std::shared_ptr<ChinExist> chin_detect = std::make_shared<ChinExist>();
    std::shared_ptr<EyebrowExist> eyebrow_detect = std::make_shared<EyebrowExist>();
    std::shared_ptr<EyeExist> eye_detect = std::make_shared<EyeExist>();
    std::shared_ptr<MouthExist> mouth_detect = std::make_shared<MouthExist>();
    std::shared_ptr<NoseExist> nose_detect = std::make_shared<NoseExist>();
    std::shared_ptr<ForeheadExist> forehead_detect = std::make_shared<ForeheadExist>();
    std::shared_ptr<FaceOcc> occ_detect = std::make_shared<FaceOcc>();
    
    std::shared_ptr<DetectContext> context = std::make_shared<DetectContext>(chin_detect.get());
    
    context->setStrategy(nose_detect.get());
    int detect_forehead = context->detect(image.data, image.cols, image.rows);
    std::cout << "detect_forehead : " << detect_forehead << std::endl;

    // float fasok;
    // std::shared_ptr<FaceFas> facefas = std::make_shared<FaceFas>();
    // facefas->detect(image.data, image.cols, image.rows, &fasok);
    
    // std::cout << "fas score : " << fasok << std::endl;

    return 0;
}