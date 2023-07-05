#include "facerecg_interface.h"
#include <algorithm>
#include <vector>

const double threshold_dark = 100.0;
const double threshold_brightness = 220.0;

// {v0, v1, v2, v3}的默认值为{70, 100, 210, 230}
// 亮度区间：
// [0, v0), [v3, ~) => LOW
// [v0, v1), [v2, v3) => MEDIUM
// [v1, v2) => HIGH
const double threshold_v0 = 70.0;
const double threshold_v1 = 100.0;
const double threshold_v2 = 210.0;
const double threshold_v3 = 230.0;

// 人脸分辨率评估：
// {low, high}默认值为{80, 120} ,其映射关系为
// [0, low) => LOW
// [low, high) => MEDIUM
// [high, ~) => HIGH
const double face_resol_low = 80.0;
const double face_resol_high = 120.0;

const double face_paper_thresh = 40.0;

struct FaceBox {
    float alive_score;
    float x1;
    float y1;
    float x2;
    float y2;
};

CFosaferFaceRecogBackend::CFosaferFaceRecogBackend() {
    
}

CFosaferFaceRecogBackend::~CFosaferFaceRecogBackend(){
    if (ncnnssd_) {
        delete ncnnssd_;
        ncnnssd_ = NULL;
    }
}

bool CFosaferFaceRecogBackend::init() {
    ncnnssd_ = new ncnnssd();
    return true;
}

bool compareVector(const ssdFaceRect &a, const ssdFaceRect &b){
    return a.w * a.h > b.w * b.h;
}

void CFosaferFaceRecogBackend::detect_brightness(cv::Mat input_img, float& cast, float& da){
    cv::Mat gray_image;
    cv::cvtColor(input_img, gray_image, cv::COLOR_BGR2GRAY);
    
    float sum_deviation, average_deviation;
    std::vector<int> hist(256, 0);
    for (int i = 0; i < gray_image.rows; ++i) {
        for (int j = 0; i < gray_image.cols; ++j) {
            sum_deviation += gray_image.at<uchar>(i, j) - 128;
            hist[gray_image.at<uchar>(i, j)]++;
        }
    }
    average_deviation = sum_deviation / float(gray_image.total());   
}

double CFosaferFaceRecogBackend::calculate_average_gray_value(const cv::Mat& image_color, const ssdFaceRect &face) {
    double total_gray_value = 0.0;
    //cv::Mat image_gray;
    //cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);
    double total_blue_value = 0.0;
    double total_green_value = 0.0;
    double total_red_value = 0.0;
    for (int i = face.x; i < face.x + face.w; ++i) {
        for (int j = face.y; j < face.y + face.h; ++j) {
            //total_gray_value += static_cast<double>(image_gray.at<uchar>(i, j));
            total_blue_value += image_color.at<cv::Vec3b>(i,j)[0];
            total_green_value += image_color.at<cv::Vec3b>(i,j)[1];
            total_red_value += image_color.at<cv::Vec3b>(i,j)[2];
        }
    }
    double average_gray_value = total_gray_value / (face.w * face.h);
    
    double average_blue_value = total_blue_value / (face.w * face.h);
    double average_green_value = total_green_value / (face.w * face.h);
    double average_red_value = total_red_value / (face.w * face.h);
    // std::cout << "RGB:" << std::endl;
    // std::cout << average_blue_value << std::endl;
    // std::cout << average_green_value << std::endl;
    // std::cout << average_red_value << std::endl;
    
    return average_gray_value;
}
double CFosaferFaceRecogBackend::cal_variance(const cv::Mat& image_color, const ssdFaceRect &face) {
    cv::Scalar mean;
    mean = cv::mean(image_color);
    
    // 图像均值 和 标准方差
    cv::Mat meanMat, stdMat;
 
    cv::meanStdDev(image_color, meanMat, stdMat);
 
    // std::cout << "图像均值 和 标准方差" << std::endl;
    // std::cout << meanMat << std::endl;
    // std::cout << stdMat << std::endl;
 
    //std::cout << meanMat.at<double>(0) << " " << meanMat.at<double>(1) << " " << meanMat.at<double>(2) << std::endl;
    //std::cout << stdMat.at<double>(0) << " " << stdMat.at<double>(1) << " " << stdMat.at<double>(2) << std::endl;
    return (stdMat.at<double>(0) + stdMat.at<double>(1) + stdMat.at<double>(2)) / 3;
}

int CFosaferFaceRecogBackend::detect(Image* image_input, int rotateCW) {
    if (!image_input) { 
        return FACE_ERROR_NULL; 
    }
    cv::Mat image_color;
    cv::Mat image_color_small;
    cv::Rect big_rect;
    cv::Mat gray_image_small;

    cv::Mat image_buf2((image_input->height), image_input->width, CV_8UC3, image_input->data);
    image_color.create(image_buf2.size().height, image_buf2.size().width, image_buf2.channels() == 3 ? CV_8UC3 : CV_8UC1);
    memcpy(image_color.data, image_buf2.data, image_buf2.size().height* image_buf2.size().width * image_buf2.channels());
    rotate_image_90n(image_color, image_color, rotateCW);

    if (image_color.empty()) {
        return FACE_ERROR_DECODE;
    }
    if (-1 != image_max_dim_) { 
        image_color_small = ResizeImage(image_color, image_max_dim_, &scale_factor_); 
    }

    // 人脸检测,记得要缩放图像到400
    std::vector<ssdFaceRect> faces;
    std::vector<ssdFaceRect> faces_success;
    bool ret = ncnnssd_->detect(image_color_small.data, image_color_small.size().height, image_color_small.size().width, faces, 260, 260, 0.65);
    if(!ret){
        return FACE_DET_FAILED;
    }
    if(faces.size() == 0){
        image_input->face_num = 0;
        return FACE_NO_FACE;
    }
    // if(faces.size() > 1) {
    //      image_input->face_num = 0;
    //      return FACE_ERROR_MULTIFACE;
    // }
    for(int i = 0; i < faces.size(); i++){
        ssdFaceRect face_tmp;
        face_tmp.x = faces.at(i).x;
        face_tmp.y = faces.at(i).y;
        face_tmp.w = faces.at(i).w;
        face_tmp.h = faces.at(i).h;
        face_tmp.confidence = faces.at(i).confidence;
        if(face_tmp.confidence > 0.65)
            faces_success.push_back(face_tmp); 
    }
    if(faces_success.size() == 0){
        image_input->face_num = 0;
        return FACE_NO_FACE;
    }

    std::sort(faces_success.begin(), faces_success.end(), compareVector);
    image_input->face_num = faces_success.size();
    for(int i = 0; i < faces_success.size(); i++){
        image_input->face_rect[i][0] = faces_success.at(i).x * scale_factor_;
        image_input->face_rect[i][1] = faces_success.at(i).y * scale_factor_;
        image_input->face_rect[i][2] = faces_success.at(i).w * scale_factor_;
        image_input->face_rect[i][3] = faces_success.at(i).h * scale_factor_;
        image_input->face_rect[i][4] = faces_success.at(i).confidence;
    }

    //cv::imwrite("gray.jpg", gray_image_small);
    //计算均方差
    double aver_variance = cal_variance(image_color_small, faces_success[0]);
    if(aver_variance < face_paper_thresh){
        return FACE_LOW_QUALITY;
    }
    // {v0, v1, v2, v3}的默认值为{70, 100, 210, 230}
    // 亮度区间：
    // [0, v0), [v3, ~) => LOW
    // [v0, v1), [v2, v3) => MEDIUM
    // [v1, v2) => HIGH
    double aver_grayscale_value = calculate_average_gray_value(image_color_small, faces_success[0]);
    if ((aver_grayscale_value >= 0 && aver_grayscale_value < threshold_v0) ||
        (aver_grayscale_value >= threshold_v3)) {
        image_input->brightness = 0;
    }
    else if ((aver_grayscale_value >= threshold_v0 && aver_grayscale_value < threshold_v1) ||
        (aver_grayscale_value >= threshold_v2 && aver_grayscale_value < threshold_v3)) {
        image_input->brightness = 1;
    }
    else if ((aver_grayscale_value >= threshold_v1 && aver_grayscale_value < threshold_v2)) {
        image_input->brightness = 2;
    }
    //人脸分辨率
    // 人脸分辨率评估：
    // {low, high}默认值为{80, 120} ,其映射关系为
    // [0, low) => LOW
    // [low, high) => MEDIUM
    // [high, ~) => HIGH
    float face_width = image_input->face_rect[0][2];
    float face_height = image_input->face_rect[0][3];
    float min_widhei = std::min(face_width, face_height);
    if (min_widhei < face_resol_low){
        image_input->resolution = 0;
    }
    else if (min_widhei >= face_resol_low && min_widhei < face_resol_high) {
        image_input->resolution = 1;
    }
    else if (min_widhei >= face_resol_high) {
        image_input->resolution = 2;
    }

    //关键点检测
    //ret = alive_detector_.update(image_color_small, &big_rect, &pts, info, m_min_percent, m_max_percent);
    

}

void CFosaferFaceRecogBackend::rotate_image_90n(cv::Mat &src, cv::Mat &dst, int angle)
{
    if(angle > 180) angle = angle - 360; // kzq
    if (src.data != dst.data) { 
        src.copyTo(dst); 
    }
    angle = ((angle / 90) % 4) * 90;
    if (0 == angle) return;

    //0 : flip vertical; 1 flip horizontal
    bool const flip_horizontal_or_vertical = angle > 0 ? 1 : 0;
    int const number = std::abs(angle / 90);

    for (int i = 0; i != number; ++i) {
        cv::transpose(dst, dst);
        cv::flip(dst, dst, flip_horizontal_or_vertical);
    }
}

cv::Mat CFosaferFaceRecogBackend::ResizeImage(cv::Mat image, int maxDimSize, double *scale_used) {
    cv::Mat image_small;
    double scale = 1.0;
    if (image.size().height > maxDimSize || image.size().width > maxDimSize) {
        if (image.size().height > image.size().width) {
            scale = double(image.size().height) / maxDimSize;
            cv::resize(image, image_small, cv::Size(image.size().width / scale, maxDimSize));
        }
        else {
            scale = double(image.size().width) / maxDimSize;
            cv::resize(image, image_small, cv::Size(maxDimSize, image.size().height / scale));
        }
    } else {
        image_small = image.clone();
    }

    if(scale_used) *scale_used = scale;

    return image_small;
}

DLL_PUBLIC FACERECOG_ENGINE_HANDLE FOSAFER_FaceRECOG_Initialize() {
    CFosaferFaceRecogBackend *instance = new CFosaferFaceRecogBackend();
    instance->init();
    return (FACERECOG_ENGINE_HANDLE)instance;
}

DLL_PUBLIC int FOSAFER_FaceRECOG_Release(FACERECOG_ENGINE_HANDLE pHandle) {
    if (pHandle) {
        try {
            delete (CFosaferFaceRecogBackend*)pHandle;
            pHandle = nullptr;
            return FACE_SUCCESS;
        }
        catch (...) {
            return FACE_ERROR_RELEASE;
        }
        return FACE_SUCCESS;
    }
    else{
        return FACE_ERROR_NULL;
    }
    
}

DLL_PUBLIC int FOSAFER_FaceRECOG_Detect(FACERECOG_ENGINE_HANDLE pHandle, Image* image, int rotateCW) {
    CFosaferFaceRecogBackend* instance = (CFosaferFaceRecogBackend*)pHandle;
    return instance->detect(image, rotateCW);
}