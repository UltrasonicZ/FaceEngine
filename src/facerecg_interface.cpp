#include "facerecg_interface.h"

#include "ncnnssd.h"
#include "facealign.h"
#include "rgbalive.h"

#include <algorithm>
#include <vector>

#include "utils.h"

struct FaceBox {
    float alive_score;
    float x1;
    float y1;
    float x2;
    float y2;
};

class CFosaferFaceRecogBackend {
public:
    CFosaferFaceRecogBackend();
    ~CFosaferFaceRecogBackend();
    
    bool init();
    int detect(Image* image_input, int rotateCW);
    
    void rotate_image_90n(cv::Mat &src, cv::Mat &dst, int angle);
    cv::Mat ResizeImage(cv::Mat image, int maxDimSize, double *scale_used);

private:
    void detect_brightness(cv::Mat input_img, float& cast, float& da);
    double calculate_average_gray_value(const cv::Mat& image_gray, const ssdFaceRect &face);
    double cal_variance(const cv::Mat& image_gray, const ssdFaceRect &face);
    bool PoseEstimation2(const std::vector<cv::Point2f> &pts, 
        float *pose_pitch, 
        float *pose_yaw, 
        float *pose_roll);
    cv::Rect CalculateBox(FaceBox& box, float scale_, int w, int h);


private:
    double image_max_dim_ = 600;
    double scale_factor_ = 1;

    ncnnssd *ncnnssd_;
    FOSAFER_alive_detection *alive_detector_;
    RGBFacefas *rgbalive_;
};


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

CFosaferFaceRecogBackend::CFosaferFaceRecogBackend() {
    
}

CFosaferFaceRecogBackend::~CFosaferFaceRecogBackend(){
    if (ncnnssd_) {
        delete ncnnssd_;
        ncnnssd_ = nullptr;
    }
    if (alive_detector_) {
        delete alive_detector_;
        alive_detector_ = nullptr;
    }
    if (rgbalive_) {
        delete rgbalive_;
        rgbalive_ = nullptr;
    }
}

bool CFosaferFaceRecogBackend::init() {
    ncnnssd_ = new ncnnssd();
    alive_detector_ = new FOSAFER_alive_detection();
    rgbalive_ = new RGBFacefas();
    return true;
}

bool compareVector(const ssdFaceRect &a, const ssdFaceRect &b){
    return a.w * a.h > b.w * b.h;
}

bool CFosaferFaceRecogBackend::PoseEstimation2(const std::vector<cv::Point2f> &pts, float *pose_pitch, float *pose_yaw, float *pose_roll) {
    if(pts.empty()) return false;
    float head_pose_array[15][3] = {
        {0.139791, 27.4028, 7.02636},
        {-2.48207, 9.59384, 6.03758},
        {1.27402, 10.4795, 6.20801},
        {1.17406, 29.1886, 1.67768},
        {0.306761, -103.832, 5.66238},
        {4.78663, 17.8726, -15.3623},
        {-5.20016, 9.29488, -11.2495},
        {-25.1704, 10.8649, -29.4877},
        {-5.62572, 9.0871, -12.0982},
        {-5.19707, -8.25251, 13.3965},
        {-23.6643, -13.1348, 29.4322},
        {67.239, 0.666896, 1.84304},
        {-2.83223, 4.56333, -15.885},
        {-4.74948, -3.79454, 12.7986},
        {-16.1, 1.47175, 4.03941}
    };
    float tmp[15]  = {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,1.0f};
    float ret[3] = {0,0,0};

    int samplePdim = 7;
    float miny = 10000000000.0f;
    float maxy = 0.0f;
    float sumx = 0.0f;
    float sumy = 0.0f;
    int index_list[] = { 16, 20, 28, 24, 32, 43, 49 };

    float minx = 10000000000.0f;
    float maxx = 0.0f;
    //找最大最小y
    for (int idx = 0; idx < samplePdim; idx++) {
        float y = pts[index_list[idx]].y;
        float x = pts[index_list[idx]].x;
        sumx += x;
        sumy += y;
        if (miny > y)
            miny = y;
        if (maxy < y)
            maxy = y;

        if (minx > x)
            minx = x;
        if (maxx < x)
            maxx = x;
    }
    float distx = maxx - minx;
    float disty = maxy - miny;
    //人脸中心点
    sumx = sumx / samplePdim;
    sumy = sumy / samplePdim;
    
    for (int i = 0; i < samplePdim; i++) {
        tmp[i] = (pts[index_list[i]].x - sumx) / distx;
        tmp[i+samplePdim] = (pts[index_list[i]].y - sumy) / disty;
    }

    for(int j = 0; j < 3; j++) {
        float s = 0;
        for(int k = 0; k < 15; k++) {
            s = s + tmp[k] * head_pose_array[k][j];
        }
        ret[j] = s;
    }

    *pose_pitch = ret[0];
    *pose_yaw = ret[1];
    *pose_roll = ret[2];
    return true;
}

cv::Rect CFosaferFaceRecogBackend::CalculateBox(FaceBox& box, float scale_, int w, int h) {
    int x = static_cast<int>(box.x1);
    int y = static_cast<int>(box.y1);
    int box_width = static_cast<int>(box.x2 - box.x1 + 1);
    int box_height = static_cast<int>(box.y2 - box.y1 + 1);

    float scale = std::min(scale_, std::min((w - 1) / (float)box_width, (h - 1) / (float)box_height));
    std::cout << "real scale:" << scale << std::endl;
    int box_center_x = box_width / 2 + x;
    int box_center_y = box_height / 2 + y;

    int new_width = static_cast<int>(box_width * scale);
    int new_height = static_cast<int>(box_height * scale);

    int left_top_x = box_center_x - new_width / 2;
    int left_top_y = box_center_y - new_height / 2;
    int right_bottom_x = box_center_x + new_width / 2;
    int right_bottom_y = box_center_y + new_height / 2;

    if (left_top_x < 0) {
        right_bottom_x -= left_top_x;
        left_top_x = 0;
    }

    if (left_top_y < 0) {
        right_bottom_y -= left_top_y;
        left_top_y = 0;
    }

    if (right_bottom_x >= w) {
        int s = right_bottom_x - w + 1;
        left_top_x -= s;
        right_bottom_x -= s;
    }

    if (right_bottom_y >= h) {
        int s = right_bottom_y - h + 1;
        left_top_y -= s;
        right_bottom_y -= s;
    }

    return cv::Rect(left_top_x, left_top_y, new_width, new_height);
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
    
    cv::Rect image_small_face_rect;
    image_small_face_rect.x = faces_success.at(0).x;
    image_small_face_rect.y = faces_success.at(0).y;
    image_small_face_rect.width = faces_success.at(0).w;
    image_small_face_rect.height = faces_success.at(0).h;

    cv::Rect large_face_rect;
    large_face_rect.x = image_input->face_rect[0][0];
    large_face_rect.y = image_input->face_rect[0][1];
    large_face_rect.width = image_input->face_rect[0][2];
    large_face_rect.height = image_input->face_rect[0][3];

    cv::rectangle(image_color, large_face_rect, cv::Scalar(0, 255, 0), 2);
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
    Info *info;
    std::vector<cv::Point2f> pts;
    float m_min_percent = 0.0;
    float m_max_percent = 100.0;
    ret = alive_detector_->update(image_color_small, image_small_face_rect, &pts, info, m_min_percent, m_max_percent);

    int count = 0;
    std::vector<cv::Point2f> ori_points;
    for(auto point : pts) {
        cv::Point2f ori_point;
        ori_point.x = point.x * scale_factor_;
        ori_point.y = point.y * scale_factor_;

        ori_points.push_back(ori_point);

        std::string label = std::to_string(count);
        cv::Point2f textPos(ori_point.x, ori_point.y);
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        cv::Scalar textColor(0, 0, 255);  // 以BGR格式指定颜色，这里为红色
        int fontThickness = 1;

        cv::putText(image_color, label, textPos, fontFace, fontScale, textColor, fontThickness);
        count++;
    }

    //计算两眼之间的距离
    double dist = DistanceTo(ori_points[80], ori_points[81]);
    if(dist >= 60 && dist < 90) {
        image_input->eye_dist = 0;
    }
    else if(dist >= 90) {
        image_input->eye_dist = 1;
    }
    std::cout << "eye distance:" << dist << std::endl;

    float pitch, yaw, roll;        
    //bool PoseEstimation2(std::vector<cv::Point> &pts, float *pose_pitch, float *pose_yaw, float *pose_roll) {
    ret = PoseEstimation2(pts, &pitch, &yaw, &roll);

    std::cout << "pitch:" << pitch << std::endl;
    std::cout << "yaw:" << yaw << std::endl;
    std::cout << "roll:" << roll << std::endl;
    
    // 活体检测
    FaceBox box;
    box.x1 = image_input->face_rect[0][0];
    box.y1 = image_input->face_rect[0][1];
    box.x2 = image_input->face_rect[0][0] + image_input->face_rect[0][2];
    box.y2 = image_input->face_rect[0][1] + image_input->face_rect[0][3];

    float scale1 = 2.7;
    float scale2 = 4.0;
    
    std::cout << "alive1" << std::endl;
    cv::Rect rect27 = CalculateBox(box, scale1, image_color.cols, image_color.rows);
    std::cout << rect27.x << std::endl;
    std::cout << rect27.y << std::endl;
    std::cout << rect27.width << std::endl;
    std::cout << rect27.height << std::endl;
    
    cv::Rect rect40 = CalculateBox(box, scale2, image_color.cols, image_color.rows);
    std::cout << rect40.x << std::endl;
    std::cout << rect40.y << std::endl;
    std::cout << rect40.width << std::endl;
    std::cout << rect40.height << std::endl;
    
    std::cout << "alive2" << std::endl;
    
    cv::Mat roi27 = image_color(rect27);
    cv::Mat roi40 = image_color(rect40);
    cv::imwrite("result_roi27.jpg", roi27);
    cv::imwrite("result_roi40.jpg", roi40);
    
    std::cout << "alive3" << std::endl;
    
    // image_input->alive_score = rgbalive->detect(roi27.data, 80, 80, roi40.data, 80, 80);
    image_input->alive_score = rgbalive_->detect(roi27.data, roi27.cols, roi27.rows, roi40.data, roi40.cols, roi40.rows);

    std::cout << "alive_score:" << image_input->alive_score << std::endl;

    cv::imwrite("result.jpg", image_color);
    cv::imshow("result", image_color);
	cv::waitKey(0);
    return 0;
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