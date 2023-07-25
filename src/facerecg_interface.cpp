#include "facerecg_interface.h"

#include "ncnnssd.h"
#include "face_alignment.h"
#include "fas.h"
#include "occ.h"
#include "fas_structure.h"

#include "detectstrategy.h"
#include "eye_exist.h"
#include "mouth_exist.h"
#include "nose_exist.h"
#include "eyebrow_exist.h"
#include "chin_exist.h"
#include "forehead_exist.h"
#include "detectcontext.h"

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

struct Box {
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
    int detect_deepth(DeepthImage *image_input, float face_rect[4]);
    int detect_nir(NirImage *image_input, float face_rect[4]);
    
    void rotate_image_90n(cv::Mat &src, cv::Mat &dst, int angle);
    cv::Mat ResizeImage(cv::Mat image, int maxDimSize, double *scale_used);

private:
    void detect_brightness(cv::Mat input_img, float& cast, float& da);
    double calculate_average_gray_value(const cv::Mat& image_color, const cv::Rect& rect_image);
    int getbrightness(double average_gray_value);
    double cal_variance(const cv::Mat& image_gray, const ssdFaceRect &face);
    bool PoseEstimation2(const std::vector<cv::Point2f> &pts, 
        float *pose_pitch, 
        float *pose_yaw, 
        float *pose_roll);
    cv::Rect CalculateBox(FaceBox& box, float scale_, int w, int h);
    cv::Rect CalRect(const std::vector<cv::Point> &pts, int begin, int end);
    cv::Rect CalRect(const std::vector<cv::Point> &pts, int begin, int end, int numPoint);
    
    cv::Rect EnlargeRect(const cv::Rect& rect, float width_scale_, float height_scale_, int w, int h);
    bool isMatchEdgeGap(const cv::Rect& face_rect, const cv::Rect& image_rect, const int &edge_gap);
    cv::Rect ReCalRect(const cv::Rect& face_rect, const cv::Rect& image_rect);

private:
    double image_max_dim_ = 600;
    double scale_factor_ = 1;

    ncnnssd *ncnnssd_;
    FOSAFER_alive_detection *alive_detector_;
    //RGBFacefas *rgbalive_;
    FaceFas *facefas_;
    FaceOcc *faceocc_;
    FasStructure * fasstructure_;
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

//人脸百分比
const float min_face_percent = 0.05;   //60cm
const float max_face_percent = 0.15;   //30cm

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
    // if (rgbalive_) {
    //     delete rgbalive_;
    //     rgbalive_ = nullptr;
    // }
    if (facefas_) {
        delete facefas_;
        facefas_ = nullptr;
    }
    if (faceocc_) {
        delete faceocc_;
        faceocc_ = nullptr;
    }
    if (fasstructure_) {
        delete fasstructure_;
        fasstructure_ = nullptr;
    }
}

bool CFosaferFaceRecogBackend::init() {
    ncnnssd_ = new ncnnssd();
    alive_detector_ = new FOSAFER_alive_detection();
    // rgbalive_ = new RGBFacefas();
    facefas_ = new FaceFas();
    faceocc_ = new FaceOcc();
    fasstructure_ = new FasStructure();
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

cv::Rect CFosaferFaceRecogBackend::CalRect(const std::vector<cv::Point> &pts, int begin, int end, int numPoint) {
    int lefttop_x = 1000000000, lefttop_y = 1000000000;
    int rightbottom_x = 0, rightbottom_y = 0; 

    for(int i = begin; i <= end; ++i) {
        lefttop_x = std::min(pts[i].x, lefttop_x);
        lefttop_y = std::min(pts[i].y, lefttop_y);
        rightbottom_x = std::max(pts[i].x, rightbottom_x);
        rightbottom_y = std::max(pts[i].y, rightbottom_y);
    }
    lefttop_x = std::min(pts[numPoint].x, lefttop_x);
    lefttop_y = std::min(pts[numPoint].y, lefttop_y);
    rightbottom_x = std::max(pts[numPoint].x, rightbottom_x);
    rightbottom_y = std::max(pts[numPoint].y, rightbottom_y);

    int width = std::abs(rightbottom_x - lefttop_x);
    int height = std::abs(rightbottom_y - lefttop_y);
    return cv::Rect(lefttop_x, lefttop_y, width, height); 
}

cv::Rect CFosaferFaceRecogBackend::CalRect(const std::vector<cv::Point> &pts, int begin, int end) {
    int lefttop_x = 1000000000, lefttop_y = 1000000000;
    int rightbottom_x = 0, rightbottom_y = 0; 

    for(int i = begin; i <= end; ++i) {
        lefttop_x = std::min(pts[i].x, lefttop_x);
        lefttop_y = std::min(pts[i].y, lefttop_y);
        rightbottom_x = std::max(pts[i].x, rightbottom_x);
        rightbottom_y = std::max(pts[i].y, rightbottom_y);
    }
    int width = std::abs(rightbottom_x - lefttop_x);
    int height = std::abs(rightbottom_y - lefttop_y);
    return cv::Rect(lefttop_x, lefttop_y, width, height); 
}

cv::Rect CFosaferFaceRecogBackend::EnlargeRect(const cv::Rect& rect, float width_scale_, float height_scale_, int w, int h) {
    int x = rect.x;
    int y = rect.y;
    int box_width = rect.width;
    int box_height = rect.height;

    int box_center_x = box_width / 2 + x;
    int box_center_y = box_height / 2 + y;

    int new_width = static_cast<int>(box_width * width_scale_);
    int new_height = static_cast<int>(box_height * height_scale_);

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

bool CFosaferFaceRecogBackend::isMatchEdgeGap(const cv::Rect& face_rect, const cv::Rect& image_rect, const int &edge_gap) {
    int face_lefttop_x = face_rect.x;
    int face_lefttop_y = face_rect.y;
    int face_rightbottom_x = face_rect.x + face_rect.width;
    int face_rightbottom_y = face_rect.y + face_rect.height;

    int image_lefttop_x = image_rect.x;
    int image_lefttop_y = image_rect.y;
    int image_rightbottom_x = image_rect.x + image_rect.width;
    int image_rightbottom_y = image_rect.y + image_rect.height;

    if ((face_lefttop_x - image_lefttop_x > edge_gap && face_lefttop_y - image_lefttop_y > edge_gap) &&
        (image_rightbottom_x - face_rightbottom_x > edge_gap && image_rightbottom_y - face_rightbottom_y > edge_gap)) {
        return true;
    }
    return false;
}

cv::Rect CFosaferFaceRecogBackend::ReCalRect(const cv::Rect& face_rect, const cv::Rect& image_rect) {
    int face_lefttop_x = face_rect.x;
    int face_lefttop_y = face_rect.y;
    int face_rightbottom_x = face_rect.x + face_rect.width;
    int face_rightbottom_y = face_rect.y + face_rect.height;

    int image_lefttop_x = image_rect.x;
    int image_lefttop_y = image_rect.y;
    int image_rightbottom_x = image_rect.x + image_rect.width;
    int image_rightbottom_y = image_rect.y + image_rect.height;

    if (face_lefttop_x < image_lefttop_x) {
        face_lefttop_x = image_lefttop_x;
    }
    if (face_lefttop_y < image_lefttop_y) {
        face_lefttop_y = image_lefttop_y;
    }
    if (face_rightbottom_x > image_rightbottom_x) {
        face_rightbottom_x = image_rightbottom_x;
    }
    if (face_rightbottom_y > image_rightbottom_y) {
        face_rightbottom_y = image_rightbottom_y;
    }
    int face_width = face_rightbottom_x - face_lefttop_x;
    int face_height = face_rightbottom_y - face_lefttop_y;
    
    return cv::Rect(face_lefttop_x, face_lefttop_y, face_width, face_height);
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


double CFosaferFaceRecogBackend::calculate_average_gray_value(const cv::Mat& image_color, const cv::Rect& rect_image) {
    double total_gray_value = 0.0;
    cv::Mat image_gray;
    cv::cvtColor(image_color, image_gray, cv::COLOR_BGR2GRAY);
    
    // std::cout << "image_gray.x : " << image_gray.x << std::endl;
    // std::cout << "image_gray.y : " << image_gray.y << std::endl;
    std::cout << "image_gray.width : " << image_gray.size().width << std::endl;
    std::cout << "image_gray.height : " << image_gray.size().height << std::endl;
    
    
    std::cout << "rect_image.x : " << rect_image.x << std::endl;
    std::cout << "rect_image.y : " << rect_image.y << std::endl;
    std::cout << "rect_image.width : " << rect_image.width << std::endl;
    std::cout << "rect_image.height : " << rect_image.height << std::endl;

    for (int i = rect_image.x; i < rect_image.x + rect_image.width; ++i) {
        for (int j = rect_image.y; j < rect_image.y + rect_image.height; ++j) {
            total_gray_value += static_cast<double>(image_gray.at<uchar>(i, j));
            //std::cout << total_gray_value << std::endl;
        }
        //std::cout << "row : " << i << std::endl;
    }
    
    std::cout << "rect_image.x : " << rect_image.x << std::endl;
    std::cout << "rect_image.y : " << rect_image.y << std::endl;
    std::cout << "rect_image.width : " << rect_image.width << std::endl;
    std::cout << "rect_image.height : " << rect_image.height << std::endl;

    double average_gray_value = total_gray_value / (rect_image.width * rect_image.height);
    return average_gray_value;
}

// {v0, v1, v2, v3}的默认值为{70, 100, 210, 230}
// 亮度区间：
// [0, v0), [v3, ~) => LOW
// [v0, v1), [v2, v3) => MEDIUM
// [v1, v2) => HIGH
int CFosaferFaceRecogBackend::getbrightness(double average_gray_value) {
    int brightness = 2;
    if ((average_gray_value >= 0 && average_gray_value < threshold_v0) ||
        (average_gray_value >= threshold_v3)) {
        brightness = 0;
    }
    else if ((average_gray_value >= threshold_v0 && average_gray_value < threshold_v1) ||
        (average_gray_value >= threshold_v2 && average_gray_value < threshold_v3)) {
        brightness = 1;
    }
    else if ((average_gray_value >= threshold_v1 && average_gray_value < threshold_v2)) {
        brightness = 2; 
    }
    return brightness;
}
    
double CFosaferFaceRecogBackend::cal_variance(const cv::Mat& image_color, const ssdFaceRect &face) {
    // 图像均值 和 标准方差
    cv::Mat meanMat, stdMat;
    cv::meanStdDev(image_color, meanMat, stdMat);
 
    return (stdMat.at<double>(0) + stdMat.at<double>(1) + stdMat.at<double>(2)) / 3;
}

int CFosaferFaceRecogBackend::detect(Image* image_input, int rotateCW) {
    if (!image_input) { 
        return FACE_ERROR_NULL; 
    }
    cv::Mat image_color;
    cv::Mat image_color_small;
    cv::Rect rect_face;
    cv::Rect rect_image(0, 0, image_input->width, image_input->height);

    cv::Mat image_buf2(image_input->height, image_input->width, CV_8UC3, image_input->data);
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
    if(faces.size() > 1) {
         image_input->face_num = 0;
         return FACE_ERROR_MULTIFACE;
    }
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

    rect_face.x = image_input->face_rect[0][0];
    rect_face.y = image_input->face_rect[0][1];
    rect_face.width = image_input->face_rect[0][2];
    rect_face.height = image_input->face_rect[0][3];
    
    //如果人脸框超出边界,则截掉
    rect_face = ReCalRect(rect_face, rect_image);

    //计算均方差，均方差小于阈值，则判定为纸张
    double aver_variance = cal_variance(image_color_small, faces_success[0]);
    if(aver_variance < face_paper_thresh){
        //return FACE_LOW_QUALITY;
    }
    // {v0, v1, v2, v3}的默认值为{70, 100, 210, 230}
    // 亮度区间：
    // [0, v0), [v3, ~) => LOW
    // [v0, v1), [v2, v3) => MEDIUM
    // [v1, v2) => HIGH

/*
    double aver_grayscale_value = calculate_average_gray_value(image_color, rect_image);
    image_input->image_average_gray_value = aver_grayscale_value;
    image_input->image_brightness = getbrightness(aver_grayscale_value);
    
    aver_grayscale_value = calculate_average_gray_value(image_color, rect_face);
    image_input->face_average_gray_value = aver_grayscale_value;
    image_input->face_brightness = getbrightness(aver_grayscale_value);    
*/    
    
    /*
    //人脸分辨率
    // 人脸分辨率评估：
    // {low, high}默认值为{80, 120} ,其映射关系为
    // [0, low) => LOW
    // [low, high) => MEDIUM
    // [high, ~) => HIGH
    float face_width = rect_face.width;
    float face_height = rect_face.height;

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
    */
    //计算人脸面积百分比
    float face_area = rect_face.width * rect_face.height; 
    float image_area = image_color.cols * image_color.rows;
    float face_percent = face_area / image_area;
    image_input->face_percent = face_percent;

    //边缘检测，人脸框距离边缘需大于50像素
    const int edge_gap = 50;
    image_input->isNearEdge = isMatchEdgeGap(rect_face, rect_image, edge_gap);
    
    //关键点检测
    std::vector<cv::Point2f> pts;
    ret = alive_detector_->update(image_color_small, image_small_face_rect, &pts);
    if (ret != 0) {
        return FACE_DET_FAILED;
    }

    int count = 0;
    std::vector<cv::Point> ori_points;
    for(auto pt : pts) {
        cv::Point point;
        point.x = static_cast<int>(pt.x * scale_factor_);
        point.y = static_cast<int>(pt.y * scale_factor_);

        ori_points.push_back(point);

        // std::string label = std::to_string(count);
        // cv::Point textPos(point.x, point.y);
        // int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        // double fontScale = 0.5;
        // cv::Scalar textColor(0, 0, 255);  // 以BGR格式指定颜色，这里为红色
        // int fontThickness = 1;

        //cv::putText(image_color, label, textPos, fontFace, fontScale, textColor, fontThickness);
        count++;
    }
    //计算两眼之间的距离
    double dist = DistanceTo(ori_points[80], ori_points[81]);
    image_input->eye_dist = dist;
    
    // if(dist >= 60 && dist < 90) {
    //     image_input->eye_dist = 0;
    // }
    // else if(dist >= 90) {
    //     image_input->eye_dist = 1;
    // }
    
    float pitch, yaw, roll;        
    //bool PoseEstimation2(std::vector<cv::Point> &pts, float *pose_pitch, float *pose_yaw, float *pose_roll) {
    ret = PoseEstimation2(pts, &pitch, &yaw, &roll);
    image_input->pitch = pitch;
    image_input->yaw = yaw;
    image_input->roll = roll;
    
    // 活体检测
    float fasok;
    cv::Mat fas_img;
    image_color.copyTo(fas_img);
    facefas_->detect(fas_img.data, fas_img.size().height, fas_img.size().width, &fasok);
    image_input->alive_score = fasok;
    

    //检测有无遮挡，眉毛，眼睛，鼻子，嘴巴，下巴，额头，耳朵
    //计算左眉毛的区域(8~15)
    const float width_scale_eyebrow = 1;
    const float height_scale_eyebrow = 1.5;
    cv::Rect rect_left_eyebrow = CalRect(ori_points, 8, 15);
    rect_left_eyebrow = EnlargeRect(rect_left_eyebrow, width_scale_eyebrow, height_scale_eyebrow, image_color.cols, image_color.rows);
    cv::Mat image_left_eyebrow = image_color(rect_left_eyebrow);
    cv::imwrite("../data/images/left_eyebrow.jpg", image_left_eyebrow);
  
    //计算右眉毛的区域(0~7)
    cv::Rect rect_right_eyebrow = CalRect(ori_points, 0, 7);
    rect_right_eyebrow = EnlargeRect(rect_right_eyebrow, width_scale_eyebrow, height_scale_eyebrow, image_color.cols, image_color.rows);
    cv::Mat image_right_eyebrow = image_color(rect_right_eyebrow);
    cv::imwrite("../data/images/right_eyebrow.jpg", image_right_eyebrow);

    //计算左眼的区域(24~31) 
    // std::cout << "left eye begin" << std::endl;
    int w_eye = (int)((pts[81].x - pts[80].x) * 0.4);
    int left_eye_x = (int)(pts[81].x - w_eye);
    int left_eye_y = (int)(pts[81].y - 0.8*w_eye);
    int left_eye_w = (int)(w_eye + w_eye);
    int left_eye_h = (int)(1.6 * w_eye);
    if ( left_eye_x <= 0) 
    {
        left_eye_x = 1;
    }
    if (left_eye_y <= 0)
    {
        left_eye_y = 1;
    }
    if (left_eye_x + left_eye_w + 20 >= image_color_small.size().width)
    {
        left_eye_w = image_color_small.size().width - left_eye_x - 20;
    }
    if (left_eye_y + left_eye_h + 20 >= image_color_small.size().height)
    {
        left_eye_h = image_color_small.size().height - left_eye_y - 20;
    }
    cv::Rect rect_left_eye(left_eye_x, left_eye_y, left_eye_w, left_eye_h);
    cv::Mat image_left_eye = image_color_small(rect_left_eye);

    // const float height_scale_eye = 1.8;
    // const float width_scale_eye = 1.5;
    // cv::Rect rect_left_eye = CalRect(ori_points, 24, 31);
    // rect_left_eye = EnlargeRect(rect_left_eye, width_scale_eye, height_scale_eye, image_color.cols, image_color.rows);
    // cv::Mat image_left_eye = image_color(rect_left_eye);
    cv::imwrite("../data/images/left_eye.jpg", image_left_eye);
    
    //计算右眼的区域(16~23)
    // std::cout << "right eye begin" << std::endl;
    int right_eye_x = (int)(pts[80].x - w_eye);
    int right_eye_y = (int)(pts[80].y - 0.8 * w_eye);
    int right_eye_w = (int)(w_eye+w_eye);
    int right_eye_h = (int)(1.6*w_eye);
    if ( right_eye_x <= 0)
    {
        right_eye_x = 1;
    }
    if (right_eye_y <= 0)
    {
        right_eye_y = 1;
    }
    if (right_eye_x + right_eye_w + 20 >= image_color_small.size().width)
    {
        right_eye_w = image_color_small.size().width - right_eye_x - 20;
    }
    if (right_eye_y + right_eye_h + 20 >= image_color_small.size().height)
    {
        right_eye_h = image_color_small.size().height - right_eye_y - 20;
    }
    cv::Rect rect_right_eye(right_eye_x, right_eye_y, right_eye_w, right_eye_h);
    cv::Mat image_right_eye = image_color_small(rect_right_eye);

    // cv::Rect rect_right_eye = CalRect(ori_points, 16, 23);
    // rect_right_eye = EnlargeRect(rect_right_eye, width_scale_eye, height_scale_eye, image_color.cols, image_color.rows);
    // cv::Mat image_right_eye = image_color(rect_right_eye);
    cv::imwrite("../data/images/right_eye.jpg", image_right_eye);
    
    //计算鼻子的区域(32~42)
    const float height_scale_nose = 1.2;
    const float width_scale_nose = 1.2;
    cv::Rect rect_nose = CalRect(ori_points, 32, 42);
    rect_nose = EnlargeRect(rect_nose, width_scale_nose, height_scale_nose, image_color.cols, image_color.rows);
    cv::Mat image_nose = image_color(rect_nose);
    cv::imwrite("../data/images/nose.jpg", image_nose);

    //计算嘴巴的区域(43~60)
    // std::cout << "mouth begin" << std::endl;
    int w = (int)((pts[48].x - pts[44].x) * 0.3); // 嘴宽
    int mouth_x = (int)(pts[44].x - w);
    int mouth_y = (int)(pts[44].y - 1.6 * w);
    int mouth_w = (int)(pts[48].x - pts[44].x + w + w);
    int mouth_h = 3.2 * w;
    if ( mouth_x <= 0) {
        mouth_x = 1;
    }
    if (mouth_y <= 0) {
        mouth_y = 1;
    }
    if (mouth_x + mouth_w + 20 >= image_color_small.size().width) {
        mouth_w = image_color_small.size().width - mouth_x - 20;
    }
    if (mouth_y + mouth_h + 20 >= image_color_small.size().height) {
        mouth_h = image_color_small.size().height - mouth_y - 20;
    }
    cv::Rect rect_mouth(mouth_x, mouth_y, mouth_w, mouth_h);
    cv::Mat image_mouth = image_color_small(rect_mouth);

    // const float height_scale_mouth = 2.5;
    // const float width_scale_mouth = 2;
    // cv::Rect rect_mouth = CalRect(ori_points, 43, 60);
    // rect_mouth = EnlargeRect(rect_mouth, width_scale_mouth, height_scale_mouth, image_color.cols, image_color.rows);
    // cv::Mat image_mouth = image_color(rect_mouth);    
    cv::imwrite("../data/images/mouth.jpg", image_mouth);

    //计算下巴的区域,嘴巴往下
    const float width_scale_chin = 3;
    const float height_scale_chin = 4;
    cv::Rect rect_chin = CalRect(ori_points, 43, 60);
    //左上角确定
    rect_chin.y = rect_chin.y + rect_chin.height * height_scale_chin / 2;
    rect_chin = EnlargeRect(rect_chin, width_scale_chin, height_scale_chin, image_color.cols, image_color.rows);
    cv::Mat image_chin = image_color(rect_chin);
    cv::imwrite("../data/images/chin.jpg", image_chin);

    //计算额头的区域,眉毛往上
    const float width_scale_forehead = 1.2;
    const float height_scale_forehead = 5;
    cv::Rect rect_forehead = CalRect(ori_points, 0, 15);
    rect_forehead.y = rect_forehead.y - rect_forehead.height * height_scale_forehead / 2;
    rect_forehead = EnlargeRect(rect_forehead, width_scale_forehead, height_scale_forehead, image_color.cols, image_color.rows);
    cv::Mat image_forehead = image_color(rect_forehead);
    cv::imwrite("../data/images/forehead.jpg", image_forehead);

    //计算左耳的区域(77~79)
    const float width_scale_ear = 5;
    const float height_scale_ear = 2;
    cv::Rect rect_left_ear = CalRect(ori_points, 77, 79, 8);
    rect_left_ear.x = rect_left_ear.x + rect_left_ear.width * width_scale_ear / 2.0;
    rect_left_ear = EnlargeRect(rect_left_ear, width_scale_ear, height_scale_ear, image_color.cols, image_color.rows);
    cv::Mat image_left_ear = image_color(rect_left_ear);
    cv::imwrite("../data/images/left_ear.jpg", image_left_ear);
    
    //计算右耳的区域(61~63)
    cv::Rect rect_right_ear = CalRect(ori_points, 61, 63, 0);
    rect_right_ear.x = rect_right_ear.x - rect_right_ear.width * width_scale_ear / 2.0;
    rect_right_ear = EnlargeRect(rect_right_ear, width_scale_ear, height_scale_ear, image_color.cols, image_color.rows);
    cv::Mat image_right_ear = image_color(rect_right_ear);
    cv::imwrite("../data/images/right_ear.jpg", image_right_ear);

    //分别检测有无遮挡
    std::shared_ptr<EyeExist> eye_detect = std::make_shared<EyeExist>();
    std::shared_ptr<MouthExist> mouth_detect = std::make_shared<MouthExist>();
    std::shared_ptr<NoseExist> nose_detect = std::make_shared<NoseExist>();
    std::shared_ptr<EyebrowExist> eyebrow_detect = std::make_shared<EyebrowExist>();
    std::shared_ptr<ChinExist> chin_detect = std::make_shared<ChinExist>();
    std::shared_ptr<ForeheadExist> forehead_detect = std::make_shared<ForeheadExist>();

    //左眼
    std::shared_ptr<DetectContext> context = std::make_shared<DetectContext>(eye_detect.get());
    int detect_lefteye = context->detect(image_left_eye.data, image_left_eye.cols, image_left_eye.rows);
    image_input->detect_lefteye = detect_lefteye;
    //右眼
    int detect_righteye = context->detect(image_right_eye.data, image_right_eye.cols, image_right_eye.rows);
    image_input->detect_righteye = detect_righteye;
    //嘴巴
    context->setStrategy(mouth_detect.get());
    int detect_mouth = context->detect(image_mouth.data, image_mouth.cols, image_mouth.rows);
    image_input->detect_mouth = detect_mouth;

    //鼻子
    context->setStrategy(nose_detect.get());
    int detect_nose = context->detect(image_nose.data, image_nose.cols, image_nose.rows);
    image_input->detect_nose = detect_nose;

    //左眉毛
    context->setStrategy(eyebrow_detect.get());
    int detect_lefteyebrow = context->detect(image_left_eyebrow.data, image_left_eyebrow.cols, image_left_eyebrow.rows);
    image_input->detect_lefteyebrow = detect_lefteyebrow;

    //右眉毛
    context->setStrategy(eyebrow_detect.get());
    int detect_righteyebrow = context->detect(image_right_eyebrow.data, image_right_eyebrow.cols, image_right_eyebrow.rows);
    image_input->detect_righteyebrow = detect_righteyebrow;

    //下巴
    context->setStrategy(chin_detect.get());
    int detect_chin = context->detect(image_chin.data, image_chin.cols, image_chin.rows);
    image_input->detect_chin = detect_chin;

    //脸部轮廓
    float occok;
    cv::Mat subimage_occ;
    cv::resize(image_color_small(cv::Rect(faces[0].x, faces[0].y, faces[0].w, faces[0].h)), subimage_occ, cv::Size(64, 64));
    faceocc_->detect(subimage_occ.data, subimage_occ.size().height, subimage_occ.size().width, &occok);
    std::cout << "occ detect : " <<occok << std::endl;
    image_input->detect_occ = 0;
    
    //额头
    context->setStrategy(forehead_detect.get());
    int detect_forehead = context->detect(image_forehead.data, image_forehead.cols, image_forehead.rows);
    image_input->detect_forehead = detect_forehead;

    count = 0;
    for(auto point : ori_points) {
        std::string label = std::to_string(count);
        cv::Point textPos(point.x, point.y);
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        cv::Scalar textColor(0, 0, 255);  // 以BGR格式指定颜色，这里为红色
        int fontThickness = 1;

        cv::putText(image_color, label, textPos, fontFace, fontScale, textColor, fontThickness);
        count++;
    }

    image_input->face_rect[1][0] = rect_mouth.x * scale_factor_;
    image_input->face_rect[1][1] = rect_mouth.y * scale_factor_;
    image_input->face_rect[1][2] = rect_mouth.width * scale_factor_;
    image_input->face_rect[1][3] = rect_mouth.height * scale_factor_;
     
    //cv::Rect rect_mouth_real(rect_mouth.x * scale_factor_, rect_mouth.y * scale_factor_, rect_mouth.width * scale_factor_, rect_mouth.height * scale_factor_);
    //cv::rectangle(image_color, rect_mouth_real, cv::Scalar(0, 0, 255), 2);

    // cv::rectangle(image_color, rect_face, cv::Scalar(0, 255, 0), 2);

    cv::imwrite("../data/images/result.jpg", image_color);
	// cv::imshow("result", image_color);
	// cv::waitKey(0);
	
    return 0;
}

int CFosaferFaceRecogBackend::detect_nir(NirImage *image_input, float face_rect[4]) {
    return 0;
}
    

int CFosaferFaceRecogBackend::detect_deepth(DeepthImage *image_input, float face_rect[4]) {
    int ret = 0;
    cv::Mat image_color;
    cv::Rect rect_face(face_rect[0], face_rect[1], face_rect[2], face_rect[3]);
    cv::Rect rect_image(0, 0, image_input->width, image_input->height);

    cv::Mat image_buf(image_input->height, image_input->width, CV_8UC3, image_input->data);
    image_color.create(image_buf.size().height, image_buf.size().width, image_buf.channels() == 3 ? CV_8UC3 : CV_8UC1);
    memcpy(image_color.data, image_buf.data, image_buf.size().height* image_buf.size().width * image_buf.channels());
    
    cv::Mat image_face = image_color(rect_face);

    float score;
    ret = fasstructure_->detect(image_face.data, image_face.cols, image_face.rows, &score);
    image_input->alive_score = score;
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

DLL_PUBLIC int FOSAFER_FaceRECOG_RGBDetect(FACERECOG_ENGINE_HANDLE pHandle, Image* image, int rotateCW) {
    CFosaferFaceRecogBackend* instance = (CFosaferFaceRecogBackend*)pHandle;
    return instance->detect(image, rotateCW);
}

DLL_PUBLIC int FOSAFER_FaceRECOG_DeepthDetect(FACERECOG_ENGINE_HANDLE pHandle, DeepthImage* image, float face_rect[4]) {
    CFosaferFaceRecogBackend* instance = (CFosaferFaceRecogBackend*)pHandle;
    return instance->detect_deepth(image, face_rect);
}

DLL_PUBLIC int FOSAFER_FaceRECOG_NirDetect(FACERECOG_ENGINE_HANDLE pHandle, DeepthImage* image, float face_rect[4]) {
    CFosaferFaceRecogBackend* instance = (CFosaferFaceRecogBackend*)pHandle;
    return instance->detect_nir(image, face_rect);
}
