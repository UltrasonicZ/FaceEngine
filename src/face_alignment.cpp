#include "face_alignment.h"

#include "landmarks_ljw.id.h"
#include "landmarks_ljw.mem.h"
#include "tool.h"

#ifndef _MSC_VER
#include <sys/time.h>
#endif

// #if !USE_NCNN
// extern char *model_fosafer_tracking_prototxt;
// extern const unsigned int model_fosafer_tracking_prototxt_len;
// extern char *model_fosafer_tracking_caffemodel;
// extern const unsigned int model_fosafer_tracking_caffemodel_len;
// #endif

int clv_res = 0;

inline cv::Rect getrect(FaceRect const &rect) {
    return cv::Rect(rect.x, rect.y, rect.w, rect.h);
}


static float check_mid(vector<cv::Point2f> const & pts) {
    cv::Point2f const & mid = pts[32];
    cv::Point2f const & left = pts[63];
    cv::Point2f const & right = pts[77];

    float ldis = abs(mid.x - left.x);
    float rdis = abs(right.x - mid.x);

    return abs(rdis - ldis) / max(ldis, rdis);
}

static float check_tilt(vector<cv::Point2f> const & pts) {
    cv::Point2f const & up = pts[33];
    cv::Point2f const & down = pts[38];

    float distance = sqrt(square(up.x - down.x) + square(up.y - down.y));
    float projection = down.y - up.y;

    if(distance < 1e-20f) return 90.0f;

    return acos(projection / distance) / (1.5707963267948966) * 90;
}

CNCNNFeatureExtractor::CNCNNFeatureExtractor() {
    net_ = new ncnn::Net();
    net_->load_param(landmarks_ljw_params_bin);
    net_->load_model(landmarks_ljw_bin);
}

CNCNNFeatureExtractor::~CNCNNFeatureExtractor() {
    if(net_) {
     	delete net_;
        net_ = nullptr;
	}
}

float CNCNNFeatureExtractor::ExtractFeature(cv::Mat const & image, std::vector<float> &ret) {
	ncnn::Extractor extractor = net_->create_extractor();
    std::vector<float> tmp;

	ret.clear();
    if(image.channels() != 3) { return -1;}
    
	int w = image.cols; 
    int h = image.rows; 
	
	float mean[3] = { 0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f }; 
    float stds[3] = { 1 / 255.0f, 1 / 255.0f, 1 / 255.0f }; 
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 160, 160); 
    in.substract_mean_normalize(mean, stds); 

    ncnn::Mat prob, landmarks; 
    #ifdef __LINUX__
    extractor.set_num_threads(1);
    #endif
    extractor.set_light_mode(true);
    extractor.input(landmarks_ljw_params_id::BLOB_input_1, in); 
    extractor.extract(landmarks_ljw_params_id::BLOB_579, prob);
    extractor.extract(landmarks_ljw_params_id::BLOB_565, landmarks);   
    
    int size = landmarks.h * landmarks.w;
    //FLOG("97point prob: %d, %d, %d\n", prob.c, prob.h, prob.w); 
    //FLOG("97point mark: %d, %d, %d\n", landmarks.c, landmarks.h, landmarks.w); 

    for(int c = 0;c < landmarks.c;c++) {
        float *ptr = (float *)landmarks.data + landmarks.cstep * c;
        for(int z = 0;z < size;z++) {
            float value =0;
            if( z%2==0 ) value = ptr[z] * w ;
            if( z%2==1 ) value = ptr[z] * h ;
            //if(value < 0   ) value = 0;
            //if(value >=160 ) value = 159; 
            tmp.push_back(value);
        }
    }

    transformPts(tmp, ret);
    //FLOG("97point size：%d", tmp.size());
    //FLOG("82point size：%d", ret.size());
    //FLOG("97point prob: %f", ((float *)prob.data)[1]);

    return ((float *)prob.data)[1];

    // if(g_ncnndebug)
    // {
    //     cv::Mat debug;
    //     debug.create(image.size().height, image.size().width, image.channels() == 3 ? CV_8UC3 : CV_8UC1);
    //     memcpy(debug.data, image.data, image.size().height* image.size().width * image.channels());
    //     for(int j=0; j<ret.size()/2; j++)
    //         cv::circle(debug, cv::Point(ret[2*j+0], ret[2*j+1]), 1, cv::Scalar(0,255,0),1,8,0);
    //     cv::imwrite("/mnt/sdcard/fosafer/landmark_3.jpg",debug);
    //     g_ncnndebug=0;
    // }
}

void CNCNNFeatureExtractor::transformPts(std::vector<float> &ret2, std::vector<float> &ret)
{
    if(ret2.size() == 194)
    {
        //FLOG("97point alldata: ------------------------------------");
        //for(int i=0; i<97; i++)
        //{
        //    FLOG("97point alldata: %d %f %f", i, ret2[2*i+0], ret2[2*i+1]);
        //}
        //FLOG("97point alldata: ------------------------------------");

        int i = 0;
        ret.push_back(ret2[2*16+0]);    ret.push_back(ret2[2*16+1]);     // 00-00
        for(i=23; i>=17; i-- ) 
        {
            //if(i>18)FLOG("97point data: %f %f",ret2[2*i+0],ret2[2*i+1]);
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 01-07
        }
        ret.push_back(ret2[2*24+0]);    ret.push_back(ret2[2*24+1]);     // 08-08
        for(i=31; i>=25; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 09-15
        }
        ret.push_back(ret2[2*32+0]);    ret.push_back(ret2[2*32+1]);     // 16-16
        for(i=39; i>=33; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 17-23
        }
        ret.push_back(ret2[2*40+0]);    ret.push_back(ret2[2*40+1]);     // 24-24
        for(i=47; i>=41; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 25-31
        }

        ret.push_back(ret2[2*63+0]);    ret.push_back(ret2[2*63+1]);     // 32-32
        ret.push_back(ret2[2*60+0]);    ret.push_back(ret2[2*60+1]);     // 33-33
        ret.push_back(ret2[2*50+0]);    ret.push_back(ret2[2*50+1]);     // 34-34

        for(i=52; i<=54; i++ ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 35-37
        }
        ret.push_back(ret2[2*64+0]);    ret.push_back(ret2[2*64+1]);     // 38-38
        for(i=59; i>=57; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);     // 39-41
        }
        ret.push_back(ret2[2*55+0]);    ret.push_back(ret2[2*55+1]);     // 42-42

        ret.push_back(ret2[2*96+0]);    ret.push_back(ret2[2*96+1]);     // 43-43
        ret.push_back(ret2[2*81+0]);    ret.push_back(ret2[2*81+1]);     // 44-44
        ret.push_back(ret2[2*83+0]);    ret.push_back(ret2[2*83+1]);     // 45-45
        ret.push_back(ret2[2*85+0]);    ret.push_back(ret2[2*85+1]);     // 46-46
        ret.push_back(ret2[2*86+0]);    ret.push_back(ret2[2*86+1]);     // 47-47
        ret.push_back(ret2[2*88+0]);    ret.push_back(ret2[2*88+1]);     // 48-48
        ret.push_back(ret2[2*89+0]);    ret.push_back(ret2[2*89+1]);     // 49-49

        for(i=73; i>=65; i-- ) 
        {
            if(i%2==1) { ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);} // 50-54
        }
        for(i=95; i>=91; i-- ) 
        {
            if(i%2==1) { ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);} // 55-57
        }
        for(i=75; i<=79; i++ ) 
        {
            if(i%2==1) { ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);} // 58-60
        }

        ret.push_back(ret2[2*7+0]);    ret.push_back(ret2[2*7+1]);      // 61-61
        ret.push_back(ret2[2*7+0]);    ret.push_back(ret2[2*7+1]);      // 62-62

        for(i=7; i>=1; i-- ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);   // 63-69
        }
        for(i=8; i<=15; i++ ) 
        {
            ret.push_back(ret2[2*i+0]); ret.push_back(ret2[2*i +1]);   // 70-77
        }

        ret.push_back(ret2[2*15+0]);    ret.push_back(ret2[2*15+1]);   // 78-78
        ret.push_back(ret2[2*15+0]);    ret.push_back(ret2[2*15+1]);   // 79-79
        ret.push_back(ret2[2*48+0]);    ret.push_back(ret2[2*48+1]);   // 80-80
        ret.push_back(ret2[2*49+0]);    ret.push_back(ret2[2*49+1]);   // 81-81

        //FLOG("82point alldata: ------------------------------------");
        //for(int j=0; j<82; j++)
        //{
        //    FLOG("82point alldata: %d %f %f", j, ret[2*j+0], ret[2*j+1]);
        //}
        //FLOG("82point alldata: ------------------------------------");
    }
}

FOSAFER_face_align::FOSAFER_face_align() {
    predictor = new CNCNNFeatureExtractor();

    rect_scale = 5.0f;
    has_last = false;
}

FOSAFER_face_align::~FOSAFER_face_align() {
    if(predictor != nullptr)
    {
        delete predictor;
        predictor = nullptr;
    }
}

void FOSAFER_face_align::RectRegionAdjust(double& center_x, double& center_y, double& width, const cv::Rect& face) {
    double bs = 1.1189;
    double bx = 0.029;
    double by = 0.081;

    width = std::max(face.width, face.height);

    center_x = face.x + face.width / 2.0 + bx * width;
    center_y = face.y + face.height / 2.0 + by * width;

    width /= bs;
}


float FOSAFER_face_align::update(cv::Mat const &frame_image, const cv::Rect &face_rect, std::vector<cv::Point2f> *pts) {
    cv::Rect large_face_rect;

    FLOG("align frame_image: %d(H) %d(W) %d(C)", frame_image.size().height, frame_image.size().width, frame_image.channels());
    //cv::imwrite("/mnt/sdcard/fosafer/align_1.jpg",frame_image);


    double cx, cy, width;
    //cv::Point2f lt(frame_image.size().width, frame_image.size().height), rb(0, 0);
    RectRegionAdjust(cx, cy, width, face_rect);
    width *= 1.3;
    FLOG("align large_face_rect: %d(X) %d(Y) %d(W) %d(H)", large_face_rect.x, large_face_rect.y, large_face_rect.width, large_face_rect.height);
    large_face_rect.width = large_face_rect.height = width;
    large_face_rect.x = cx - width / 2;
    large_face_rect.y = cy - width / 2;
    large_face_rect = strip_rect(frame_image.size(), large_face_rect);
    FLOG("align after large_face_rect: %d(X) %d(Y) %d(W) %d(H)", large_face_rect.x, large_face_rect.y, large_face_rect.width, large_face_rect.height);


    FLOG("align after large_face_rect: %d(X) %d(Y) %d(W) %d(H)", large_face_rect.x, large_face_rect.y, large_face_rect.width, large_face_rect.height);
    cv::Rect image_rect = cv::Rect(0, 0, frame_image.size().width, frame_image.size().height);
    cv::Rect inner_rect = get_intersect_rect(image_rect, large_face_rect);
    cv::Mat tmp1 = frame_image(inner_rect).clone();
    //cv::imwrite("/mnt/sdcard/fosafer/align_2.jpg",tmp1);
    cv::Rect image_offset_rect = get_offset_rect(image_rect, inner_rect);
    cv::Rect subimage_rect = get_offset_rect(large_face_rect, inner_rect);
    cv::Mat image(large_face_rect.height, large_face_rect.width, frame_image.channels() == 3 ? CV_8UC3 : CV_8UC1);
    cv::Mat subimage = image(subimage_rect);
    frame_image(image_offset_rect).copyTo(subimage);

    //cv::imwrite("/mnt/sdcard/fosafer/align_3.jpg",subimage);
    //cv::imwrite("/mnt/sdcard/fosafer/align_4.jpg",image);
    FLOG("align inner_rect: %d(X) %d(Y) %d(W) %d(H)", inner_rect.x, inner_rect.y, inner_rect.width, inner_rect.height);
    FLOG("align image_offset_rect: %d(X) %d(Y) %d(W) %d(H)", image_offset_rect.x, image_offset_rect.y, image_offset_rect.width, image_offset_rect.height);
    FLOG("align subimage_rect: %d(X) %d(Y) %d(W) %d(H)", subimage_rect.x,  subimage_rect.y, subimage_rect.width, subimage_rect.height);
    int size_w = large_face_rect.width;
    int size_h = large_face_rect.width;
    vector<float> res;
    cv::Mat image_gray;
    double tms = now_ms();

    if(image.channels() == 3) {
        image_gray = image;
        //cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);     
    } else {
        image_gray = image;
    }

    float score = predictor->ExtractFeature(image_gray, res);
    
    FLOG("point timeX: %f", now_ms() - tms); 
    pts->resize(res.size() / 2);
    for(int idx = 0; idx < res.size() / 2; idx++) {
        cv::Point2f p((int)res[idx * 2], (int)res[idx * 2 + 1]);
        //p *= size / 160.;
        (*pts)[idx] = p + cv::Point2f(large_face_rect.x, large_face_rect.y);
        // if(idx<5)
        // {
        //     FLOG("align  82point after pointx %f", (*pts)[idx].x);
        //     FLOG("align  82point after pointy %f", (*pts)[idx].y);
        // }       
    }
    
    has_last = true;
    cv::Point2f lt(frame_image.size().width, frame_image.size().height), rb(0, 0);

    for(int idx = 0; idx < pts->size(); idx++) {
        lt.x = min(lt.x, (*pts)[idx].x);
        lt.y = min(lt.y, (*pts)[idx].y);
        rb.x = max(rb.x, (*pts)[idx].x);
        rb.y = max(rb.y, (*pts)[idx].y);
    }

    FLOG("align  82point rect  %f %f", lt.x,lt.y);
    FLOG("align  82point rect  %f %f", rb.x,rb.y);

    cv::Point2f landmark_center = lt + rb;
    landmark_center.x /= 2.0f;
    landmark_center.y /= 2.0f;
    width = std::max(rb.x - lt.x + 1, rb.y - lt.y + 1);
    landmark_border_rect = strip_rect(frame_image.size(), cv::Rect(landmark_center.x - width / 2, landmark_center.y - width / 2, width, width));
    width *= 1.35;
    //cv::imwrite("/mnt/sdcard/fosafer/landmark_border_rect.jpg",frame_image(landmark_border_rect));
    FLOG("align  landmark_center %f %f", landmark_center.x, landmark_center.y);
    FLOG("align  landmark_border_rect %d, %d, %d, %d", landmark_border_rect.x, landmark_border_rect.y, landmark_border_rect.width, landmark_border_rect.height);

    double half_width = width / 2;
    last_rect.x = landmark_center.x - half_width;
    last_rect.y = landmark_center.y - half_width;
    last_rect.width = width;
    last_rect.height = width;
    FLOG("align last_rect %d, %d, %d, %d", last_rect.x, last_rect.y, last_rect.width, last_rect.height);

    if(clv_res) {
        return 0.0;
    }

    return score;
}

cv::Rect FOSAFER_face_align::get_rect() {
    return landmark_border_rect;
}

void FOSAFER_face_align::clear_state() {
    has_last = false;
}

static void remap_pts(vector<cv::Point2f> const &pts, vector<cv::Point2f> *res) {
    int map_size = 400;

    cv::Point2f lt, rb(0, 0);
    lt = pts[0];
    rb = pts[0];
    for(int idx = 0; idx < pts.size(); idx++) {
        lt.x = min(lt.x, (pts)[idx].x);
        lt.y = min(lt.y, (pts)[idx].y);
        rb.x = max(rb.x, (pts)[idx].x);
        rb.y = max(rb.y, (pts)[idx].y);
    }

    cv::Mat image(map_size, map_size, CV_8UC3);
    memset(image.data, 0, map_size * map_size * 3);

    cv::Rect2f alignrect(lt, rb);
    double factor = double(map_size - 80) /  max(alignrect.width, alignrect.height);

    if(res) res->resize(pts.size());

    for(int idx = 0; idx < pts.size(); idx++) {
        cv::Point2f target_point = (pts[idx] - lt) * factor + cv::Point2f(40, 40);
        if (res) (*res)[idx] = target_point;
    }
}

//FOSAFER_alive_detection::FOSAFER_alive_detection(const char *model_dir):align_(model_dir) {

FOSAFER_alive_detection::FOSAFER_alive_detection(){
    align_ = new FOSAFER_face_align();
    this->open_alive_detection_ = false;
    this->last_score_ = 0;
    this->asum_area_buffer_.resize(this->history_len_, 0.0);
    this->filled_.resize(this->history_len_, false);
    this->asum_nose_buffer_.resize(this->history_len_);
    this->asum_mouth_buffer_.resize(this->history_len_);
    this->cur_asum_idx_ = 0;
    this->mouth_points_.resize(12);
    this->cur_frame_idx_ = 0;
    this->last_face_rect_.x = -1;
}

FOSAFER_alive_detection::~FOSAFER_alive_detection() {
    if(align_) {
        delete align_;
        align_ = nullptr;
    }
}


void FOSAFER_alive_detection::init(FaceDetectParam param) {
    this->param_ = param;
    this->open_alive_detection_ = false;
    this->last_score_ = 0;
    this->asum_area_buffer_.resize(this->history_len_, 0.0);
    this->filled_.resize(this->history_len_, false);
    this->asum_nose_buffer_.resize(this->history_len_);
    this->asum_mouth_buffer_.resize(this->history_len_);
    this->cur_asum_idx_ = 0;
    this->mouth_points_.resize(12);
    this->cur_frame_idx_ = 0;
    this->last_face_rect_.x = -1;
}

void FOSAFER_alive_detection::set_status(int status) {
    if(status) {
        open_alive_detection_ = true;
    } else {
        open_alive_detection_ = false;
    }

    cur_frame_idx_ = 0;

    FLOG("Inner open alive detection status: %d", open_alive_detection_ ? 1 : 0);
}
#ifdef TEST_FUNCTION
void FOSAFER_alive_detection::mouse_detection(cv::Mat const &frame_image, float *close_prob, float *open_prob) {
    cv::Mat frame_gray;
    cv::cvtColor(frame_image, frame_gray, CV_RGB2GRAY);
    cv::Point2f central;
    central.x = (pts_[52].x + pts_[46].x) / 2;
    central.y = (pts_[52].y + pts_[46].y) / 2;
    
    int cropsize = MAX(pts_[49].x - pts_[43].x, pts_[46].y - pts_[52].y) * 1.1;
    int cropsize_half = cropsize / 2;

    cv::Point2f upleft;
    upleft.x = central.x - cropsize_half;
    upleft.y = central.y - cropsize_half;

    cv::Mat mouth_image = crop_with_black(frame_gray, cv::Rect(upleft.x, upleft.y, cropsize, cropsize));
    mouth_image = mouth_image(cv::Rect(0, mouth_image.size().height / 5.0, mouth_image.size().width, mouth_image.size().height * 3.0 / 5.0));
    int ret = checkMouthState(mouth_image);

    if(close_prob)  *close_prob = (ret == 1 ? 1 : 0);
    if(open_prob) *open_prob = (ret == -1 ? 1 : 0);
}
#endif

float eval_face_quality(cv::Mat const &face_image) {

    FLOG("image size: %d %d", face_image.size().width, face_image.size().height);

    if(face_image.empty()) return -1;
    unsigned int pxnum = face_image.size().area();
    double sum = 0;
    if(face_image.channels() == 3) {
        cv::Mat gray_image;
        cv::cvtColor(face_image, gray_image, cv::COLOR_RGB2GRAY);
        for(unsigned int z = 0;z < pxnum;z++) {
            sum += gray_image.data[z];
        }
    } else if(face_image.channels() == 1) {
        for(unsigned int z = 0;z < pxnum;z++) {
            sum += face_image.data[z];
        }
    } else {
        FLOG("image channal must be 1 or 3!");
        return -1.0f;
    }

    double mean = sum / pxnum;

    FLOG("ILLU: %f", mean);
    return (float)mean;
}

bool compareVector(const FaceRect &a, const FaceRect &b){
    return a.w * a.h > b.w * b.h;
}

int FOSAFER_alive_detection::update(cv::Mat const &frame_image, const cv::Rect &face_rect, std::vector<cv::Point2f> *pts,
Info *info, float minPercent, float maxPercent) 
{
    std::cout << "Enter alive detect update" << std::endl;
    #define MIN_MOUTH_AREA_CHANGE 4500
    #define MAX_NOSE_POS_CHANGE 0.7
    #define MAX_TILE_DEGREE 8
    #define MAX_MID_DEGREE 0.35
    #define MIN_MOUTH_DISTANCE 0.6
    #define MIN_CLOSE_TIMES 7
    #define MIN_LIGHT 40
    #define MAX_LIGHT 210

    std::cout << "alive detect 1" << std::endl;
    
    vector<FaceRect> faces;
    bool use_landmark = false;
    
    int not_use_facedet = 1;
    use_landmark = true;
    FLOG("detectAlign USE LANDMARK");
    double start_time = now_ms();
    last_score_ = align_->update(frame_image, face_rect, &pts_);
    FLOG("detectAlign landmark face: %.02ff ms !", now_ms() - start_time);
    if(last_score_ < 0.6) {
        FLOG("detectAlign NO FACE LANDMARK %f.", last_score_);
        return FACE_ALIVE_UNDETECTED;
    }

    if(!__CLV__()) {
        return FACE_ALIVE_UNDETECTED;
    }

    if(pts)
        *pts = pts_;

    if(!use_landmark) return FACE_ALIVE_DETECTED;
    
#ifdef TEST_FUNCTION
    float mouth_open_prob, mouth_close_prob;
    mouse_detection(frame_image, &mouth_open_prob, &mouth_close_prob);
#endif
    
    //for debug
#ifdef TEST_FUNCTION
    cv::Mat frame_image_copy = frame_image.clone();
    cv::cvtColor(frame_image_copy, frame_image_copy, COLOR_RGB2BGR);
    char buf[100];
    sprintf(buf, "%f", last_score_);
    cv::putText(frame_image_copy, buf, cv::Point(20, 20), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0));

    for(int idx = 0; idx < pts_.size(); idx++) {
        cv::circle(frame_image_copy, pts_[idx], 2, cv::Scalar(0, 255, 0), 1);
    }
#endif
    float mid_degree = check_mid(pts_);
    float tilt_degree = check_tilt(pts_);
    float face_percent = align_->get_rect().width / float(frame_image.size().width);

    FLOG("align_ rect: %d %d %d %d", align_->get_rect().x, align_->get_rect().y, align_->get_rect().width, align_->get_rect().height);

    float illu_status = eval_face_quality(frame_image(strip_rect(frame_image.size(), align_->get_rect())));

    FLOG("mid_degree: %f tilt_degree: %f face_percent: %f illu %f", 
        mid_degree, tilt_degree, face_percent, illu_status);
    // if(info) {
    //     info->face_status = 0;
    //     if(face_percent < minPercent) {
    //         //info->face_status |= FACE_STATUS_SMALL;
    //         FLOG("too small, X:%d Y:%d W:%d H:%d", align_->get_rect().x, align_->get_rect().y, align_->get_rect().width, align_->get_rect().height);
    //     } 
    //     if(face_percent > maxPercent) {
    //         //info->face_status |= FACE_STATUS_LARGE;
    //         FLOG("too large, X:%d Y:%d W:%d H:%d", align_->get_rect().x, align_->get_rect().y, align_->get_rect().width, align_->get_rect().height);
    //     } 
    //     if(illu_status < MIN_LIGHT) {
    //         //info->face_status |= FACE_STATUS_DARK;
    //         FLOG("too dark %f", illu_status);
    //     } 
    //     if(illu_status > MAX_LIGHT) {
    //         //info->face_status |= FACE_STATUS_BRIGHT;
    //         FLOG("too light %f", illu_status);
    //     } 
    // }
    
    if(!open_alive_detection_) {
        FLOG("FACE_ALIVE_DETECTED");
        return FACE_ALIVE_DETECTED;
    } else {
        FLOG("FACE_ALIVE_DETECTED_AND_ALIVE");
        return FACE_ALIVE_DETECTED_AND_ALIVE;
    }
}
    

