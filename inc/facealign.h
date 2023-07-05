#ifndef __FOSAFER_FACE_ALIGN__
#define __FOSAFER_FACE_ALIGN__

#include <vector>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include "ncnnssd.h"
#include "ncnn_feature_extractor.h"
#include <vector>

#define FACE_ALIVE_DETECTED_AND_ALIVE 1
#define FACE_ALIVE_DETECTED 0
#define FACE_ALIVE_DETECTED_AND_FAKE -1
#define FACE_ALIVE_UNDETECTED -2
#define FACE_ALIVE_MULTIFACE -3

class CNCNNFeatureExtractor2;

#ifndef FACERECT_STRUCT
#define FACERECT_STRUCT

struct FaceRect{
	int x, y, w, h;
	float confidence;
};

#endif

#define FRONT 1
#define LEFT_TILT_FRONT 2
#define RIGHT_TILT_FRONT 4
#define LEFT_HALF_PROFILE 8
#define RIGHT_HALF_PROFILE 16

struct FaceDetectParam
{
	int nFaceMinSize; 
	int nStep;        
	int nNumCaThresh; 
	int nLayer;       
	int cf_flag;     
	int num_expect;   
	float imageScale; 
	bool bBiggestFaceOnly; 
	bool bUseSkinColor; 
} ;

typedef struct _Image
{
	unsigned char* data;
	char name[256];
	float eye_center[2];
	float mouth_top[2];
	float face_rect[4];
	float land_mark[164];
	int height;
	int width;
	int channel;
	int size;
	int type; // type为1表示活体，为0表示非活体
} Image;

typedef struct _Info
{
	float face_rect[4];
	float left_eye[2];
	float right_eye[2];
	short land_mark[256];         // 预留128个关键点位置: (x[0],x[1]), (x[2],x[3]),....
	short land_mark_count;        // 关键点数量，如果有10个关键点，land_mark中合法长度为20
	short face_status;            // 人脸状态，用(face_status & FACE_STATUS_POSTURE_VALID) 查询对应标志位是否合法
	int collected_image_count;    // 收集的人脸数量
	int processed_image_count;    // 收集的进程图像
	int alive_action;             // 0:眨眼 1:左摇 2:右摇 3:点头 4:张嘴
	int error_action;             // 1表示动作错误
	int multi_face;               // 1:多人脸 0:单人脸
	float debug[3];
} Info;

class FOSAFER_face_align {
public:
    FOSAFER_face_align(const char *model_dir);
    ~FOSAFER_face_align();
    void RectRegionAdjust(double& cx, double& cy, double& width, const cv::Rect& face);
    float update(cv::Mat const &frame_image, cv::Rect facerect, std::vector<cv::Point2f> *pts);
    void clear_state();
    cv::Rect get_rect();
private:
    CNCNNFeatureExtractor2 *predictor;
    
    float rect_scale;
    bool has_last;
    cv::Rect last_rect;
    cv::Rect landmark_border_rect;
};

class FOSAFER_alive_detection {
public:
	FOSAFER_alive_detection(const char *model_dir);
	~FOSAFER_alive_detection();

	void init(FaceDetectParam param);

    //  
	int update(cv::Mat const &frame_image, cv::Rect *face_rect, std::vector<cv::Point2f> *pts, Info *info, float minPercent, float maxPercent);

	void set_status(int status);

    // 
    void fosaferdetectface_ssd(cv::Mat const &frame_image, std::vector<FaceRect> &faces);

private:
#ifdef TEST_FUNCTION
    void mouse_detection(cv::Mat const &frame_image, float *close_prob, float *open_prob);
#endif
	FaceDetectParam param_;
	static const int history_len_ = 25;
	static const int inner_landmark_cycle_ = 30;
    int cur_asum_idx_;
    unsigned int cur_frame_idx_;

    std::vector< std::vector<cv::Point2f> > asum_nose_buffer_;
    std::vector<double> asum_area_buffer_;
    std::vector<int> asum_mouth_buffer_;

    std::vector<bool> filled_;
    float last_score_;
    cv::Rect last_face_rect_;

    ncnnssd * ncnnssd_;

    FOSAFER_face_align align_;
    std::vector<cv::Point2f> pts_;
    std::vector<cv::Point2f> normaled_landmark_;

    std::vector<cv::Point2f> mouth_points_;
    bool open_alive_detection_;
};
#endif
