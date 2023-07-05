#include "ncnnssd.h"

#define FACE_SUCCESS 0
#define FACE_PROCESS 1
#define FACE_ONE_SUCCESS 2
#define FACE_NO_READY -1
#define FACE_ERROR_RELEASE -2
#define FACE_ERROR_ACTION -3
#define FACE_ERROR_NULL -4
#define FACE_ERROR_NO_FACE -5
#define FACE_ERROR_MULTIFACE -6
#define FACE_ERROR_DECODE -7
#define FACE_DET_FAILED -8
#define FACE_NO_FACE -9
#define FACE_LOW_QUALITY -10


#ifndef DLL_PUBLIC
    #if defined(__clang__) || defined(__GNUC__)
        #define DLL_PUBLIC __attribute__((visibility("default")))
    #else
        #ifdef DLL_EXPORT
            #define DLL_PUBLIC __declspec(dllexport)
        #else
            #define DLL_PUBLIC __declspec(dllimport)
        #endif
    #endif
#endif

#ifndef DLL_LOCAL
    #if defined(__clang__) || defined(__GNUC__)
        #define DLL_LOCAL  __attribute__ ((visibility("hidden")))
    #else
        #define DLL_LOCAL
    #endif
#endif

#ifndef APIENTRY
    #ifdef _MSC_VER
        #define APIENTRY __stdcall
    #else
        #define APIENTRY
    #endif
#endif


#ifdef __cplusplus
extern "C" {
#endif
    typedef struct Image
    {
        unsigned char* data;
        int height;
        int width;
        int channel;
        int size;
        float face_rect[10][5];
        float alive_score;
        int face_num;
        int brightness;
        int resolution;
    } Image;

    typedef void* FACERECOG_ENGINE_HANDLE;
    // return 成功时返回引擎句柄，否则返回NULL
    DLL_PUBLIC FACERECOG_ENGINE_HANDLE APIENTRY FOSAFER_FaceRECOG_Initialize();

    // pHandle[in] 人脸检测引擎句柄
    // return 释放成功返回FACE_SUCCESS，释放失败返回FACE_ERROR_RELEASE，句柄为空返回FACE_ERROR_NULL
    DLL_PUBLIC int APIENTRY FOSAFER_FaceRECOG_Release(FACERECOG_ENGINE_HANDLE pHandle);

    // 动作活体调用的主函数
    // pHandle[in] 人脸检测引擎句柄
    // info[out] 人脸信息输出，目前只输出face_rect，四个值对应x,y,w,h
    // image[in], 已编码的输入图像，image->size表示图像字节长度
    //            当需要限定画面中人脸检测的区域时, 需要传入image->face_rect, 并且设置好image->width和height
    // rotateCW[in]: 旋转角度,0/90/180/270
    // return 图像解码失败或未检测到人脸返回FACE_ERROR_NO_FACE，图像质量过低返回FACE_ERROR_LOW_QUALITY，句柄未空返回FACE_ERROR_NULL
    DLL_PUBLIC int APIENTRY FOSAFER_FaceRECOG_Detect(FACERECOG_ENGINE_HANDLE pHandle, Image* image, int rotateCW);

#ifdef __cplusplus
}
#endif

class CFosaferFaceRecogBackend {
public:
    CFosaferFaceRecogBackend();
    ~CFosaferFaceRecogBackend();
    
    bool init();
    int detect(Image* image_input, int rotateCW);
    
    void rotate_image_90n(cv::Mat &src, cv::Mat &dst, int angle);
    //static bool compareVector(const ssdFaceRect &a, const ssdFaceRect &b);
    cv::Mat ResizeImage(cv::Mat image, int maxDimSize, double *scale_used);

private:
    void detect_brightness(cv::Mat input_img, float& cast, float& da);
    double calculate_average_gray_value(const cv::Mat& image_gray, const ssdFaceRect &face);
    double cal_variance(const cv::Mat& image_gray, const ssdFaceRect &face);

private:
    double image_max_dim_ = 600;
    double scale_factor_ = 1;

    ncnnssd *ncnnssd_;
};



