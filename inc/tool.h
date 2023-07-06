#ifndef __FOSAFER_FACEDET_TOOL_
#define __FOSAFER_FACEDET_TOOL_
#include <string>
#include <vector>
#include <cstdio>
#include <map>

#include<opencv2/opencv.hpp>

using namespace std;

extern int g_pic_alive;  		  // kzq
extern bool g_isCheckingEnv;    // kzq

extern map<unsigned int, unsigned int> g_code;

#ifdef __ANDROID__
#include <android/log.h>
void ExtendLog(const char *file, int lineNumber, const char *functionName, const char *format, ...);
#elif defined(__IOS__)
#include <Foundation/Foundation.h>
void ExtendNSLog(const char *file, int lineNumber, const char *functionName, NSString *format, ...);
#else
void ExtendLog(const char *file, int lineNumber, const char *functionName, const char *format, ...);
#endif

#if NO_LOG
#define FLOG(...)
#else
#ifdef __ANDROID__
    #define FLOG(...) ExtendLog(__FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)
#elif defined(__IOS__)
    #define FLOG(format, ...) ExtendNSLog(__FILE__, __LINE__, __PRETTY_FUNCTION__, [NSString stringWithUTF8String:format], ##__VA_ARGS__)
#else
    #ifdef _MSC_VER
        #define FLOG(...) ExtendLog(__FILE__, __LINE__, NULL, __VA_ARGS__)
    #else
        #define FLOG(...) ExtendLog(__FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)
    #endif
#endif
#endif //NO_LOG

std::string GetTimeStr();
std::string GetTimeStr2();
std::string int2str(int a);
cv::Rect get_intersect_rect(cv::Rect a, cv::Rect b);

template <class TYPE>
inline TYPE square(TYPE x) {
    return x * x;
}

template <class TYPE>
double get_point_distance(TYPE p1, TYPE p2) {
    return sqrt(double(square(p1.x - p2.x)) + double(square(p1.y - p2.y)));
}


cv::Mat crop_with_black(cv::Mat const &image, cv::Rect rect); 
double square(double x);
cv::Rect get_offset_rect(cv::Rect base, cv::Rect inner);
cv::Rect strip_rect(cv::Size imgSize, cv::Rect rect);

void decode_string(const char *str, int len, std::string &decoded_str);
double now_ms(void);

int __CLV__();

// namespace cv {
    
// }
#endif