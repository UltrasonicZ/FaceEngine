#include "tool.h"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cstdarg>
#include <ctime>
#include <string>
#include <cstring>
#ifndef _MSC_VER
#include <sys/time.h>
#else
#include <Windows.h>
#endif
#ifdef __ANDROID__
#include <android/log.h>
#endif
#include <map>

#ifdef __IOS__
#import <Foundation/Foundation.h>
#endif
using namespace std;

const char *key;

#ifdef OPEN_FILELOG
extern FILE *global_logout;
#endif

#ifdef OPEN_APILOG
extern vector<string> global_logstrs;
extern bool open_apilog;
extern unsigned int global_process_call_number;
#endif

int g_pic_alive = 0;             // kzq
bool g_isCheckingEnv = true;  // kzq

map<unsigned int, unsigned int> g_code
{
	{0x48752111, 0x6964}, 
	{0x46357267, 0x3452}, 
	{0x16452826, 0x3E63}
};

double get_point_distance(cv::Point2f p1, cv::Point2f p2);
double now_ms(void);
/*
void FLOG(const char * __restrict format, ...) { 
    va_list args;
    va_start(args, format);
#ifdef __IOS__
    NSLogv([NSString stringWithUTF8String:format], args);
    va_end(args);
#elif defined(__ANDROID__)
    __android_log_print(ANDROID_LOG_DEBUG, "FosaferFaceDet", format, args);
    va_end(args);
#else
    static char szMsg[4096];
    vsprintf(szMsg, format, args);    
    va_end(args);
    printf("%s\n",szMsg);
#endif
}*/

#ifdef _MSC_VER
int gettimeofday(struct timeval *tp,void *tzp){
	time_t clock;
	struct tm tm;
	SYSTEMTIME wtm;
	GetLocalTime(&wtm);
	tm.tm_year = wtm.wYear - 1900;
	tm.tm_mon = wtm.wMonth - 1;
	tm.tm_mday = wtm.wDay;
	tm.tm_hour = wtm.wHour;
	tm.tm_min = wtm.wMinute;
	tm.tm_sec = wtm.wSecond;
	tm.tm_isdst = -1;
	clock = mktime(&tm);
	tp->tv_sec = clock;
	tp->tv_usec = wtm.wMilliseconds*1000;
	return 0;
}
#endif

#ifdef __ANDROID__

void ExtendLog(const char *file, int lineNumber, const char *functionName, const char *format, ...) {
//#ifndef RELEASE
    static vector<char> buf;

    va_list argList;
    va_start(argList, format);
    va_list argList2;
    va_copy(argList2, argList);
    unsigned int target_size = 1 + vsnprintf(NULL, 0, format, argList);
    va_end(argList);

    if(target_size > buf.size()) {
        buf.resize(target_size);
    }

    vsnprintf(buf.data(), buf.size(), format, argList2);
    va_end(argList2);

    #ifdef OPEN_APILOG
    if(open_apilog) {
        char strbuf[1024];
        snprintf(strbuf, 1023, "[%d:%d] %s", global_process_call_number, lineNumber, buf.data());
        global_logstrs.push_back(string(strbuf));
    } else {
        __android_log_print(ANDROID_LOG_DEBUG, "FosaferFaceDet", "[%s:%d:%f] %s", file, lineNumber, now_ms(), buf.data());
    }
    #elif defined(OPEN_FILELOG)
    if(global_logout) {
        fprintf(global_logout, "[%s:%d time:%s] %s\n", file, lineNumber, GetTimeStr2().c_str(), buf.data());
    } else {
        __android_log_print(ANDROID_LOG_DEBUG, "FosaferFaceDet", "[%s:%d:%f] %s", file, lineNumber, now_ms(), buf.data());
    }
    #else 
        __android_log_print(ANDROID_LOG_DEBUG, "FosaferFaceDet", "[%s:%d:time:%s] %s", file, lineNumber, GetTimeStr2().c_str(), buf.data());
    #endif
   
//#endif
}
#elif defined(__IOS__)
void ExtendNSLog(const char *file, int lineNumber, const char *functionName, NSString *format, ...)
{

//#ifndef RELEASE
    // Type to hold information about variable arguments.
    va_list ap;
 
    // Initialize a variable argument list.
    va_start (ap, format);
     
    // NSLog only adds a newline to the end of the NSLog format if
    // one is not already there.
    // Here we are utilizing this feature of NSLog()
    if (![format hasSuffix: @"\n"])
    {
        format = [format stringByAppendingString: @"\n"];
    }
     
    NSString *body = [[NSString alloc] initWithFormat:format arguments:ap];
     
    // End using variable argument list.
    va_end (ap);
     
    NSString *fileName = [[NSString stringWithUTF8String:file] lastPathComponent];
    fprintf(stderr, "[%s] [%s:%d] %s",
            functionName, [fileName UTF8String],
            lineNumber, [body UTF8String]);
//#endif

}
#else
void ExtendLog(const char *file, int lineNumber, const char *functionName, const char *format, ...) {
#ifndef RELEASE
    time_t timer;
    struct tm *tblock;
    timer = time(NULL);
    tblock = localtime(&timer);
    tblock->tm_year += 1900;

	static vector<char> buf;

    va_list argList;
    va_start(argList, format);
    va_list argList2;
    va_copy(argList2, argList);
    int target_size = 1 + vsnprintf(NULL, 0, format, argList);
    va_end(argList);

    if(target_size > buf.size()) {
        buf.resize(target_size);
    }

    vsnprintf(buf.data(), buf.size(), format, argList2);
    va_end(argList2);
    printf("[%s:%d] %s\n", file, lineNumber, buf.data());
#endif
}
#endif

std::string int2str(int a){
    char   t[256];
    memset(t, 0, sizeof(t));
    sprintf(t, "%d", a);
    std::string s = t;
    return s;
}

std::string GetTimeStr() {
    time_t timer;
    struct tm *t;
    timer = time(NULL);
    t = localtime(&timer);
    char fpath[100];

    sprintf(fpath, "%04d-%02d-%02d-%02d-%02d-%02d", t->tm_year+1900,t->tm_mon+1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    return string(fpath);
}

std::string GetTimeStr2() {
    time_t timer;
    struct tm *t;

    struct timeval tv;
     
    gettimeofday(&tv, NULL);
    timer = time(NULL);
    t = localtime(&timer);
    char fpath[100];

    sprintf(fpath, "%04d-%02d-%02d-%02d-%02d-%02d-%06d", t->tm_year+1900, t->tm_mon+1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec, tv.tv_usec);
    return string(fpath);
}

cv::Rect get_intersect_rect(cv::Rect a, cv::Rect b) {
    cv::Point lefttop, rightbottom;
    lefttop.x = max(a.x, b.x);
    lefttop.y = max(a.y, b.y);

    rightbottom.x = min(a.x + a.width, b.x + b.width);
    rightbottom.y = min(a.y + a.height, b.y + b.height);

    if(lefttop.x > rightbottom.x || lefttop.y > rightbottom.y) 
        return cv::Rect(0, 0, 0, 0);

    cv::Rect res(lefttop, rightbottom);
    return res;
}

cv::Rect get_offset_rect(cv::Rect base, cv::Rect inner) {
    cv::Rect res;
    res.x = inner.x - base.x;
    res.y = inner.y - base.y;
    res.width = inner.width;
    res.height = inner.height;

    return res;
}

cv::Mat crop_with_black(cv::Mat const &image, cv::Rect rect) {
    cv::Rect image_rect = cv::Rect(0, 0, image.size().width, image.size().height);
    cv::Rect inner_rect = get_intersect_rect(image_rect, rect);
    cv::Mat res_image(rect.height, rect.width, image.channels() == 3 ? CV_8UC3 : CV_8UC1);
    memset(res_image.data, 0, rect.height * rect.width * res_image.channels());
        
    if(inner_rect.area() > 0) {
        cv::Rect image_offset_rect = get_offset_rect(image_rect, inner_rect);
        cv::Rect subimage_rect = get_offset_rect(rect, inner_rect);
        cv::Mat subimage = res_image(subimage_rect);
        image(image_offset_rect).copyTo(subimage);
    } 

    return res_image;
}

cv::Rect strip_rect(cv::Size imgSize, cv::Rect rect) {
    cv::Point2f lefttop, rightbottom;
    lefttop.x = rect.x;
    lefttop.y = rect.y;
    rightbottom.x = lefttop.x + rect.width - 1;
    rightbottom.y = lefttop.y + rect.height - 1;

    lefttop.x = max(0.0f, lefttop.x);
    lefttop.y = max(0.0f, lefttop.y);
    rightbottom.x = min(imgSize.width - 1.0f, rightbottom.x);
    rightbottom.y = min(imgSize.height - 1.0f, rightbottom.y);

    return cv::Rect(lefttop, rightbottom);
}

void decode_string(const char *str, int len, std::string &decoded_str) {
    int key_len = strlen(key);
    decoded_str.resize(len);
    const unsigned char *ptr = (const unsigned char *)str;
    for(int idx = 0;idx < len;idx++) {
        unsigned char cur = *ptr++;
        cur ^= key[idx % key_len];
        decoded_str[idx] = cur;
    }
    
}

extern int clv_res;
int __CLV__() {
    time_t timer;
    struct tm *tblock;
    timer = time(NULL);
    tblock = localtime(&timer);
    tblock->tm_year += 1000 + 900;
    int success = 0;
    do {
        if (tblock->tm_year > (2000 + 19)) {
            break;
        } else if(tblock->tm_year < (2000 + 19)) {
            success = 1;
            break;
        } else if(tblock->tm_year == (2000 + 19)) {
            if(tblock->tm_mon + 1 < 8) {
                success = 1;
                break;
            }
        }
    } while(0);

    if(!success) {
    	clv_res += 1;
    	if(clv_res == 0) clv_res = 1;
    } else {
        clv_res = 0;
    }

    clv_res = 0;
    success = 1;
    
    return success;
}

double now_ms(void) {
#ifndef _MSC_VER
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000. + tv.tv_usec/1000.;
#else
	return GetTickCount();
#endif
}
