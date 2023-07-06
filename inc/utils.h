#ifndef __FOSAFER_FACE_UTILS_
#define __FOSAFER_FACE_UTILS_

#include<opencv2/opencv.hpp>
#include <cmath>

template <class TYPE>
double DistanceTo(TYPE point1, TYPE point2) {
    const double dx = std::abs(point1.x - point2.x);
    const double dy = std::abs(point1.y - point2.y);
    if (dx <= 0.0) {
        return std::max(0.0, dy);
    }
    if (dy <= 0.0) {
        return dx;
    }
    return std::hypot(dx, dy);
}


#endif //__FOSAFER_FACE_UTILS_