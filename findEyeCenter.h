#ifndef EYE_CENTER_H
#define EYE_CENTER_H

#include "opencv2/imgproc/imgproc.hpp"
#include "eyeRegion.h"

cv::Point findEyeCenter(cv::Mat eyeROI);

#endif
