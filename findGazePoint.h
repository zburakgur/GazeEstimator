#ifndef EYE_GAZE_H
#define EYE_GAZE_H

#include "opencv2/imgproc/imgproc.hpp"

cv::Point2f findGazePoint(cv::Point2f right_eye, cv::Point2f left_eye);

#endif
