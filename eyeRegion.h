/*
 * this class is used to detect any object
 * */

#ifndef EYEREGION_H
#define EYEREGION_H

#include "opencv2/imgproc/imgproc.hpp"

class EyeRegion
{
	public:
		cv::Rect eyeRect;
		cv::Point2f eyeCorner;
		cv::Rect eyeBorder;
		cv::Point2f pupil;
		cv::Mat mask;
		cv::RotatedRect eyerect;
};

#endif
