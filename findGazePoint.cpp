#include "findGazePoint.h"

cv::Point2f findGazePoint(cv::Point2f right_eye, cv::Point2f left_eye)
{
	float x = -917.1 + right_eye.x * 55.027 + 100;
	float y = 355.7 + right_eye.y * -61.32 - 150;
	cv::Point2f point(x,y);
	return point;
}
