/*
 * this class is used to detect gaze feature
 * */

#ifndef FEATURESET_H
#define FEATURESET_H

#include "opencv2/imgproc/imgproc.hpp"

class FeatureSet
{
	public:
		int pupil_x;
		int pupil_y;
		int vert_distanceOfeyeLids;		
		int n_comp;
		int centroids;
		int area;
};

#endif
