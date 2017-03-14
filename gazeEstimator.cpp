#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ml.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include "FeatureSet.h"
#include "findEyeCenter.h"
#include "constants.h"

using namespace dlib;
using namespace std;

//global variables
FeatureSet f_set;
cv::Mat frame;
int gazeArea = 6;
int gazeAreaX;
int gazeAreaY;
string gazeEstimatorXML;
int label_indexX = 0;
int label_indexY = 0;
cv::Ptr<cv::ml::ANN_MLP> mlpX;
cv::Ptr<cv::ml::ANN_MLP> mlpY;
int numberOfHiddenLayer;
cv::Mat trainingFeaturesX;
cv::Mat trainingFeaturesY;
cv::Mat calibScreen;
std::vector<int> trainingLabelsX;
std::vector<int> trainingLabelsY;
std::vector<int> bufferX;
std::vector<int> bufferY;
int buffer_size = 15;
int prev_index = -1;
bool right_click = false;
bool left_click = false;
string message_text;

//declerations
FeatureSet detectFeature(std::vector<image_window::overlay_line> &eye_lines, int eye_id);
void analize_eye(dlib::point & left_corner, dlib::point & rigth_corner, int & top_eye, int & bottom_eye,
				std::vector< std::vector<cv::Point> > &  eye_contour, std::vector<image_window::overlay_line> &eye_lines);
void find_EyeCenterFeatures(cv::Rect rect_eyeBorder, FeatureSet & ftmp_set, dlib::point corner);
cv::Point refining(cv::Mat & eye_gray, cv::Point pupil);
float calTotalVal(cv::Mat & tmp);
void find_corneaFeatures(cv::Rect rect_eyeBorder, std::vector< std::vector<cv::Point> > &  eye_contour, FeatureSet & ftmp_set);
void gazeEstimate();
void draw_gazePoint(int x_position, int y_position);
void trainData();
bool elemination(int & x_position, int & y_position);
void showCalibration(int index_val);

/*
* mouse click listener of OpenCV
*/
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if  ( event == cv::EVENT_LBUTTONDOWN )
    {
		if(!left_click)
		{
			left_click = true;
		}
		else
		{
			left_click = false;
			label_indexX++;	
		}
		
	}
	else if(event == cv::EVENT_RBUTTONDOWN)
    {
		right_click = true;  
		label_indexX = 0;
		label_indexY = 0;
		
		cv::Mat trainClassesX = cv::Mat::zeros(trainingFeaturesX.rows,3,CV_32FC1);
		for( int i = 0; i <  trainClassesX.rows; i++ )
		{	
			int col = trainingLabelsX[i];
			trainClassesX.at<float>(i,col) = 1;
		}
		
		cv::Mat trainClassesY = cv::Mat::zeros(trainingFeaturesY.rows,2,CV_32FC1);
		for( int i = 0; i <  trainClassesY.rows; i++ )
		{	
			int col = trainingLabelsY[i];
			trainClassesY.at<float>(i,col) = 1;
		}
		
		mlpX = cv::ml::ANN_MLP::create();
		mlpY = cv::ml::ANN_MLP::create();

		cv::Mat layersX = cv::Mat(3,1,CV_32SC1);
		layersX.row(0) = cv::Scalar(trainingFeaturesX.cols);
		layersX.row(1) = cv::Scalar(numberOfHiddenLayer);
		layersX.row(2) = cv::Scalar(3);
		cv::Mat layersY = cv::Mat(3,1,CV_32SC1);
		layersY.row(0) = cv::Scalar(trainingFeaturesY.cols);
		layersY.row(1) = cv::Scalar(numberOfHiddenLayer);
		layersY.row(2) = cv::Scalar(2);

		mlpX->setLayerSizes(layersX);
		mlpX->setActivationFunction( cv::ml::ANN_MLP::SIGMOID_SYM );
		mlpX->train( trainingFeaturesX, cv::ml::ROW_SAMPLE, trainClassesX );
		cout<<"mlpx trained"<<endl;
		mlpY->setLayerSizes(layersY);
		mlpY->setActivationFunction( cv::ml::ANN_MLP::SIGMOID_SYM );
		mlpY->train( trainingFeaturesY, cv::ml::ROW_SAMPLE, trainClassesY );
		cout<<"right click mlpy trained"<<endl;
	}
}

int main( int argc, const char** argv )
{
	if(argc != 4 )
	{
		cout<<"parameter error!"<<endl;
		cout<<"./gazeEstimator (4,3,2) (3,2) 5";
		return 0;
	}
	else
	{
		gazeAreaX = atoi(argv[1]);
		gazeAreaY = atoi(argv[2]);
		if(gazeAreaX!=4&&gazeAreaX!=3&&gazeAreaX!=2)
		{
			cout<<"parameter error!"<<endl;
			return 0;
		}
		if(gazeAreaY!=3&&gazeAreaY!=2)
		{
			cout<<"parameter error!"<<endl;
			return 0;
		}
		numberOfHiddenLayer = atoi(argv[3]);
	}
	
    try
    {
        cv::VideoCapture cap(0);
        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cap >> frame;
			
			cv::namedWindow("calibration",CV_WINDOW_NORMAL);
			cv::moveWindow("calibration", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			cv::setMouseCallback("calibration", CallBackFunc, NULL);
			
            cv_image<bgr_pixel> cimg(frame);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            if(faces.size() > 0)
            {
				full_object_detection face_pose = pose_model(cimg, faces[0]);
				shapes.push_back(face_pose);
			}
			
            // Display it all on the screen
            win.clear_overlay();
            std::vector<image_window::overlay_line> pose_lines = render_face_detections(shapes);
            
            if(pose_lines.size() == 65)
            {
				//find left and right eye region
				std::vector<image_window::overlay_line> left_eye_lines;
				std::vector<image_window::overlay_line> right_eye_lines;
				std::vector<image_window::overlay_line> left_eyebrow_lines;
				std::vector<image_window::overlay_line> right_eyebrow_lines;
				for(int i=19; i<45; i++)
				{
					if(i>=19 && i<=22)
						left_eyebrow_lines.push_back(pose_lines.at(i));
					else if(i>=23 && i<=26)
						right_eyebrow_lines.push_back(pose_lines.at(i));
					if(i>=33 && i<=38)
						left_eye_lines.push_back(pose_lines.at(i));
					else if(i>=39 && i<=44)
						right_eye_lines.push_back(pose_lines.at(i));
				}
				f_set = detectFeature(left_eye_lines,0);
			}
			
			if(left_click)		trainData();
			if(right_click)		gazeEstimate();
			else 				showCalibration(-1);
			
            cv_image<bgr_pixel> cimg2(frame);
            win.set_image(cimg2);
            //imshow("face",frame);
            int c = cv::waitKey(30);
            if((char)c == 'c')		gazeAreaX = 4;
            else if((char)c == 'v')	gazeAreaX = 3;
            else if((char)c == 'b')	gazeAreaX = 2;
            else if((char)c == 'n')	gazeAreaY = 3;
            else if((char)c == 'm')	gazeAreaY = 2;
            
            win.add_overlay(pose_lines);
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

FeatureSet detectFeature(std::vector<image_window::overlay_line> &eye_lines, int eye_id)
{	
	FeatureSet ftmp_set;
	
	dlib::point left_corner(10000, 0);
	dlib::point rigth_corner(0, 0);
	int top_eye = 10000;
	int bottom_eye = 0;
	
	std::vector< std::vector<cv::Point> >  eye_contour;
    eye_contour.push_back(std::vector<cv::Point>());
	analize_eye(left_corner, rigth_corner, top_eye, bottom_eye, eye_contour, eye_lines);
	cv::Rect rect_eyeBorder;
	rect_eyeBorder.x = left_corner.x();
	rect_eyeBorder.y = top_eye;
	rect_eyeBorder.width = rigth_corner.x() - left_corner.x();
	rect_eyeBorder.height = bottom_eye - top_eye;
	if(eye_id == 0)		find_EyeCenterFeatures(rect_eyeBorder, ftmp_set, rigth_corner);		//left eye
	else 				find_EyeCenterFeatures(rect_eyeBorder, ftmp_set, left_corner);		//rigth eye
	find_corneaFeatures(rect_eyeBorder, eye_contour, ftmp_set);
	return ftmp_set;
}

void analize_eye(dlib::point & left_corner, dlib::point & rigth_corner, int & top_eye, int & bottom_eye,
				std::vector< std::vector<cv::Point> > &  eye_contour, std::vector<image_window::overlay_line> &eye_lines)
{
	for(int i=0; i<eye_lines.size(); i++)
	{
		image_window::overlay_line * line_ptr = & eye_lines.at(i);
		dlib::point * point1_ptr = &(line_ptr->p1);
		dlib::point * point2_ptr = &(line_ptr->p2);
		
		cv::Point point1( point1_ptr->x() , point1_ptr->y() );
		cv::Point point2( point2_ptr->x() , point2_ptr->y() );
		
		cv::LineIterator it(frame, point1, point2, 8);
		
		for(int j=0; j<it.count; j++, ++it)
			 eye_contour[0].push_back(it.pos());
		
		//find corners in eye
		
		if(point1_ptr->x() < left_corner.x())
			left_corner = *point1_ptr;
		if(point2_ptr->x() < left_corner.x())
			left_corner = *point2_ptr;
		if(point1_ptr->x() > rigth_corner.x())
			rigth_corner = *point1_ptr;
		if(point2_ptr->x() > rigth_corner.x())
			rigth_corner = *point2_ptr;
			
		if(point1_ptr->y() < top_eye)
			top_eye = point1_ptr->y();
		if(point2_ptr->y() < top_eye)
			top_eye = point2_ptr->y();
		if(point1_ptr->y() > bottom_eye)
			bottom_eye = point1_ptr->y();
		if(point2_ptr->y() > bottom_eye)
			bottom_eye = point2_ptr->y();
	}
}

void find_EyeCenterFeatures(cv::Rect rect_eyeBorder, FeatureSet & ftmp_set, dlib::point corner)
{
	//fix to a standart size
	cv::Mat eye_tmp = frame(rect_eyeBorder);
	std::vector<cv::Mat> rgbChannels(3);
    cv::split(eye_tmp, rgbChannels);
    cv::Mat eye_gray = rgbChannels[0];
	cv::Mat resultResized;
	float fix_val = (((float)kEyeWidth)/rect_eyeBorder.width);
	resultResized.create(round(fix_val*rect_eyeBorder.height),kEyeWidth, CV_8UC1);
	cv::resize(eye_gray, resultResized, resultResized.size(), 0, 0, cv::INTER_CUBIC);
	eye_gray = resultResized;
	
	ftmp_set.vert_distanceOfeyeLids = eye_gray.rows;
	
	//preprocess to find eye center
    if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * rect_eyeBorder.width;
		GaussianBlur( eye_gray, eye_gray, cv::Size( 0, 0 ), sigma);
	}
    equalizeHist(eye_gray, eye_gray);
	//find eye center
	cv::Point pupil = findEyeCenter(eye_gray);
	cv::Point pupil_refined = refining(eye_gray, pupil); 
	//set pupil corrdinate feature
	
	ftmp_set.pupil_x = corner.x() - rect_eyeBorder.x - pupil_refined.x;
	ftmp_set.pupil_y = (corner.y() - rect_eyeBorder.y)*fix_val - pupil_refined.y;
	
	/*
	pupil merkezini göstremek için fix value ile eski haline getirmeli
	pupil.x+=rect_eyeBorder.x;
	pupil.y+=rect_eyeBorder.y;
	circle(frame, pupil, 3, 1234);
	cv::rectangle( frame, rect_eyeBorder, cv::Scalar( 0, 0, 255));*/
}

cv::Point refining(cv::Mat & eye_gray, cv::Point pupil)
{
	cv::Point p;
	float total=10000, r=10;	
	
	medianBlur(eye_gray, eye_gray, 5);
	
	for(int i=-5; i<5; i++)
	{
		int tmp_x = pupil.x +i;
		if(tmp_x < 0 || tmp_x > eye_gray.cols)	continue;
		for(int j=-5; j<5; j++)
		{
			int tmp_y = pupil.y +j;
			if(tmp_y < 0 || tmp_y > eye_gray.rows)	continue;
			
			cv::Mat mask(eye_gray.rows, eye_gray.cols, CV_8UC1, cv::Scalar(0));
			cv::Mat tmp(eye_gray.rows, eye_gray.cols, CV_8UC1, cv::Scalar(0));
			circle(mask, cv::Point(tmp_x, tmp_y), r, cv::Scalar(255), -1);
			eye_gray.copyTo(tmp, mask);
			
			int total_tmp = calTotalVal(tmp);
			if(total_tmp < total)
			{
				total = total_tmp;
				p.x = tmp_x;
				p.y = tmp_y;
			}
		}
	}
	
	circle(eye_gray, p, r, 1234);
	circle(eye_gray, p, 3, 1234);
	imshow("eye gray", eye_gray);
	return p;
} 

float calTotalVal(cv::Mat & tmp)
{
	float val = 0;
	int pixel_num;
	for (int y = 0; y < tmp.rows; ++y) {
		const uchar *Mr = tmp.ptr<uchar>(y);
		for (int x = 0; x < tmp.cols; ++x) 	
		{
			if(Mr[x] != 0)
			{
				val += Mr[x];
				pixel_num++;
			}		
		}
	}
	return val/pixel_num;
}

void find_corneaFeatures(cv::Rect rect_eyeBorder, std::vector< std::vector<cv::Point> > &  eye_contour, FeatureSet & ftmp_set)
{
	cv::Mat eye_tmp = frame(rect_eyeBorder);
	cv::Mat eye_gray; 
	cvtColor(eye_tmp, eye_gray, CV_RGB2GRAY);
	//preprocess to find eye cornea
    if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * rect_eyeBorder.width;
		GaussianBlur( eye_gray, eye_gray, cv::Size( 0, 0 ), sigma);
	}
    equalizeHist(eye_gray, eye_gray);
	//find cornea
	cv::Mat mask(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
	cv::RotatedRect eyerect = cv::minAreaRect(cv::Mat(eye_contour[0]));
	cv::drawContours( mask,eye_contour,0, cv::Scalar(255),CV_FILLED, 8 );
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10));
	cv::morphologyEx(mask, mask, CV_MOP_CLOSE, element);
	cv::Mat maskROI = mask(rect_eyeBorder);
	
	eyerect.center.x -= rect_eyeBorder.x;
	eyerect.center.y -= rect_eyeBorder.y;
	
	cv::Mat cornea;
	eye_gray.copyTo(cornea, maskROI);
	
	//fix to a standart orientation
	cv::Mat rotmat= cv::getRotationMatrix2D(eyerect.center, eyerect.angle,1);
    cv::warpAffine(cornea, cornea, rotmat, cornea.size(), CV_INTER_CUBIC);
    cv::getRectSubPix(cornea, eyerect.size, eyerect.center, cornea);
    
    //fix to a standart size
	cv::Mat resultResized;
	float fix_val = (((float)kEyeWidth)/rect_eyeBorder.width);
	resultResized.create(round(fix_val*rect_eyeBorder.height),kEyeWidth, CV_8UC1);
	cv::resize(cornea, resultResized, resultResized.size(), 0, 0, cv::INTER_CUBIC);
	cornea = resultResized;
    
    cv::threshold(cornea, cornea, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);
    cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,7));
	cv::morphologyEx(cornea, cornea, CV_MOP_OPEN, element1);
	imshow("cornea", cornea);
	
	cv::Mat labels, stats, centroids;
	int ncomps = connectedComponentsWithStats(cornea, labels, stats, centroids, 8, CV_32S);

	if(ncomps == 2)
	{
		ftmp_set.n_comp = 1;
		ftmp_set.centroids = (int) centroids.at<double>(1,0);
		ftmp_set.area = stats.at<int>(1,cv::CC_STAT_AREA);
	}
	else if(ncomps > 2)
	{
		int l1area = stats.at<int>(1,cv::CC_STAT_AREA);
		int l2area = stats.at<int>(2,cv::CC_STAT_AREA);
		if(l1area > l2area)
		{
			ftmp_set.n_comp = 2;
			ftmp_set.centroids = (int) centroids.at<double>(1,0);
			ftmp_set.area = l1area;
		}
		else
		{
			ftmp_set.n_comp = ncomps - 1;
			ftmp_set.centroids = (int) centroids.at<double>(2,0);
			ftmp_set.area = l2area;
		}
	}
}

void gazeEstimate()
{
	int x_position;
	int y_position;
	
	cv::Mat responseX, responseY;
	cv::Mat sampleX, sampleY;
	sampleX.create(1,4,CV_32FC1);
	sampleY.create(1,2,CV_32FC1);
	
	sampleX.at<float>(0) = f_set.pupil_x;
	sampleX.at<float>(1) = f_set.centroids;
	sampleX.at<float>(2) = f_set.n_comp;
	sampleX.at<float>(3) = f_set.area;	
	
	sampleY.at<float>(0) = f_set.pupil_y;
	sampleY.at<float>(1) = f_set.vert_distanceOfeyeLids;
	
	cv::Point maxLocX, maxLocY, minLocY;
    double maxValX, maxValY;
	
	mlpX->predict(sampleX, responseX);
	cv::minMaxLoc(responseX, 0, &maxValX, 0, &maxLocX);
	x_position = maxLocX.x;
	cout<<endl<<"x direction,"<<x_position<<"nolu sınıf"<<endl;
	for(int i=0; i<responseX.cols; i++)
		cout<<"result "<<i<<":"<<responseX.at<float>(i)<<endl;
		
	mlpY->predict(sampleY, responseY);
	cv::minMaxLoc(responseY, 0, &maxValY, &minLocY, &maxLocY);
	y_position = maxLocY.x;
	cout<<endl<<"y direction,"<<y_position<<"nolu sınıf"<<endl;
	for(int i=0; i<responseY.cols; i++)
		cout<<"result "<<i<<":"<<responseY.at<float>(i)<<endl;
	
	if(elemination(x_position, y_position))		draw_gazePoint(x_position, y_position);
}

void draw_gazePoint(int x_position, int y_position)
{	
	cout<<"x position:"<<x_position<<" y position:"<<y_position<<endl;
	int index = y_position*3+x_position;
	cout<<"calculated index:"<<index<<endl;
	showCalibration(index);
}

void trainData()
{
	//cout<<"x:"<<x<<" y:"<<y<<endl;
	right_click = false;
	cv::Mat featureX = cv::Mat::zeros(1,4,CV_32FC1);
	cv::Mat featureY = cv::Mat::zeros(1,2,CV_32FC1);
	//classify x direction first
	featureX.at<float>(0) = f_set.pupil_x;
	featureX.at<float>(1) = f_set.centroids;
	featureX.at<float>(2) = f_set.n_comp;				
	featureX.at<float>(3) = f_set.area;				
	//classify y direction
	featureY.at<float>(0) = f_set.pupil_y;
	featureY.at<float>(1) = f_set.vert_distanceOfeyeLids;
	
	trainingFeaturesX.push_back(featureX);
	int x_index = (label_indexX%3);
	trainingLabelsX.push_back(x_index);
	trainingFeaturesY.push_back(featureY);
	label_indexY = label_indexX / 3;
	int y_index = (label_indexY%(2));
	trainingLabelsY.push_back(y_index);
	
	//cout<<"pupil x:"<<f_set.pupil_x<<" n_comp:"<<f_set.n_comp<<" centroids:"<<f_set.centroids<<" area:"<<f_set.area<<endl;
	//cout<<"pupil y:"<<f_set.pupil_y<<" vert_distance:"<<f_set.vert_distanceOfeyeLids<<endl;
	cout<<"left click x label:"<<x_index<<endl;
	cout<<"left click y label:"<<y_index<<endl;
}

bool elemination(int & x_position, int & y_position)
{
	if(bufferX.size() < buffer_size)
	{		
		bufferX.push_back(x_position);
		bufferY.push_back(y_position);
		return false;
	}
	else
	{
		int indexForX0 = 0, indexForX1 = 0, indexForX2 = 0, indexForX3 = 0,
		    indexForY0 = 0, indexForY1 = 0, indexForY2 = 0;
		
		bufferX.erase(bufferX.begin() + 0);
		bufferY.erase(bufferY.begin() + 0);
		bufferX.push_back(x_position);
		bufferY.push_back(y_position);
		
		for(int i=0; i<bufferX.size(); i++)
		{
			if(bufferX[i] == 0)				indexForX0++;
			else if(bufferX[i] == 1)		indexForX1++;
			else if(bufferX[i] == 2)		indexForX2++;
			
			if(bufferY[i] == 0)				indexForY0++;
			else if(bufferY[i] == 1)		indexForY1++;
			
		}
		
		if(indexForX0>indexForX1 && indexForX0>indexForX2)			x_position = 0;
		else if(indexForX1>indexForX0 && indexForX1>indexForX2)		x_position = 1;
		else if(indexForX2>indexForX0 && indexForX2>indexForX1)		x_position = 2;
		
		if(indexForY0>indexForY1)									y_position = 0;
		else if(indexForY1>indexForY0) 								y_position = 1;
		//bufferX.clear();
		//bufferY.clear();
		
		return true;
	}
}

void showCalibration(int index_val)
{
	calibScreen = cv::Mat::zeros(720,1280,CV_8UC3);
	
	int index = label_indexX % 6;
	cout<<"index:"<<index<<endl;
	std::vector<cv::Point> vec_point;
	vec_point.push_back(cv::Point(20,20));
	vec_point.push_back(cv::Point(640,20));
	vec_point.push_back(cv::Point(1260,20));
	vec_point.push_back(cv::Point(20,700));
	vec_point.push_back(cv::Point(640,700));
	vec_point.push_back(cv::Point(1260,700));
	
	char alphabet1[] = " abcçdefg";
	char alphabet1a[]= " ab";
	char alphabet1b[]= "cçd";
	char alphabet1c[]= "efg";
	char alphabet2[] = "ğhıijklmn";
	char alphabet2a[]= "ğhı";
	char alphabet2b[]= "ijk";
	char alphabet2c[]= "lmn";
	char alphabet3[] = "oöprsştuüvyz";
	char alphabet3a[]= "oöp";
	char alphabet3b[]= "rsş";
	char alphabet3c[]= "tuüvyz";
	char alphabet3c1[]= "tuü";
	char alphabet3c2[]= "vyz";
	
	string messages[6];
	messages[0] = '1';messages[1] = '2';messages[2] = '3';messages[3] = '4';messages[4] = message_text; messages[5] = '6';
	
	std::vector<cv::Point> vec_message;
	vec_message.push_back(cv::Point(0,90));
	vec_message.push_back(cv::Point(620,90));
	vec_message.push_back(cv::Point(1240,90));
	vec_message.push_back(cv::Point(0,670));
	vec_message.push_back(cv::Point(620,670));
	vec_message.push_back(cv::Point(1240,670));
	
	if(index_val >= 0)	
	{	
		index = index_val;
		if(prev_index !=-1)
		{
			if(index == 4)
			{
				message_text = messages[prev_index]; 
				messages[4] = message_text;
			}
		}
		prev_index = index;
	}
	for(int i=0; i<6; i++)
	{
		cv::circle(calibScreen, vec_point[i], 20, cv::Scalar( 0, 0, 255), 1);
		cv::putText(calibScreen, messages[i], vec_message[i], cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar( 0, 0, 255), 2);
		if(i==index)
			cv::circle(calibScreen, vec_point[i], 20, cv::Scalar( 0, 0, 255), -1);
	}

	cv::imshow("calibration",calibScreen);
}
