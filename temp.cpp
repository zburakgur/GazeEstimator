// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "findEyeCenter.h"
#include "constants.h"
#include "eyeRegion.h"
#include "findGazePoint.h"
#include <iostream>
#include <queue>
#include <stdio.h>
#include <fstream>
#include <math.h>

using namespace dlib;
using namespace std;


void show_pose_lines(std::vector<image_window::overlay_line> &lines, cv::Mat &img);
EyeRegion set_eyeRegion(cv::Mat &temp, std::vector<image_window::overlay_line> &eye_lines, int eye_id,
						std::vector<cv::Point> &eye_contour);
cv::Point2f find_EyeCenter(cv::Mat &temp, EyeRegion &eyeRegion, std::vector<cv::Point> &eye_contour);

cv::Point2f left_pupil;
cv::Point2f right_pupil;

cv::Mat trainingData;
std::vector<int> trainingLabels;
int label_index = 0;

cv::Mat right_eye_area;

/*
* mouse click listener of OpenCV
*/
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == cv::EVENT_LBUTTONDOWN )
     {/*
          cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
          cout<<"sol göz x:"<<left_pupil.x<<" y:"<<left_pupil.y<<" sağ göz x:"<<right_pupil.x<<" y:"<<right_pupil.y<<endl;
          ofstream datafile ("data.txt",ios::app);
          datafile<<left_pupil.x<<" "<<left_pupil.y<<"/"<<right_pupil.x<<" "<<right_pupil.y<<"/"<<x<<" "<<y<<endl;
          datafile.close();*/
          cv::Mat feature = cv::Mat::zeros(1,2,CV_32FC1);
          feature.at<float>(0) = right_pupil.x;
          feature.at<float>(1) = right_pupil.y;
          
          trainingData.push_back(feature);
          //trainingLabels.push_back((label_index%8));
          //label_index++;
          string img_name = "img/";
          img_name+=trainingData.rows;
          img_name+=".jpg";
          cv::imwrite(img_name, right_eye_area);
     }
     else if(event == cv::EVENT_RBUTTONDOWN)
     {
		 for(int i=0; i<6; i++)
			trainingLabels.push_back(i);
		cv::Mat classes;
		trainingData.convertTo(trainingData, CV_32FC1);
		cv::Mat(trainingLabels).copyTo(classes);
		cv::FileStorage fs("gazeLoc.xml", cv::FileStorage::WRITE);
		fs << "trainingData" << trainingData;
		fs << "classes" << classes;
		fs.release();
	 }
}

int main()
{
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
			cv::Mat temp;
            cap >> temp;
            cv::namedWindow("face",CV_WINDOW_NORMAL);
			cv::moveWindow("face", 400, 100);
            cv::setMouseCallback("face", CallBackFunc, NULL);
			
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            //for (unsigned long i = 0; i < faces.size(); ++i)
            //{
            if(faces.size() > 0)
            {
				full_object_detection face_pose = pose_model(cimg, faces[0]);
				shapes.push_back(face_pose);
			}
				
			//}

            // Display it all on the screen
            win.clear_overlay();
            std::vector<image_window::overlay_line> pose_lines = render_face_detections(shapes);
            //show_pose_lines(pose_lines, temp);
            
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
				std::vector<cv::Point> leye_contour;
				std::vector<cv::Point> reye_contour;
				EyeRegion leftEyeRegion = set_eyeRegion(temp, left_eye_lines,0, leye_contour);
				EyeRegion rightEyeRegion = set_eyeRegion(temp, right_eye_lines,1, reye_contour);
				left_pupil = find_EyeCenter(temp, rightEyeRegion, reye_contour);
				right_pupil = find_EyeCenter(temp, leftEyeRegion, leye_contour);
				cout<<"left x:"<<left_pupil.x<<" y:"<<left_pupil.y;
				cout<<" right x:"<<right_pupil.x<<" y:"<<right_pupil.y<<endl;
				cv::Point2f gaze_point = findGazePoint(right_pupil, left_pupil);
				circle(temp, gaze_point, 3, cv::Scalar( 0, 0, 255));
			}
            cv_image<bgr_pixel> cimg2(temp);
            
            win.set_image(cimg2);
            imshow("face",temp);
			cv::waitKey(30);
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

cv::Point2f find_EyeCenter(cv::Mat &temp, EyeRegion &eyeRegion, std::vector<cv::Point> &eye_contour)
{
	std::vector<cv::Mat> rgbChannels(3);
    cv::split(temp, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];
    
    cv::Mat eyeROI = frame_gray(eyeRegion.eyeRect);
    
    //blur(frame_gray, frame_gray, cv::Size(3,3));
    if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * eyeRegion.eyeRect.width;
		GaussianBlur( eyeROI, eyeROI, cv::Size( 0, 0 ), sigma);
	}
    equalizeHist(eyeROI, eyeROI);
    
    //Mat black(eyeROI.rows, eyeROI.cols, eyeROI.type(), cv::Scalar::all(0));
	//Mat mask(eyeROI.rows, eyeROI.cols, CV_8UC1, cv::Scalar(0));
    //drawContours( mask,eye_contour,0, Scalar(255),CV_FILLED, 8 );	
    
    right_eye_area = eyeROI;
    
    cv::Point pupil = findEyeCenter(eyeROI,eyeRegion);
    
    // change eye centers to global coordinates
	pupil.x += eyeRegion.eyeRect.x;
	pupil.y += eyeRegion.eyeRect.y;
    
    //cout<<"eye x="<<pupil.x<<"/y="<<pupil.y<<endl;
    
    circle(temp, pupil, 3, 1234);
    circle(temp, eyeRegion.eyeCorner, 3, 1234);
    
    line( temp, eyeRegion.eyeCorner, pupil, cv::Scalar( 255, 0, 255), 4 );
    
    cv::Point2f result;
    result.x = fabs(eyeRegion.eyeCorner.x - pupil.x);
    result.y = eyeRegion.eyeCorner.y - pupil.y;
    
	return result;
}

EyeRegion set_eyeRegion(cv::Mat &temp, std::vector<image_window::overlay_line> &eye_lines, int eye_id,
						std::vector<cv::Point> &eye_contour)
{
	dlib::point left_corner(10000, 0);
	dlib::point rigth_corner(0, 0);
	
	float top_eye = 10000;
	float bottom_eye = 0;
	
	for(int i=0; i<eye_lines.size(); i++)
	{
		image_window::overlay_line * line_ptr = & eye_lines.at(i);
		dlib::point * point1_ptr = &(line_ptr->p1);
		dlib::point * point2_ptr = &(line_ptr->p2);
		/*
		cv::Point point1( point1_ptr->x() , point1_ptr->y() );
		cv::Point point2( point2_ptr->x() , point2_ptr->y() );
		
		cv::LineIterator it(temp, point1, point2, 8);
		for(int j=0; j<it.count; i++, ++it)
			eye_contour.push_back(it.pos());*/
		
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
	
	//line( temp, cv::Point2f( left_corner_left_eye.x(), left_corner_left_eye.y()), 
	//	        cv::Point2f( rigth_corner_left_eye.x(), rigth_corner_left_eye.y()), cv::Scalar( 255, 0, 0), 4 );
	
	//line( temp, cv::Point2f( left_corner_left_eye.x(), top_left_eye), 
	//	        cv::Point2f( rigth_corner_left_eye.x(), top_left_eye), cv::Scalar( 255, 0, 0), 4 );
		        
	//line( temp, cv::Point2f( left_corner_left_eye.x(), bottom_eyebrow.y()), 
	//	        cv::Point2f( left_corner_left_eye.x(), top_left_eye), cv::Scalar( 0, 0, 255), 4 );
	
	cv::Rect rect_eyeBorder;
	rect_eyeBorder.x = left_corner.x();
	rect_eyeBorder.y = top_eye;
	rect_eyeBorder.width = rigth_corner.x() - left_corner.x();
	rect_eyeBorder.height = bottom_eye - top_eye;
	
	cv::Rect rect;
	rect.x = left_corner.x() - (rigth_corner.x() - left_corner.x())/2;
	rect.y = top_eye - 2*(bottom_eye - top_eye);
	rect.width = 2*(rigth_corner.x() - left_corner.x());
	rect.height = (bottom_eye - top_eye)*5;
		
	cv::rectangle( temp, cv::Point2f(rect.x, rect.y),
                       cv::Point2f(rect.x+rect.width, rect.y + rect.height), cv::Scalar( 0, 0, 255));	
	
	rect_eyeBorder.x -= rect.x;
	rect_eyeBorder.y -= rect.y;
	
	
	EyeRegion eyeRegion;
	eyeRegion.eyeRect = rect;
	if(eye_id == 0)//left eye
		eyeRegion.eyeCorner = cv::Point2f(rigth_corner.x(), rigth_corner.y());
	else//rigth eye
		eyeRegion.eyeCorner = cv::Point2f(left_corner.x(), left_corner.y());
		
	eyeRegion.eyeBorder = rect_eyeBorder;		
		
	return eyeRegion;
}

void show_pose_lines(std::vector<image_window::overlay_line> &lines, cv::Mat &img)
{
	for(int i=0; i<lines.size(); i++)
	{
		image_window::overlay_line * line_ptr = & lines.at(i);
		dlib::point * point1_ptr = &(line_ptr->p1);
		dlib::point * point2_ptr = &(line_ptr->p2);
		//point_ptr->x();
		//(line_ptr->p2).y();
		if(i < 39)
			line( img, cv::Point2f( point1_ptr->x(), point1_ptr->y()), cv::Point2f( point2_ptr->x(), point2_ptr->y()), cv::Scalar( 255, 0, 0), 4 );
		else
			line( img, cv::Point2f( point1_ptr->x(), point1_ptr->y()), cv::Point2f( point2_ptr->x(), point2_ptr->y()), cv::Scalar( 0, 0, 255), 4 );
	}
}
