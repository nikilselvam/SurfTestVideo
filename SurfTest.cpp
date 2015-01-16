#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

// Constant to control webcam device number
const int WEBCAM_DEVICE_NUMBER = 0;

// Declare webcamCapture object associated with selected webcam device
VideoCapture webcamCapture(WEBCAM_DEVICE_NUMBER);

void readme();

std::string targetImage = "testLegoGirl.jpg";
std::string targetImage1 = "legoGirl-1.jpg";
std::string targetImage2 = "legoGirl-2.jpg";
std::string targetImageBlue = "blueLegoCar.png";

int numberOfImages = 2;

//Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
Mat img_1 = imread(targetImage1, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_2 = imread(targetImage2, CV_LOAD_IMAGE_GRAYSCALE );

//Mat blueLegoCarImg = imread("blueLegoCar.png", CV_LOAD_IMAGE_GRAYSCALE);

int minHessian = 400;
SurfFeatureDetector detector(minHessian);
std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_frame;

SurfDescriptorExtractor extractor;
Mat descriptors_1, descriptors_2, descriptors_frame;
Mat img_keypoints_1, img_keypoints_2, img_keypoints_frame;

// Set up matching
double max_dist = 0; double min_dist = 100;

void detectImageKeypoints(int img_number) {
	Mat img_to_process;
	std::vector<KeyPoint> keypoints_to_process;
	Mat descriptors_to_process;
	Mat img_keypoints_to_process;
	std::string string_to_show;

	switch(img_number) {
		case 1:
			img_to_process = img_1;
			keypoints_to_process = keypoints_1;
			descriptors_to_process = descriptors_1;
			img_keypoints_to_process = img_keypoints_1;
			string_to_show = "Keypoints 1";
			break;
		case 2:
			img_to_process = img_2;
			keypoints_to_process = keypoints_2;
			descriptors_to_process = descriptors_2;
			img_keypoints_to_process = img_keypoints_2;
			string_to_show = "Keypoints 2";
			break;
		default:
			return;
	}

	detector.detect( img_to_process, keypoints_to_process );
	extractor.compute( img_to_process, keypoints_to_process, descriptors_to_process );
	drawKeypoints( img_to_process, keypoints_to_process, img_keypoints_to_process, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	imshow(string_to_show, img_keypoints_to_process );

	switch(img_number) {
		case 1:
			keypoints_1 = keypoints_to_process;
			descriptors_1 = descriptors_to_process;
			img_keypoints_1 = img_keypoints_to_process;
			break;
		case 2:
			keypoints_2 = keypoints_to_process;
			descriptors_2 = descriptors_to_process;
			img_keypoints_2 = img_keypoints_to_process;
			break;
		default:
			return;
	}
}

std::vector<DMatch> findGoodMatches(int img_number) {
	Mat descriptors_to_process;
	std::vector<DMatch> matches, good_matches;
	FlannBasedMatcher matcher;

	switch(img_number) {
		case 1:
			descriptors_to_process = descriptors_1;
			break;
		case 2:
			descriptors_to_process = descriptors_2;
			break;
		default:
			return good_matches;
	}

	matcher.match(descriptors_to_process, descriptors_frame, matches);

	// Determine good matches between frame and image 1.
	for( int i = 0; i < descriptors_1.rows; i++ ) {
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	for( int i = 0; i < descriptors_1.rows; i++ ) {
		if( matches[i].distance <= max(2*min_dist, 0.02) ) {
			good_matches.push_back(matches[i]);
		}
	}

	return good_matches;
}
/** @function main */
int main( int argc, char** argv )
{
	// ============== Lego Girl ===============================

	switch(numberOfImages) {
		case 1:
			detectImageKeypoints(1);
			break;
		case 2:
			detectImageKeypoints(1);
			detectImageKeypoints(2);
			break;
		default:
			break;
	}

	/*
	if (numberOfImages >= 1) {
		detectImages(1);
	}
	
	if (numberOfImages >= 2) {
		detectImages(2);
	}
	*/

	// ============== Capture Camera Frame ====================
	Mat frame;
	Mat img_matches_1, img_matches_2;
	// Capturing data from camera
		// Caution: It requires webcam to be attached
	VideoCapture cap(0);
		// you can also use
		// cv::VideoCapture cap("video filename");
		// to capture the frame from a video instead of webcam
	if(!cap.isOpened())
		printf("No Camera Detected");
	else {
		namedWindow("Webcam Video");
		while(true) {
			cap >> frame; // get a new frame from camera

			detector.detect(frame, keypoints_frame);
			extractor.compute( frame, keypoints_frame, descriptors_frame);
			
			std::vector<DMatch> matches_1, matches_2;
			std::vector<DMatch> good_matches_1, good_matches_2;
			FlannBasedMatcher matcher;

			if (descriptors_frame.rows > 0) {

				switch(numberOfImages) {
					case 1:
						good_matches_1 = findGoodMatches(1);
						break;
					case 2:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						break;
					default:
						break;
				}

				/*
				if (numberOfImages >= 1) {
					matcher.match(descriptors_1, descriptors_frame, matches_1);

					// Determine good matches between frame and image 1.
					for( int i = 0; i < descriptors_1.rows; i++ ) {
						double dist = matches_1[i].distance;
						if( dist < min_dist ) min_dist = dist;
						if( dist > max_dist ) max_dist = dist;
					}

					for( int i = 0; i < descriptors_1.rows; i++ ) {
						if( matches_1[i].distance <= max(2*min_dist, 0.02) ) {
							good_matches_1.push_back(matches_1[i]);
						}
					}
				}

				if (numberOfImages >= 2) {
					matcher.match(descriptors_1, descriptors_frame, matches_2);

					// Determine good matches between frame and image 2.
					for( int i = 0; i < descriptors_2.rows; i++ ) {
						double dist = matches_2[i].distance;
						if( dist < min_dist ) min_dist = dist;
						if( dist > max_dist ) max_dist = dist;
					}

					for( int i = 0; i < descriptors_2.rows; i++ ) {
						if( matches_2[i].distance <= max(2*min_dist, 0.02) ) {
							good_matches_2.push_back(matches_2[i]);
						}
					}
				}
				*/
			}

			if (numberOfImages >= 1) {
				// Draw good matches between frame and image 1.
				drawMatches( img_1, keypoints_1, frame, keypoints_frame, 
					good_matches_1, img_matches_1, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
				imshow("Good Matches for Image 1", img_matches_1);


				// Print out good matches between frame and image 1.
				printf("max(2*min_dist, 0.02) is %f\n\n", max(2*min_dist, 0.02));
				for( int i = 0; i < (int)good_matches_1.size(); i++ )
				{ 
					printf( "-- Good Match 1[%d] Keypoint 1: %d  -- Keypoint 2: %d with distance %f \n", i, good_matches_1[i].queryIdx, good_matches_1[i].trainIdx, good_matches_1[i].distance );
				}			
			}

			if (numberOfImages >= 2) {
				// Draw good matches between frame and image 2.
				drawMatches( img_2, keypoints_2, frame, keypoints_frame, 
					good_matches_2, img_matches_2, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
				imshow("Good Matches for Image 2", img_matches_2);


				// Print out good matches between frame and image 2.
				printf("max(2*min_dist, 0.02) is %f\n\n", max(2*min_dist, 0.02));
				for( int i = 0; i < (int)good_matches_2.size(); i++ )
				{ 
					printf( "-- Good Match 2[%d] Keypoint 1: %d  -- Keypoint 2: %d with distance %f \n", i, good_matches_2[i].queryIdx, good_matches_2[i].trainIdx, good_matches_2[i].distance );
				}
			}

			//drawKeypoints(frame, keypoints_frame, img_keypoints_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

			//imshow("Webcam Video", img_keypoints_frame);
			if(waitKey(30) >= 0) break;
		}
	}
	return 0;
}

/** @function readme */
void readme() {
	std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl;
}