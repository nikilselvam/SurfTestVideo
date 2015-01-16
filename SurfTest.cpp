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
std::string targetImage3 = "legoGirl-3.jpg";
std::string targetImageBlue = "blueLegoCar.png";

int numberOfImages = 3;

//Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
Mat img_1 = imread(targetImage1, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_2 = imread(targetImage2, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_3 = imread(targetImage2, CV_LOAD_IMAGE_GRAYSCALE );

//Mat blueLegoCarImg = imread("blueLegoCar.png", CV_LOAD_IMAGE_GRAYSCALE);

int minHessian = 400;
SurfFeatureDetector detector(minHessian);
std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_3, keypoints_frame;

SurfDescriptorExtractor extractor;
Mat descriptors_1, descriptors_2, descriptors_3, descriptors_frame;
Mat img_keypoints_1, img_keypoints_2, img_keypoints_3, img_keypoints_frame;

// Set up matching
double max_dist = 0; double min_dist = 100;

// Data for frames
Mat frame;
Mat img_matches_1, img_matches_2, img_matches_3;

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
		case 3:
			img_to_process = img_3;
			keypoints_to_process = keypoints_3;
			descriptors_to_process = descriptors_3;
			img_keypoints_to_process = img_keypoints_3;
			string_to_show = "Keypoints 3";
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
		case 3:
			keypoints_3 = keypoints_to_process;
			descriptors_3 = descriptors_to_process;
			img_keypoints_3 = img_keypoints_to_process;
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
		case 3:
			descriptors_to_process = descriptors_3;
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


void drawGoodMatches(int img_number, std::vector<DMatch> good_matches) {
	Mat img_to_process;
	std::vector<KeyPoint> keypoints_to_process;
	std::string string_to_show;
	Mat img_matches;

	switch(img_number) {
		case 1:
			img_to_process = img_1;
			keypoints_to_process = keypoints_1;
			string_to_show = "Good Matches for Image 1";
			break;
		case 2:
			img_to_process = img_2;
			keypoints_to_process = keypoints_2;
			string_to_show = "Good Matches for Image 2";
			break;
		case 3:
			img_to_process = img_3;
			keypoints_to_process = keypoints_3;
			string_to_show = "Good Matches for Image 3";
			break;
		default:
			return;
	}

	// Draw good matches between frame and image 1.
	drawMatches( img_1, keypoints_1, frame, keypoints_frame, 
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	imshow(string_to_show, img_matches);
}

void printGoodMatches(int img_number, std::vector<DMatch> good_matches) {
	// Print out good matches between frame and image img_number.
	printf("max(2*min_dist, 0.02) is %f\n\n", max(2*min_dist, 0.02));
	for( int i = 0; i < (int)good_matches.size(); i++ )
	{ 
		printf( "-- Good Match %d[%d] Keypoint 1: %d  -- Keypoint 2: %d with distance %f \n", img_number, i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance );
	}
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
		case 3:
			detectImageKeypoints(1);
			detectImageKeypoints(2);
			detectImageKeypoints(3);
			break;
		default:
			break;
	}

	// ============== Capture Camera Frame ====================
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
			
			std::vector<DMatch> matches_1, matches_2, matches_3;
			std::vector<DMatch> good_matches_1, good_matches_2, good_matches_3;
			FlannBasedMatcher matcher;

			if (descriptors_frame.rows > 0) {

				// Find good matches.
				switch(numberOfImages) {
					case 1:
						good_matches_1 = findGoodMatches(1);
						break;
					case 2:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						break;
					case 3:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						good_matches_3 = findGoodMatches(3);
						break;
					default:
						break;
				}
			}

			// Draw good matches.
			switch(numberOfImages) {
				case 1:
					drawGoodMatches(1, good_matches_1);
					break;
				case 2:
					drawGoodMatches(1, good_matches_1);
					drawGoodMatches(2, good_matches_2);
					break;
				case 3:
					drawGoodMatches(1, good_matches_1);
					drawGoodMatches(2, good_matches_2);
					drawGoodMatches(3, good_matches_3);
					break;
				default:
					break;
			}

			// Print good matches.
			switch(numberOfImages) {
				case 1:
					printGoodMatches(1, good_matches_1);
					break;
				case 2:
					printGoodMatches(1, good_matches_1);
					printGoodMatches(2, good_matches_2);
					break;
				case 3:
					printGoodMatches(1, good_matches_1);
					printGoodMatches(2, good_matches_2);
					printGoodMatches(3, good_matches_3);
					break;
				default:
					break;
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