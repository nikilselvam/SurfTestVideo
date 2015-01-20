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
std::string targetImage4 = "legoGirl-4.jpg";
std::string targetImage5 = "legoGirl-5.jpg";
std::string targetImage6 = "legoGirl-6.jpg";
std::string targetImage7 = "legoGirl-7.jpg";
std::string targetImage8 = "legoGirl-8.jpg";
std::string targetImage9 = "legoGirl-9.jpg";
std::string targetImageBlue = "blueLegoCar.png";

int numberOfImages = 9;

//Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
Mat img_1 = imread(targetImage1, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_2 = imread(targetImage2, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_3 = imread(targetImage3, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_4 = imread(targetImage4, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_5 = imread(targetImage5, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_6 = imread(targetImage6, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_7 = imread(targetImage7, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_8 = imread(targetImage8, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_9 = imread(targetImage9, CV_LOAD_IMAGE_GRAYSCALE );

//Mat blueLegoCarImg = imread("blueLegoCar.png", CV_LOAD_IMAGE_GRAYSCALE);

int minHessian = 400;
SurfFeatureDetector detector(minHessian);
std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_3, keypoints_4, keypoints_5, keypoints_6, keypoints_7, keypoints_8, keypoints_9, keypoints_frame;

SurfDescriptorExtractor extractor;
Mat descriptors_1, descriptors_2, descriptors_3, descriptors_4, descriptors_5, descriptors_6, descriptors_7, descriptors_8, descriptors_9, descriptors_frame;
Mat img_keypoints_1, img_keypoints_2, img_keypoints_3, img_keypoints_4, img_keypoints_5, img_keypoints_6, img_keypoints_7, img_keypoints_8, img_keypoints_9, img_keypoints_frame;

// Set up matching
double max_dist = 0; double min_dist = 100;

// Data for frames
Mat frame;
Mat img_matches_1, img_matches_2, img_matches_3, img_matches_4, img_matches_5, img_matches_6, img_matches_7, img_matches_8, img_matches_9;

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
		case 4:
			img_to_process = img_4;
			keypoints_to_process = keypoints_4;
			descriptors_to_process = descriptors_4;
			img_keypoints_to_process = img_keypoints_4;
			string_to_show = "Keypoints 4";
			break;
		case 5:
			img_to_process = img_5;
			keypoints_to_process = keypoints_5;
			descriptors_to_process = descriptors_5;
			img_keypoints_to_process = img_keypoints_5;
			string_to_show = "Keypoints 5";
			break;
		case 6:
			img_to_process = img_6;
			keypoints_to_process = keypoints_6;
			descriptors_to_process = descriptors_6;
			img_keypoints_to_process = img_keypoints_6;
			string_to_show = "Keypoints 6";
			break;
		case 7:
			img_to_process = img_7;
			keypoints_to_process = keypoints_7;
			descriptors_to_process = descriptors_7;
			img_keypoints_to_process = img_keypoints_7;
			string_to_show = "Keypoints 7";
			break;
		case 8:
			img_to_process = img_8;
			keypoints_to_process = keypoints_8;
			descriptors_to_process = descriptors_8;
			img_keypoints_to_process = img_keypoints_8;
			string_to_show = "Keypoints 8";
			break;
		case 9:
			img_to_process = img_9;
			keypoints_to_process = keypoints_9;
			descriptors_to_process = descriptors_9;
			img_keypoints_to_process = img_keypoints_9;
			string_to_show = "Keypoints 9";
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
		case 4:
			keypoints_4 = keypoints_to_process;
			descriptors_4 = descriptors_to_process;
			img_keypoints_4 = img_keypoints_to_process;
			break;
		case 5:
			keypoints_5 = keypoints_to_process;
			descriptors_5 = descriptors_to_process;
			img_keypoints_5 = img_keypoints_to_process;
			break;
		case 6:
			keypoints_6 = keypoints_to_process;
			descriptors_6 = descriptors_to_process;
			img_keypoints_6 = img_keypoints_to_process;
			break;
		case 7:
			keypoints_7 = keypoints_to_process;
			descriptors_7 = descriptors_to_process;
			img_keypoints_7 = img_keypoints_to_process;
			break;
		case 8:
			keypoints_8 = keypoints_to_process;
			descriptors_8 = descriptors_to_process;
			img_keypoints_8 = img_keypoints_to_process;
			break;
		case 9:
			keypoints_9 = keypoints_to_process;
			descriptors_9 = descriptors_to_process;
			img_keypoints_9 = img_keypoints_to_process;
			break;
		default:
			return;
	}
}

std::vector<DMatch> findGoodMatches(int img_number) {
	Mat descriptors_to_process;
	std::vector<DMatch> matches, good_matches;
	FlannBasedMatcher matcher;
	BFMatcher matcherSecond;

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
		case 4:
			descriptors_to_process = descriptors_4;
			break;
		case 5:
			descriptors_to_process = descriptors_5;
			break;
		case 6:
			descriptors_to_process = descriptors_6;
			break;
		case 7:
			descriptors_to_process = descriptors_7;
			break;
		case 8:
			descriptors_to_process = descriptors_8;
			break;
		case 9:
			descriptors_to_process = descriptors_9;
			break;
		default:
			return good_matches;
	}

	//matcher.match(descriptors_to_process, descriptors_frame, matches);
	//matcher.knnMatch(descriptors_to_process, descriptors_frame, matches, 5, storeKnnMatches);
	matcherSecond.match(descriptors_to_process, descriptors_frame, matches);

	// Reset the value of max_dist and min_dist for comparison's sake.
	max_dist = 0;
	min_dist = 100;

	// Determine good matches between frame and image i.
	for( int i = 0; i < descriptors_to_process.rows; i++ ) {
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	for( int i = 0; i < descriptors_to_process.rows; i++ ) {
		if( matches[i].distance <= 0.4 ) {
			good_matches.push_back(matches[i]);
		}
	}

	printf("Image %d\n", img_number);
	printf("Image descriptors: %d\n", descriptors_to_process.rows);
	printf("Frame descriptors: %d\n\n\n", descriptors_frame.rows);

	for (int i = 0; i < descriptors_to_process.rows; i++) {
		printf("matches[%d]:\t queryIdx = %d\t trainIdx= %d\t distance = %f\n", i, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance);
	}

	printf("\n\n");

	printf("Max value = %f\n\n", max(2*min_dist, 0.02));

	for (int i = 0; i < good_matches.size(); i++) {
		printf("good_matches[%d]:\t queryIdx = %d\t trainIdx= %d\t distance = %f\n", i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance);
	}
	
	printf("\n\n");
	
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
		case 4:
			img_to_process = img_4;
			keypoints_to_process = keypoints_4;
			string_to_show = "Good Matches for Image 4";
			break;
		case 5:
			img_to_process = img_5;
			keypoints_to_process = keypoints_5;
			string_to_show = "Good Matches for Image 5";
			break;
		case 6:
			img_to_process = img_6;
			keypoints_to_process = keypoints_6;
			string_to_show = "Good Matches for Image 6";
			break;
		case 7:
			img_to_process = img_7;
			keypoints_to_process = keypoints_7;
			string_to_show = "Good Matches for Image 7";
			break;
		case 8:
			img_to_process = img_8;
			keypoints_to_process = keypoints_8;
			string_to_show = "Good Matches for Image 8";
			break;
		case 9:
			img_to_process = img_9;
			keypoints_to_process = keypoints_9;
			string_to_show = "Good Matches for Image 9";
			break;
		default:
			return;
	}

	// Draw good matches between frame and image to show.
	drawMatches( img_to_process, keypoints_to_process, frame, keypoints_frame, 
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

void printTotalGoodMatches(vector <int> total_good_matches) {
	int total_matches_found = 0;
	
	for (int i = 0; i < total_good_matches.size(); i++) {
		printf("%d ", total_good_matches[i]);
		total_matches_found += total_good_matches[i];
	}

	printf("\t Total = %d", total_matches_found);
	printf("\n");
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
		case 4:
			detectImageKeypoints(1);
			detectImageKeypoints(2);
			detectImageKeypoints(3);
			detectImageKeypoints(4);
			break;
		case 5:
			detectImageKeypoints(1);
			detectImageKeypoints(2);
			detectImageKeypoints(3);
			detectImageKeypoints(4);
			detectImageKeypoints(5);
			break;
		case 6:
			detectImageKeypoints(1);
			detectImageKeypoints(2);
			detectImageKeypoints(3);
			detectImageKeypoints(4);
			detectImageKeypoints(5);
			detectImageKeypoints(6);
			break;
		case 7:
			detectImageKeypoints(1);
			detectImageKeypoints(2);
			detectImageKeypoints(3);
			detectImageKeypoints(4);
			detectImageKeypoints(5);
			detectImageKeypoints(6);
			detectImageKeypoints(7);
			break;
		case 8:
			detectImageKeypoints(1);
			detectImageKeypoints(2);
			detectImageKeypoints(3);
			detectImageKeypoints(4);
			detectImageKeypoints(5);
			detectImageKeypoints(6);
			detectImageKeypoints(7);
			detectImageKeypoints(8);
			break;
		case 9:
			detectImageKeypoints(1);
			detectImageKeypoints(2);
			detectImageKeypoints(3);
			detectImageKeypoints(4);
			detectImageKeypoints(5);
			detectImageKeypoints(6);
			detectImageKeypoints(7);
			detectImageKeypoints(8);
			detectImageKeypoints(9);
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
			drawKeypoints(frame, keypoints_frame, img_keypoints_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			imshow("Webcam Video", img_keypoints_frame);
			
			std::vector<DMatch> matches_1, matches_2, matches_3, matches_4, matches_5, matches_6, matches_7, matches_8, matches_9;
			std::vector<DMatch> good_matches_1, good_matches_2, good_matches_3, good_matches_4, good_matches_5, good_matches_6, good_matches_7, good_matches_8, good_matches_9;
			FlannBasedMatcher matcher;
			BFMatcher matcherSecond;

			vector <int> total_good_matches;

			if (descriptors_frame.rows > 0) {
				// Find good matches.
				switch(numberOfImages) {
					case 1:
						good_matches_1 = findGoodMatches(1);
						total_good_matches.push_back(good_matches_1.size());
						break;
					case 2:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());
						break;
					case 3:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						good_matches_3 = findGoodMatches(3);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());
						total_good_matches.push_back(good_matches_3.size());
						break;
					case 4:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						good_matches_3 = findGoodMatches(3);
						good_matches_4 = findGoodMatches(4);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());
						total_good_matches.push_back(good_matches_3.size());
						total_good_matches.push_back(good_matches_4.size());
						break;
					case 5:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						good_matches_3 = findGoodMatches(3);
						good_matches_4 = findGoodMatches(4);
						good_matches_5 = findGoodMatches(5);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());
						total_good_matches.push_back(good_matches_3.size());
						total_good_matches.push_back(good_matches_4.size());
						total_good_matches.push_back(good_matches_5.size());
						break;
					case 6:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						good_matches_3 = findGoodMatches(3);
						good_matches_4 = findGoodMatches(4);
						good_matches_5 = findGoodMatches(5);
						good_matches_6 = findGoodMatches(6);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());
						total_good_matches.push_back(good_matches_3.size());
						total_good_matches.push_back(good_matches_4.size());
						total_good_matches.push_back(good_matches_5.size());
						total_good_matches.push_back(good_matches_6.size());
						break;
					case 7:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						good_matches_3 = findGoodMatches(3);
						good_matches_4 = findGoodMatches(4);
						good_matches_5 = findGoodMatches(5);
						good_matches_6 = findGoodMatches(6);
						good_matches_7 = findGoodMatches(7);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());
						total_good_matches.push_back(good_matches_3.size());
						total_good_matches.push_back(good_matches_4.size());
						total_good_matches.push_back(good_matches_5.size());
						total_good_matches.push_back(good_matches_6.size());
						total_good_matches.push_back(good_matches_7.size());
						break;
					case 8:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						good_matches_3 = findGoodMatches(3);
						good_matches_4 = findGoodMatches(4);
						good_matches_5 = findGoodMatches(5);
						good_matches_6 = findGoodMatches(6);
						good_matches_7 = findGoodMatches(7);
						good_matches_8 = findGoodMatches(8);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());
						total_good_matches.push_back(good_matches_3.size());
						total_good_matches.push_back(good_matches_4.size());
						total_good_matches.push_back(good_matches_5.size());
						total_good_matches.push_back(good_matches_6.size());
						total_good_matches.push_back(good_matches_7.size());
						total_good_matches.push_back(good_matches_8.size());
						break;
					case 9:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						good_matches_3 = findGoodMatches(3);
						good_matches_4 = findGoodMatches(4);
						good_matches_5 = findGoodMatches(5);
						good_matches_6 = findGoodMatches(6);
						good_matches_7 = findGoodMatches(7);
						good_matches_8 = findGoodMatches(8);
						good_matches_9 = findGoodMatches(9);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());
						total_good_matches.push_back(good_matches_3.size());
						total_good_matches.push_back(good_matches_4.size());
						total_good_matches.push_back(good_matches_5.size());
						total_good_matches.push_back(good_matches_6.size());
						total_good_matches.push_back(good_matches_7.size());
						total_good_matches.push_back(good_matches_8.size());
						total_good_matches.push_back(good_matches_9.size());
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
				case 4:
					drawGoodMatches(1, good_matches_1);
					drawGoodMatches(2, good_matches_2);
					drawGoodMatches(3, good_matches_3);
					drawGoodMatches(4, good_matches_4);
					break;
				case 5:
					drawGoodMatches(1, good_matches_1);
					drawGoodMatches(2, good_matches_2);
					drawGoodMatches(3, good_matches_3);
					drawGoodMatches(4, good_matches_4);
					drawGoodMatches(5, good_matches_5);
					break;
				case 6:
					drawGoodMatches(1, good_matches_1);
					drawGoodMatches(2, good_matches_2);
					drawGoodMatches(3, good_matches_3);
					drawGoodMatches(4, good_matches_4);
					drawGoodMatches(5, good_matches_5);
					drawGoodMatches(6, good_matches_6);
					break;
				case 7:
					drawGoodMatches(1, good_matches_1);
					drawGoodMatches(2, good_matches_2);
					drawGoodMatches(3, good_matches_3);
					drawGoodMatches(4, good_matches_4);
					drawGoodMatches(5, good_matches_5);
					drawGoodMatches(6, good_matches_6);
					drawGoodMatches(7, good_matches_7);
					break;
				case 8:
					drawGoodMatches(1, good_matches_1);
					drawGoodMatches(2, good_matches_2);
					drawGoodMatches(3, good_matches_3);
					drawGoodMatches(4, good_matches_4);
					drawGoodMatches(5, good_matches_5);
					drawGoodMatches(6, good_matches_6);
					drawGoodMatches(7, good_matches_7);
					drawGoodMatches(8, good_matches_8);
					break;
				case 9:
					drawGoodMatches(1, good_matches_1);
					drawGoodMatches(2, good_matches_2);
					drawGoodMatches(3, good_matches_3);
					drawGoodMatches(4, good_matches_4);
					drawGoodMatches(5, good_matches_5);
					drawGoodMatches(6, good_matches_6);
					drawGoodMatches(7, good_matches_7);
					drawGoodMatches(8, good_matches_8);
					drawGoodMatches(9, good_matches_9);
					break;
				default:
					break;
			}


			// Print total good matches.
			printTotalGoodMatches(total_good_matches);

			/*
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
				case 4:
					printGoodMatches(1, good_matches_1);
					printGoodMatches(2, good_matches_2);
					printGoodMatches(3, good_matches_3);
					printGoodMatches(4, good_matches_4);
					break;
				case 5:
					printGoodMatches(1, good_matches_1);
					printGoodMatches(2, good_matches_2);
					printGoodMatches(3, good_matches_3);
					printGoodMatches(4, good_matches_4);
					printGoodMatches(5, good_matches_5);
					break;
				case 6:
					printGoodMatches(1, good_matches_1);
					printGoodMatches(2, good_matches_2);
					printGoodMatches(3, good_matches_3);
					printGoodMatches(4, good_matches_4);
					printGoodMatches(5, good_matches_5);
					printGoodMatches(6, good_matches_6);
					break;
				case 7:
					printGoodMatches(1, good_matches_1);
					printGoodMatches(2, good_matches_2);
					printGoodMatches(3, good_matches_3);
					printGoodMatches(4, good_matches_4);
					printGoodMatches(5, good_matches_5);
					printGoodMatches(6, good_matches_6);
					printGoodMatches(7, good_matches_7);
					break;
				case 8:
					printGoodMatches(1, good_matches_1);
					printGoodMatches(2, good_matches_2);
					printGoodMatches(3, good_matches_3);
					printGoodMatches(4, good_matches_4);
					printGoodMatches(5, good_matches_5);
					printGoodMatches(6, good_matches_6);
					printGoodMatches(7, good_matches_7);
					printGoodMatches(8, good_matches_8);
					break;
				case 9:
					printGoodMatches(1, good_matches_1);
					printGoodMatches(2, good_matches_2);
					printGoodMatches(3, good_matches_3);
					printGoodMatches(4, good_matches_4);
					printGoodMatches(5, good_matches_5);
					printGoodMatches(6, good_matches_6);
					printGoodMatches(7, good_matches_7);
					printGoodMatches(8, good_matches_8);
					printGoodMatches(9, good_matches_9);
					break;
				default:
					break;
			}
			*/

			if(waitKey(30) >= 0) break;
		}
	}
	return 0;
}

/** @function readme */
void readme() {
	std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl;
}