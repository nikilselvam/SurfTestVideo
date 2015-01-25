#include <stdio.h>
#include <iostream>
#include <fstream>
#include "DetectionCoordinates.h"
#include "FeaturefulObject.h"
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

/*
std::string targetImage1 = "Picture 9.jpg";
std::string targetImage2 = "Picture 10.jpg";
std::string targetImage3 = "Picture 11.jpg";
std::string targetImage4 = "Picture 12.jpg";
std::string targetImage5 = "Picture 13.jpg";
std::string targetImage6 = "Picture 14.jpg";
std::string targetImage7 = "Picture 15.jpg";
std::string targetImage8 = "Picture 16.jpg";
std::string targetImage9 = "Picture 17.jpg";

Mat img_1 = imread(targetImage1, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_2 = imread(targetImage2, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_3 = imread(targetImage3, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_4 = imread(targetImage4, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_5 = imread(targetImage5, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_6 = imread(targetImage6, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_7 = imread(targetImage7, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_8 = imread(targetImage8, CV_LOAD_IMAGE_GRAYSCALE );
Mat img_9 = imread(targetImage9, CV_LOAD_IMAGE_GRAYSCALE );
*/


std::string targetImage1;
std::string targetImage2;
std::string targetImage3;
std::string targetImage4;
std::string targetImage5;
std::string targetImage6;
std::string targetImage7;
std::string targetImage8;
std::string targetImage9;

Mat img_1;
Mat img_2;
Mat img_3;
Mat img_4;
Mat img_5;
Mat img_6;
Mat img_7;
Mat img_8;
Mat img_9;

float distance_threshold = 0;
int matches_threshold = 0;

std::string targetImageBlue = "blueLegoCar.png";

int numberOfImages;
std::string name;

vector <FeaturefulObject> featureful_objects;

int minHessian = 400;
SurfFeatureDetector detector(minHessian);
std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_3, keypoints_4, keypoints_5, keypoints_6, keypoints_7, keypoints_8, keypoints_9, keypoints_frame;
vector <DetectionCoordinates> coordinates_vector;

SurfDescriptorExtractor extractor;
Mat descriptors_1, descriptors_2, descriptors_3, descriptors_4, descriptors_5, descriptors_6, descriptors_7, descriptors_8, descriptors_9, descriptors_frame;
Mat img_keypoints_1, img_keypoints_2, img_keypoints_3, img_keypoints_4, img_keypoints_5, img_keypoints_6, img_keypoints_7, img_keypoints_8, img_keypoints_9, img_keypoints_frame;
std::vector<DMatch> good_matches_1, good_matches_2, good_matches_3, good_matches_4, good_matches_5, good_matches_6, good_matches_7, good_matches_8, good_matches_9, all_good_matches, unique_matches;
std::vector<Point2f> match_coordinates;

// Set up matching
double max_dist = 0; double min_dist = 100;
float min_x = 100000; float max_x = 0;
float min_y = 100000; float max_y = 0;
int min_x_index = -1; int max_x_index = -1;
int min_y_index = -1; int max_y_index = -1;
float x_range = 400; float y_range = 400;

// Set up tracking of (x,y) coordinate changes with legoGirl's location.
float previous_x = 0; float previous_y = 0;
float x_diff = 0; float y_diff = 0;

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
	//drawKeypoints( img_to_process, keypoints_to_process, img_keypoints_to_process, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	//imshow(string_to_show, img_keypoints_to_process );

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

	// Determine good matches between frame and current image.
	for( int i = 0; i < descriptors_to_process.rows; i++ ) {
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	for( int i = 0; i < descriptors_to_process.rows; i++ ) {
		if( matches[i].distance <= distance_threshold ) {
			// Set matches imgIdx to appropriate img_number and then push match to good_matches.
			matches[i].imgIdx = img_number;

			good_matches.push_back(matches[i]);
		}
	}

	/*
	// Print information about size of descriptors vector as well as the contents of the matches and good_matches vectors.
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
	*/

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

void showWebcamKeypointMatches(int img_number, std::vector<DMatch> good_matches) {
	std::vector<KeyPoint> matched_keypoints;
	std::string string_to_show;

	switch(img_number) {
		case 1:
			string_to_show = "Matched Keypoints 1";
			break;
		case 2:
			string_to_show = "Matched Keypoints 2";
			break;
		case 3:
			string_to_show = "Matched Keypoints 3";
			break;
		case 4:
			string_to_show = "Matched Keypoints 4";
			break;
		case 5:
			string_to_show = "Matched Keypoints 5";
			break;
		case 6:
			string_to_show = "Matched Keypoints 6";
			break;
		case 7:
			string_to_show = "Matched Keypoints 7";
			break;
		case 8:
			string_to_show = "Matched Keypoints 8";
			break;
		case 9:
			string_to_show = "Matched Keypoints 9";
			break;
		default:
			return;
	}

	for (int i = 0; i < good_matches.size(); i++) {
		matched_keypoints.push_back(keypoints_frame[good_matches[i].trainIdx]);
	}
	drawKeypoints(frame, matched_keypoints, img_keypoints_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	imshow(string_to_show, img_keypoints_frame);
}

void showAllWebcamKeypointMatches() {
	std::vector<KeyPoint> all_matched_keypoints;

	for (int i = 0; i < all_good_matches.size(); i++) {
		all_matched_keypoints.push_back(keypoints_frame[all_good_matches[i].trainIdx]);
	}

	drawKeypoints(frame, all_matched_keypoints, img_keypoints_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("All Matched Keypoints", img_keypoints_frame);
}

void findUniqueMatches() {
	int trainIdx = -1;
	bool matchFound = false;

	// Clear unique_matches vector.
	unique_matches.clear();

	for (int i = 0; i < all_good_matches.size(); i++) {
		trainIdx = all_good_matches[i].trainIdx;
		matchFound = false;

		// Search through unique_matches and see if a match with the same trainIdx already exist.
		// If so, this means that we already have the corresponding keypoint in our match_coordinates vector and 
		// we dont' need to add it again. Set matchFound to true.
		// If not, add the match to unique_matches and push the (x,y) information to match_coordinates.
		for (int j = 0; j < unique_matches.size(); j++) {
			if (unique_matches[j].trainIdx == trainIdx) {
				matchFound = true;
				break;
			}
		}

		if (!matchFound) {
			unique_matches.push_back(all_good_matches[i]);
		}
	}
}

void printMatchesVector(std::string name, vector<DMatch> vector, bool printDist) {
	std::cout << "Printing vector " << name << "\t size = " << vector.size() << std::endl << std::endl;

	for (int i = 0; i < vector.size(); i++) {

		std::cout << name << "[" << i << "]:\t " << "queryIdx = " << vector[i].queryIdx << "\t trainIdx = " << vector[i].trainIdx << "\t imgIdx = " << vector[i].imgIdx;

		if (printDist) {
			std::cout << "\t distance = " << vector[i].distance;
		}

		std::cout << std::endl;
	}

	printf("\n\n");
}

void printPoint2fVector(std::string name, vector<Point2f> vector) {
	std::cout << "Printing vector " << name << "\t size = " << vector.size() << std::endl << std::endl;

	for (int i = 0; i < vector.size(); i++) {
		std::cout << name << "[" << i << "]:\t " << "x = " << vector[i].x << "\t y = " << vector[i].y << std::endl;
	}

	printf("\n\n");
}

void printKeypointsVector(std::string name, vector<KeyPoint> vector) {
	std::cout << "Printing vector " << name << "\t size = " << vector.size() << std::endl << std::endl;

	for (int i = 0; i < vector.size(); i++) {
		std::cout << name << "[" << i << "]:\t " << "x = " << vector[i].pt << std::endl;
	}

	printf("\n\n");
}

void printDetectionCoordinatesVector(std::string name, vector<DetectionCoordinates> vector) {
	std::cout << "Printing vector " << name << "\t size = " << vector.size() << std::endl << std::endl;

	for (int i = 0; i < vector.size(); i++) {
		std::cout << name << "[" << i << "]:\t " << "x = " << vector[i].get_x() << "\t y = " << vector[i].get_y() << "\t total = " << vector[i].get_total() << std::endl;
	}

	printf("\n\n");
}

void printFeaturefulObjectsVector(std::string name, vector<FeaturefulObject> vector) {
	std::cout << "Printing vector " << name << "\t size = " << vector.size() << std::endl << std::endl;

	for (int i = 0; i < vector.size(); i++) {
		std::cout << name << "[" << i << "]:\t " << "name = " << vector[i].get_name() << "\t image # = " << vector[i].get_numImages() << std::endl;
		std::cout << "distance_threshold = " << vector[i].get_distanceThreshold() << " \t matches_threshold = " << vector[i].get_matchesThreshold() << std::endl;

		std::vector <std::string> targetImagesToPrint = vector[i].get_images();

		for (int j = 0; j < targetImagesToPrint.size(); j++) {
			std::cout << "targetImage[" << j << "] = " << targetImagesToPrint[j] << "\t";

			//  Print a newline after every 3 target images printed.
			if ( (j+1) % 2 == 0) {
				std::cout << std::endl;
			}
		}

		printf("\n\n");
	}

	printf("\n\n\n\n");
}

vector <DetectionCoordinates> sortCoordinates(vector <DetectionCoordinates> vector_to_sort) {
	std::sort(vector_to_sort.begin(), vector_to_sort.end());
	return vector_to_sort;
}

Point2f medianDetection() {
	Point2f lego_girl_coordinates;
	
	lego_girl_coordinates.x = 0;
	lego_girl_coordinates.y = 0;

	int trainIdx = -1;
	float x_val = -1; float y_val = -1;

	// Take all the good matches and create Detection Coordinates points of them. Push them into vector.
	for (int i = 0; i < all_good_matches.size(); i++) {
		trainIdx = all_good_matches[i].trainIdx;
		x_val = keypoints_frame[trainIdx].pt.x;
		y_val = keypoints_frame[trainIdx].pt.y;

		DetectionCoordinates coordinate = DetectionCoordinates(x_val, y_val);
		coordinates_vector.push_back(coordinate);
	}

	// Sort the coordinates.
	std::sort(coordinates_vector.begin(), coordinates_vector.end());

	// Print the coordinates.
	//printDetectionCoordinatesVector("coordinates_vector", coordinates_vector);

	// Calculate median index.
	int size = coordinates_vector.size();

	if (size % 2 == 1) {
		int median_index = size / 2;

		//printf("size = %d \t median index = %d\n", size, median_index);

		// Set coordinates.
		lego_girl_coordinates.x = coordinates_vector[median_index].get_x();
		lego_girl_coordinates.y = coordinates_vector[median_index].get_y();
	} else {
		int left_median_index = size / 2 - 1;
		int right_median_index = size / 2;

		//printf("size = %d \nleft median index = %d \t right median index = %d\n", size, left_median_index, right_median_index);

		float left_median_x = coordinates_vector[left_median_index].get_x();
		float left_median_y = coordinates_vector[left_median_index].get_y();

		//printf("left_median_x = %f \t left_median_y = %f\n", left_median_x, left_median_y);

		float right_median_x = coordinates_vector[right_median_index].get_x();
		float right_median_y = coordinates_vector[right_median_index].get_y();

		//printf("right_median_y = %f \t right_median_y = %f\n",right_median_x, right_median_y);

		lego_girl_coordinates.x = (left_median_x + right_median_x) / 2;
		lego_girl_coordinates.y = (left_median_y + right_median_y) / 2;
	}

	//printf("lego_girl_coorindates.x = %f \t lego_girl_coordinates.y = %f\n\n\n", lego_girl_coordinates.x, lego_girl_coordinates.y);

	return lego_girl_coordinates;
}

Point2f findCoordinates() {
	std::vector<KeyPoint> all_matched_keypoints;
	std::vector <KeyPoint> keypoints_to_process;
	int img_number, queryIdx;

	//printMatchesVector("all_good_matches", all_good_matches, true);
	//printMatchesVector("unique_matches", unique_matches, true);

	// Get match_coordinates of unique matches.
	for (int i = 0; i < unique_matches.size(); i++) {
		match_coordinates.push_back(keypoints_frame[unique_matches[i].trainIdx].pt);		
	}

	//printKeypointsVector("keypoints_frame", keypoints_frame);
	//printPoint2fVector("match_coordinates", match_coordinates);

	//printf("all_good_matches.size = %d, unique_matches.size = %d\n", all_good_matches.size(), unique_matches.size());

	return medianDetection();
}

void readInSingleObjectDetectionFile() {
	std::string line;
	std::fstream detection_file ("legoGirl.txt");

	if (detection_file.is_open()) {
		// Read in name.
		getline(detection_file, line);
		name = line;

		// Read in number of images.
		getline(detection_file, line);
		numberOfImages = atoi(line.c_str());

		// Read in names of the images to be processed.
		for (int i = 0; i < numberOfImages; i++) {
			getline(detection_file, line);

			switch(i) {
				case 0:
					targetImage1 = line;
					img_1 = imread(targetImage1, CV_LOAD_IMAGE_GRAYSCALE );
					break;
				case 1:
					targetImage2 = line;
					img_2 = imread(targetImage2, CV_LOAD_IMAGE_GRAYSCALE );
					break;
				case 2:
					targetImage3 = line;
					img_3 = imread(targetImage3, CV_LOAD_IMAGE_GRAYSCALE );
					break;
				case 3:
					targetImage4 = line;
					img_4 = imread(targetImage4, CV_LOAD_IMAGE_GRAYSCALE );
					break;
				case 4:
					targetImage5 = line;
					img_5 = imread(targetImage5, CV_LOAD_IMAGE_GRAYSCALE );
					break;
				case 5:
					targetImage6 = line;
					img_6 = imread(targetImage6, CV_LOAD_IMAGE_GRAYSCALE );
					break;
				case 6:
					targetImage7 = line;
					img_7 = imread(targetImage7, CV_LOAD_IMAGE_GRAYSCALE );
					break;
				case 7:
					targetImage8 = line;
					img_8 = imread(targetImage8, CV_LOAD_IMAGE_GRAYSCALE );
					break;
				case 8:
					targetImage9 = line;
					img_9 = imread(targetImage9, CV_LOAD_IMAGE_GRAYSCALE );
					break;
				default:
					break;
			}
		}

		// Read in distance threshold.
		getline(detection_file, line);
		distance_threshold = atof(line.c_str());

		// Read in matches threshold.
		getline(detection_file, line);
		matches_threshold = atoi(line.c_str());
	}

	detection_file.close();
}

void readInMultipleObjectDetectionFile() {
	std::string line;
	std::fstream detection_file ("objectDetection.txt");
	vector <string> targetImages;

	int numberOfObjects = 0;

	if (detection_file.is_open()) {
		// Read in number of objects.
		getline(detection_file, line);
		numberOfObjects = atoi(line.c_str());

		for (int i = 0; i < numberOfObjects; i++) {
			// Burn a line.
			getline(detection_file, line);

			// Read in name.
			getline(detection_file, line);
			name = line;

			// Read in number of images.
			getline(detection_file, line);
			numberOfImages = atoi(line.c_str());

			// Read in names of the images to be processed.
			for (int j = 0; j < numberOfImages; j++) {
				getline(detection_file, line);
				targetImages.push_back(line);

				switch(j) {
					case 0:
						img_1 = imread(targetImage1, CV_LOAD_IMAGE_GRAYSCALE );
						break;
					case 1:
						img_2 = imread(targetImage2, CV_LOAD_IMAGE_GRAYSCALE );
						break;
					case 2:
						img_3 = imread(targetImage3, CV_LOAD_IMAGE_GRAYSCALE );
						break;
					case 3:
						img_4 = imread(targetImage4, CV_LOAD_IMAGE_GRAYSCALE );
						break;
					case 4:
						img_5 = imread(targetImage5, CV_LOAD_IMAGE_GRAYSCALE );
						break;
					case 5:
						img_6 = imread(targetImage6, CV_LOAD_IMAGE_GRAYSCALE );
						break;
					case 6:
						img_7 = imread(targetImage7, CV_LOAD_IMAGE_GRAYSCALE );
						break;
					case 7:
						img_8 = imread(targetImage8, CV_LOAD_IMAGE_GRAYSCALE );
						break;
					case 8:
						img_9 = imread(targetImage9, CV_LOAD_IMAGE_GRAYSCALE );
						break;
					default:
						break;
				}
			}

			// Read in distance threshold.
			getline(detection_file, line);
			distance_threshold = atof(line.c_str());

			// Read in matches threshold.
			getline(detection_file, line);
			matches_threshold = atoi(line.c_str());

			// Create featurefulObject and push object into vector.
			FeaturefulObject featurefulObjectToStore = FeaturefulObject(name, numberOfImages, targetImages, distance_threshold, matches_threshold);
			featureful_objects.push_back(featurefulObjectToStore);

			// Clear targetImages vector for next object that is processed.
			targetImages.clear();
		}
	}

	detection_file.close();
}

/** @function main */
int main( int argc, char** argv )
{
	// Read in detection file parameters.
	readInSingleObjectDetectionFile();

	// readInMultipleObjectDetectionFile();
	//printFeaturefulObjectsVector("featureful_objects", featureful_objects);

	std::cout << "name = " << name << std::endl;
	printf("numberOfImages = %d\n", numberOfImages);
	printf("distance_threshold = %f\n", distance_threshold);
	printf("matches_threshold = %d\n\n\n", matches_threshold);

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
			
			/*
			std::vector<KeyPoint> firstTenKeyPoints;

			for (int i = 0; i < 10; i++) {
				firstTenKeyPoints.push_back(keypoints_frame[i]);
			}
			drawKeypoints(frame, firstTenKeyPoints, img_keypoints_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			imshow("Webcam Video", img_keypoints_frame);
			*/

			drawKeypoints(frame, keypoints_frame, img_keypoints_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			imshow("Webcam Video", img_keypoints_frame);
			std::vector<DMatch> matches_1, matches_2, matches_3, matches_4, matches_5, matches_6, matches_7, matches_8, matches_9;
//			std::vector<DMatch> good_matches_1, good_matches_2, good_matches_3, good_matches_4, good_matches_5, good_matches_6, good_matches_7, good_matches_8, good_matches_9;
			FlannBasedMatcher matcher;
			BFMatcher matcherSecond;

			vector <int> total_good_matches;

			// Clear vectors.
			all_good_matches.clear();
			match_coordinates.clear();
			unique_matches.clear();
			coordinates_vector.clear();

			// Reset the value of min_x, max_x, min_y, and max_y as well as the index counters and the range counters for comparison's sake.
			min_x = 100000;
			max_x = 0;
			min_y = 100000;
			max_y = 0;
			min_x_index = -1;
			max_x_index = -1;
			min_y_index = -1;
			max_y_index = -1;
			x_range = 400;
			y_range = 400;

			if (descriptors_frame.rows > 0) {
				// Find good matches.
				switch(numberOfImages) {
					case 1:
						good_matches_1 = findGoodMatches(1);

						for (int i = 0; i < good_matches_1.size(); i++) {
							good_matches_1[i].queryIdx = 1;
						}

						total_good_matches.push_back(good_matches_1.size());
						
						//showWebcamKeypointMatches(1, good_matches_1);
						
						all_good_matches.insert(all_good_matches.end(), good_matches_1.begin(), good_matches_1.end());
						showAllWebcamKeypointMatches();
						break;
					case 2:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());

						/*
						showWebcamKeypointMatches(1, good_matches_1);
						showWebcamKeypointMatches(2, good_matches_2);
						*/

						all_good_matches.insert(all_good_matches.end(), good_matches_1.begin(), good_matches_1.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_2.begin(), good_matches_2.end());
						showAllWebcamKeypointMatches();
						break;
					case 3:
						good_matches_1 = findGoodMatches(1);
						good_matches_2 = findGoodMatches(2);
						good_matches_3 = findGoodMatches(3);
						total_good_matches.push_back(good_matches_1.size());
						total_good_matches.push_back(good_matches_2.size());
						total_good_matches.push_back(good_matches_3.size());

						/*
						showWebcamKeypointMatches(1, good_matches_1);
						showWebcamKeypointMatches(2, good_matches_2);
						showWebcamKeypointMatches(3, good_matches_3);
						*/

						all_good_matches.insert(all_good_matches.end(), good_matches_1.begin(), good_matches_1.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_2.begin(), good_matches_2.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_3.begin(), good_matches_3.end());
						showAllWebcamKeypointMatches();
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

						/*
						showWebcamKeypointMatches(1, good_matches_1);
						showWebcamKeypointMatches(2, good_matches_2);
						showWebcamKeypointMatches(3, good_matches_3);
						showWebcamKeypointMatches(4, good_matches_4);
						*/

						all_good_matches.insert(all_good_matches.end(), good_matches_1.begin(), good_matches_1.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_2.begin(), good_matches_2.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_3.begin(), good_matches_3.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_4.begin(), good_matches_4.end());
						showAllWebcamKeypointMatches();
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

						/*
						showWebcamKeypointMatches(1, good_matches_1);
						showWebcamKeypointMatches(2, good_matches_2);
						showWebcamKeypointMatches(3, good_matches_3);
						showWebcamKeypointMatches(4, good_matches_4);
						showWebcamKeypointMatches(5, good_matches_5);
						*/

						all_good_matches.insert(all_good_matches.end(), good_matches_1.begin(), good_matches_1.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_2.begin(), good_matches_2.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_3.begin(), good_matches_3.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_4.begin(), good_matches_4.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_5.begin(), good_matches_5.end());
						showAllWebcamKeypointMatches();
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

						/*
						showWebcamKeypointMatches(1, good_matches_1);
						showWebcamKeypointMatches(2, good_matches_2);
						showWebcamKeypointMatches(3, good_matches_3);
						showWebcamKeypointMatches(4, good_matches_4);
						showWebcamKeypointMatches(5, good_matches_5);
						showWebcamKeypointMatches(6, good_matches_6);
						*/

						all_good_matches.insert(all_good_matches.end(), good_matches_1.begin(), good_matches_1.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_2.begin(), good_matches_2.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_3.begin(), good_matches_3.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_4.begin(), good_matches_4.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_5.begin(), good_matches_5.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_6.begin(), good_matches_6.end());
						showAllWebcamKeypointMatches();
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
						
						/*
						showWebcamKeypointMatches(1, good_matches_1);
						showWebcamKeypointMatches(2, good_matches_2);
						showWebcamKeypointMatches(3, good_matches_3);
						showWebcamKeypointMatches(4, good_matches_4);
						showWebcamKeypointMatches(5, good_matches_5);
						showWebcamKeypointMatches(6, good_matches_6);
						showWebcamKeypointMatches(7, good_matches_7);
						*/

						all_good_matches.insert(all_good_matches.end(), good_matches_1.begin(), good_matches_1.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_2.begin(), good_matches_2.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_3.begin(), good_matches_3.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_4.begin(), good_matches_4.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_5.begin(), good_matches_5.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_6.begin(), good_matches_6.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_7.begin(), good_matches_7.end());
						showAllWebcamKeypointMatches();
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

						/*
						showWebcamKeypointMatches(1, good_matches_1);
						showWebcamKeypointMatches(2, good_matches_2);
						showWebcamKeypointMatches(3, good_matches_3);
						showWebcamKeypointMatches(4, good_matches_4);
						showWebcamKeypointMatches(5, good_matches_5);
						showWebcamKeypointMatches(6, good_matches_6);
						showWebcamKeypointMatches(7, good_matches_7);
						showWebcamKeypointMatches(8, good_matches_8);
						*/

						all_good_matches.insert(all_good_matches.end(), good_matches_1.begin(), good_matches_1.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_2.begin(), good_matches_2.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_3.begin(), good_matches_3.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_4.begin(), good_matches_4.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_5.begin(), good_matches_5.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_6.begin(), good_matches_6.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_7.begin(), good_matches_7.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_8.begin(), good_matches_8.end());
						showAllWebcamKeypointMatches();
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

						/*
						showWebcamKeypointMatches(1, good_matches_1);
						showWebcamKeypointMatches(2, good_matches_2);
						showWebcamKeypointMatches(3, good_matches_3);
						showWebcamKeypointMatches(4, good_matches_4);
						showWebcamKeypointMatches(5, good_matches_5);
						showWebcamKeypointMatches(6, good_matches_6);
						showWebcamKeypointMatches(7, good_matches_7);
						showWebcamKeypointMatches(8, good_matches_8);
						showWebcamKeypointMatches(9, good_matches_9);
						*/

						all_good_matches.insert(all_good_matches.end(), good_matches_1.begin(), good_matches_1.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_2.begin(), good_matches_2.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_3.begin(), good_matches_3.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_4.begin(), good_matches_4.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_5.begin(), good_matches_5.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_6.begin(), good_matches_6.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_7.begin(), good_matches_7.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_8.begin(), good_matches_8.end());
						all_good_matches.insert(all_good_matches.end(), good_matches_9.begin(), good_matches_9.end());
						showAllWebcamKeypointMatches();
						break;
					default:
						break;
				}
			}

			/*
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
			*/

			// Print total good matches.
			// printTotalGoodMatches(total_good_matches);

			Point2f lego_girl_location;
			lego_girl_location.x = 0;
			lego_girl_location.y = 0;

			if (all_good_matches.size() >= matches_threshold) {
				/*
				printf("legoGirl on screen!\n");

				//Print out contents of all_good_matches.
				for (int i = 0; i < all_good_matches.size(); i++) {
					printf("all_good_matches[%d]:\t	 queryIdx:%d\t trainIdx:%d\t imgIdx:%d\t distance:%f\n", i, all_good_matches[i].queryIdx, all_good_matches[i].trainIdx, all_good_matches[i].imgIdx, all_good_matches[i].distance);
				}

				printf("\n\n");
				*/

				// Find coordinates.
				lego_girl_location = findCoordinates();

				if (previous_x != 0 && previous_y != 0) {
					x_diff = lego_girl_location.x - previous_x;
					y_diff = lego_girl_location.y - previous_y;

					printf("x = %f	(%f) \t y = %f \t (%f) \t all_good_matches.size = %d\n", lego_girl_location.x, x_diff, lego_girl_location.y, y_diff, all_good_matches.size());
				}
				else {
					printf("x = %f	\t	y = %f \t\n", lego_girl_location.x, lego_girl_location.y);
				}

				previous_x = lego_girl_location.x;
				previous_y = lego_girl_location.y;

			} else {
				std::cout << name << " is not on screen \t\t all_good_matches.size = " << all_good_matches.size() << std::endl;

				previous_x = 0;
				previous_y = 0;
			}

			/*
			// Print good matches.
			switch(numberOfImages) {
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