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
std::vector<KeyPoint> keypoints_frame;
vector <DetectionCoordinates> coordinates_vector;

SurfDescriptorExtractor extractor;
Mat descriptors_frame;
Mat img_keypoints_1, img_keypoints_2, img_keypoints_3, img_keypoints_4, img_keypoints_5, img_keypoints_6, img_keypoints_7, img_keypoints_8, img_keypoints_9, img_keypoints_frame;
std::vector<DMatch> all_good_matches, unique_matches;
std::vector<Point2f> match_coordinates;

// Set up tracking of (x,y) coordinate changes with legoGirl's location.
float previous_x = 0; float previous_y = 0;
float x_diff = 0; float y_diff = 0;

// Data for frames
Mat frame;
Mat img_matches_1, img_matches_2, img_matches_3, img_matches_4, img_matches_5, img_matches_6, img_matches_7, img_matches_8, img_matches_9;

std::vector<DMatch> findGoodMatches(int object_index, int img_number) {
	Mat descriptors_to_process;
	std::vector<DMatch> matches, good_matches;
	FlannBasedMatcher matcher;
	BFMatcher matcherSecond;

	int img_index = img_number - 1;
	descriptors_to_process = featureful_objects[object_index].get_descriptor(img_index);

	//matcher.match(descriptors_to_process, descriptors_frame, matches);
	//matcher.knnMatch(descriptors_to_process, descriptors_frame, matches, 5, storeKnnMatches);
	matcherSecond.match(descriptors_to_process, descriptors_frame, matches);

	for( int i = 0; i < descriptors_to_process.rows; i++ ) {
		if( matches[i].distance <= distance_threshold ) {
			// Set matches imgIdx to appropriate img_number and then push match to good_matches.
			matches[i].imgIdx = img_number;

			good_matches.push_back(matches[i]);
		}
	}

	return good_matches;
}

void findGoodMatches(int object_index) {
	Mat descriptors_to_process;
	std::vector<DMatch> matches, good_matches;
	FlannBasedMatcher matcher;
	BFMatcher matcherSecond;

	FeaturefulObject objectToMatch = featureful_objects[object_index];
	
	// Go through all descriptors.
	for (int i = 0; i < objectToMatch.get_numImages(); i++) {
		// Clear the vectors.
		matches.clear();
		good_matches.clear();

		descriptors_to_process = objectToMatch.get_descriptor(i);
		matcherSecond.match(descriptors_to_process, descriptors_frame, matches);
		
		for (int j = 0; j < descriptors_to_process.rows; j++) {
			matches[j].imgIdx = i;

			good_matches.push_back(matches[j]);
		}

		// Save good_matches to featureful object.
	}

	return;
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

void printMatchesVector (std::string name, vector<DMatch> vector, bool printDist) {
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
	// printDetectionCoordinatesVector("coordinates_vector", coordinates_vector);

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

	// printMatchesVector("all_good_matches", all_good_matches, true);
	// printMatchesVector("unique_matches", unique_matches, true);

	// Get match_coordinates of unique matches.
	for (int i = 0; i < unique_matches.size(); i++) {
		match_coordinates.push_back(keypoints_frame[unique_matches[i].trainIdx].pt);		
	}

	// printKeypointsVector("keypoints_frame", keypoints_frame);
	// printPoint2fVector("match_coordinates", match_coordinates);

// 	printf("all_good_matches.size = %d, unique_matches.size = %d\n", all_good_matches.size(), unique_matches.size());

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
	// Read in file with all object detection inforation and store information in vector.
	readInMultipleObjectDetectionFile();
	printFeaturefulObjectsVector("featureful_objects", featureful_objects);

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

			// Detect the keypionts in the frame and draw them on the screen.
			detector.detect(frame, keypoints_frame);
			extractor.compute( frame, keypoints_frame, descriptors_frame);

			drawKeypoints(frame, keypoints_frame, img_keypoints_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			imshow("Webcam Video", img_keypoints_frame);

			vector <int> total_good_matches;
			vector <DMatch> current_good_matches;
			vector <int> matches_per_object;
			
			// Compare each object to each frame to determine if it is on the screen or not. If it is on the screen,
			// output the (x,y) coordinates.
			for (int x = 0; x < featureful_objects.size(); x++) {
				// Clear vectors.
				all_good_matches.clear();
				total_good_matches.clear();
				current_good_matches.clear();
				matches_per_object.clear();
				match_coordinates.clear();
				unique_matches.clear();
				coordinates_vector.clear();

				featureful_objects[x].clearMatches();
				featureful_objects[x].clearGoodMatches();

				// Reset values.
				name = featureful_objects[x].get_name();
				numberOfImages = featureful_objects[x].get_numImages();
				distance_threshold = featureful_objects[x].get_distanceThreshold();
				matches_threshold = featureful_objects[x].get_matchesThreshold();
				previous_x = featureful_objects[x].get_previousX();
				previous_y = featureful_objects[x].get_previousY();

				if (descriptors_frame.rows > 0) {
					// Find good matches between the current image and the current frame..
					featureful_objects[x].findMatches(descriptors_frame);
					featureful_objects[x].findGoodMatches();

					// Add good matches found to aggregate.
					for (int y = 0; y < numberOfImages; y++) {
						current_good_matches.clear();
						current_good_matches = featureful_objects[x].get_goodMatch(y);
						total_good_matches.push_back(current_good_matches.size());
						all_good_matches.insert(all_good_matches.end(), current_good_matches.begin(), current_good_matches.end());
					}
				}

				showAllWebcamKeypointMatches();

				// Print total good matches.
				// printTotalGoodMatches(total_good_matches);

				Point2f lego_girl_location;
				lego_girl_location.x = 0;
				lego_girl_location.y = 0;
				
				// Save matches for this object.
				matches_per_object.push_back(all_good_matches.size());

				if (all_good_matches.size() >= matches_threshold) {
					std::cout << name << " is on screen \t agm = " << all_good_matches.size() << "\t ngm = " << featureful_objects[x].get_numberOfGoodMatches() << std::endl;

					// Find unique matches and then find coordinates.
					lego_girl_location = findCoordinates();

					if (previous_x != 0 && previous_y != 0) {
						x_diff = lego_girl_location.x - previous_x;
						y_diff = lego_girl_location.y - previous_y;

						printf("x = %f	(%f) \t y = %f \t (%f) \n", lego_girl_location.x, x_diff, lego_girl_location.y, y_diff);
					}
					else {
						printf("x = %f	\t	y = %f \t\n", lego_girl_location.x, lego_girl_location.y);
					}

					featureful_objects[x].set_previousX(lego_girl_location.x);
					featureful_objects[x].set_previousY( lego_girl_location.y);

				} else {
					std::cout << name << " is not on screen \t agm = " << all_good_matches.size() << "\t ngm = " << featureful_objects[x].get_numberOfGoodMatches() << std::endl;

					featureful_objects[x].set_previousX(0);
					featureful_objects[x].set_previousY(0);
				}

				printf("\n\n");
			}

			// Print out the results of each object.
			for (int x = 0; x < featureful_objects.size(); x++) {
				std::cout << featureful_objects[x].get_name() << ":" << "\t matches = " << featureful_objects[x].get_numberOfGoodMatches() << std::endl;
			}

			printf("\n\n");

			if(waitKey(30) >= 0) break;
		}
	}
	return 0;
}

/** @function readme */
void readme() {
	std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl;
}