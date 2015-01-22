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

/*
std::string targetImage = "testLegoGirl.jpg";
std::string targetImage1 = "legoGirl- 1.jpg";
std::string targetImage2 = "legoGirl- 2.jpg";
std::string targetImage3 = "legoGirl- 3.jpg";
std::string targetImage4 = "legoGirl- 4.jpg";
std::string targetImage5 = "legoGirl- 5.jpg";
std::string targetImage6 = "legoGirl- 6.jpg";
std::string targetImage7 = "legoGirl- 7.jpg";
std::string targetImage8 = "legoGirl- 8.jpg";
std::string targetImage9 = "legoGirl- 9.jpg";

*/

std::string targetImage1 = "Picture 9.jpg";
std::string targetImage2 = "Picture 10.jpg";
std::string targetImage3 = "Picture 11.jpg";
std::string targetImage4 = "Picture 12.jpg";
std::string targetImage5 = "Picture 13.jpg";
std::string targetImage6 = "Picture 14.jpg";
std::string targetImage7 = "Picture 15.jpg";
std::string targetImage8 = "Picture 16.jpg";
std::string targetImage9 = "Picture 17.jpg";

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
std::vector<DMatch> good_matches_1, good_matches_2, good_matches_3, good_matches_4, good_matches_5, good_matches_6, good_matches_7, good_matches_8, good_matches_9, all_good_matches, unique_matches;
std::vector<Point2f> match_coordinates;

// Set up matching
double max_dist = 0; double min_dist = 100;
float min_x = 100000; float max_x = 0;
float min_y = 100000; float max_y = 0;
int min_x_index = -1; int max_x_index = -1;
int min_y_index = -1; int max_y_index = -1;
float x_range = 400; float y_range = 400;

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
		if( matches[i].distance <= 0.3 ) {
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

Point2f findCoordinates() {
	std::vector<KeyPoint> all_matched_keypoints;
	std::vector <KeyPoint> keypoints_to_process;
	int img_number, queryIdx;
	Point2f lego_girl_coordinates;
	
	lego_girl_coordinates.x = 0;
	lego_girl_coordinates.y = 0;

	for (int i = 0; i < all_good_matches.size(); i++) {
		img_number = all_good_matches[i].imgIdx;

		switch(img_number) {
			case 1:
				keypoints_to_process = keypoints_1;
				break;
			case 2:
				keypoints_to_process = keypoints_2;
				break;
			case 3:
				keypoints_to_process = keypoints_3;
				break;
			case 4:
				keypoints_to_process = keypoints_4;
				break;
			case 5:
				keypoints_to_process = keypoints_5;
				break;
			case 6:
				keypoints_to_process = keypoints_6;
				break;
			case 7:
				keypoints_to_process = keypoints_7;
				break;
			case 8:
				keypoints_to_process = keypoints_8;
				break;
			case 9:
				keypoints_to_process = keypoints_9;
				break;
			default:
				return lego_girl_coordinates;
		}
		
		/*
		queryIdx = all_good_matches[i].queryIdx;
		Point2f coordinates = keypoints_to_process[queryIdx].pt;
		*/

		int trainIdx = all_good_matches[i].trainIdx;
		Point2f coordinates = keypoints_frame[trainIdx].pt;

		bool matchFound = false;

		/*
		// Print contents of unique_matches.
		printf("Printing contents of unique matches\n\n");
		for (int j = 0; j < unique_matches.size(); j++) {
			printf("unique_match[%d]:	\t queryIdx:%d\t trainIdx:%d\t imgIdx:%d\t distance:%f\n", j, unique_matches[j].queryIdx, unique_matches[j].trainIdx, unique_matches[j].imgIdx, unique_matches[j].distance);
		}

		printf("\n\n");
		*/

		// Search through unique_matches and see if a match with the same trainIdx already exist.
		// If so, this means that we already have the corresponding keypoint in our match_coordinates vector and 
		// we dont' need to add it again. Set matchFound to true.
		// If not, add the match to unique_matches and push the (x,y) information to match_coordinates.
		for (int j = 0; j < unique_matches.size(); j++) {
			if (unique_matches[j].trainIdx == trainIdx) {
				matchFound = true;
				//printf("Not unique match! Match alredy exists in unique_matches at index %d\n", j);

				break;
			}
		}

		if (!matchFound) {
			//printf("Unique match found! Adding match with img_number = %d and trainIdx = %d to unique_matches\n", img_number, trainIdx);
			unique_matches.push_back(all_good_matches[i]);

			//printf("\n\n");

			/*
			printf("Printing contents of unique matches again\n\n");
			for (int j = 0; j < unique_matches.size(); j++) {
				printf("unique_match[%d]:	\t queryIdx:%d\t trainIdx:%d\t imgIdx:%d\t distance:%f\n", j, unique_matches[j].queryIdx, unique_matches[j].trainIdx, unique_matches[j].imgIdx, unique_matches[j].distance);
			}
			*/

			// Add (x,y) point to match coordinates.
			match_coordinates.push_back(keypoints_frame[trainIdx].pt);
		}

		/*
		printf("\n\n");

		printf("image number is %d\n", img_number);
		printf("keypoints vector size is %d\n", keypoints_to_process.size());
		printf("queryIdx is %d\n\n", queryIdx);

		printf("coordinates are (%f, %f)\n", coordinates.x, coordinates.y);
		printf("\n");
		*/
	}

	//printf("all_good_matches.size = %d, unique_matches.size = %d\n", all_good_matches.size(), unique_matches.size());

	// Print matches.
	for (int i = 0; i < match_coordinates.size(); i++) {
		// Calculate min and max x coordinates.
		if (match_coordinates[i].x > max_x) {
			max_x = match_coordinates[i].x;
			max_x_index = i;
		}
		if (match_coordinates[i].y > max_y) {
			max_y = match_coordinates[i].y;
			max_y_index = i;
		}

		// Calculate min and max y coordinates.
		if (match_coordinates[i].x < min_x) {
			min_x = match_coordinates[i].x;
			min_x_index = i;
		}
		if (match_coordinates[i].y < min_y) {
			min_y = match_coordinates[i].y;
			min_y_index = i;
		}

		//printf("Matches[%d]:	\t x = %f	\t	y = %f\n", i, match_coordinates[i].x, match_coordinates[i].y);
	}

	// Calculate x_range and y_range.
	x_range = max_x - min_x;
	y_range = max_y - min_y;

	/*

	printf("\n\n");
	printf("min_x = %f	\t max_x = %f\n", min_x, max_x);
	printf("min_y = %f	\t max_y = %f\n", min_y, max_y);
	printf("x__range = %f	\t y_range = %f\n", x_range, y_range);
	printf("min_x_index = %d	\t max_x_index = %d\n", min_x_index, max_x_index);
	printf("min_y_index = %d	\t max_y_index = %d\n", min_y_index, max_y_index);
	printf("\n\n");
	*/

	// Now continue removing coordinates from vector until the x and y coordinates match the range
	// of the legoGirl.
	while (x_range > 30 || y_range > 45) {
		int first_index_remove = -1;
		int second_index_remove = -1;

		/*
		printf("match_coordinates.size() = %d\n", match_coordinates.size());
				printf("x_range = %f	\t y_range = %f\n\n", x_range, y_range);
		
		for (int i = 0; i < match_coordinates.size(); i++) {
			float total_distance = match_coordinates[i].x + match_coordinates[i].y;
			printf("Matches[%d]:	\t x = %f	\t	y = %f\n total_distance = %f\n", i, match_coordinates[i].x, match_coordinates[i].y, total_distance);
		}
			
		printf("\n\n");
		*/

		if (x_range > 30) {
			first_index_remove = min_x_index;

			// TODO: put in check to ensure that min_x_value != min_y_value.

			// Calculate the second iterator to remove based on the first element removed.
			if (min_x_index < max_x_index) {
				second_index_remove = max_x_index - 1;
			} else {
				second_index_remove = max_x_index;
			}
			
			//printf("Removing min_x of value %f at iterator index %d and max_x of value %f at iterator index %d\n", min_x, first_index_remove, max_x, second_index_remove);

			// If there is only one element left in the Point2f vector, return this point.
			if (match_coordinates.size() == 1) {
				return match_coordinates[0];
			}

			// Remove the entry with the smallest x value.
			match_coordinates.erase(match_coordinates.begin() + first_index_remove);

			// If there is only one element left in the Point2f vector, return this point.
			if (match_coordinates.size() == 1) {
				return match_coordinates[0];
			}

			// Remove the entry with the largest x value.
			match_coordinates.erase(match_coordinates.begin() + second_index_remove);

			/*
			printf("\n\n");

			// Print out vector contents.
			for (int i = 0; i < match_coordinates.size(); i++) {
				printf("Matches[%d]:	\t x = %f	\t	y = %f\n", i, match_coordinates[i].x, match_coordinates[i].y);
			}

			printf("\n\n");
			*/

			
			// Now check if we need to remove entries based on the y_range as well.
			if (y_range > 45) {
				// Check if either point has already been removed.

				bool minYRemoved = false; bool maxYRemoved = false;

				if (min_y_index == min_x_index || min_y_index == max_x_index) {
					minYRemoved = true;
					// printf("Min Y already removed!\n\n");
				}

				if (max_y_index == min_x_index || max_y_index == max_x_index) {
					maxYRemoved = true;
					// printf("Max Y already removed!\n\n");
				}

				// If not, calculate new indices to remove.
				
				if (minYRemoved && maxYRemoved) {
					// Do nothing. Both values have already been removed.
				}
				// If min y has already been removed but max y hasn't, remove the entry with max y.
				else if (minYRemoved && !maxYRemoved) {
					// Remove the entry at max_y_index given that we have already removed two other entries from the list.

					int third_index_remove = -1;

					// If both indices removed are lesser in value than max_y_index, then subtract 2 from max_y_index
					// to account for the fact that these two entries have already been removed.
					if (min_x_index < max_y_index && max_x_index < max_y_index) {
						third_index_remove = max_y_index - 2;
					}
					// If only one index removed was less than max_y_index, subtract 1 from max_y_index to account for the fact
					// that one entry has already been removed that affects the index of the current max_y match.
					else if ( (min_x_index < max_y_index && max_x_index > max_y_index) || (min_x_index > max_y_index && max_x_index < max_y_index)) {
						third_index_remove = max_y_index - 1;
					}
					// If both indices removed are greater in value than max_y_index, then use max_y_index as the third index to remove.
					else if (min_x_index > max_y_index && max_x_index > max_y_index) {
						third_index_remove = max_y_index;
					}

					// printf("Removing max_y of value %f at iterator index %d\n", max_y, third_index_remove);

					// printf("\n\n");

					// If there is only one element left in the Point2f vector, return this point.
					if (match_coordinates.size() == 1) {
						return match_coordinates[0];
					}

					// Remove the entry with the largest y value.
					match_coordinates.erase(match_coordinates.begin() + third_index_remove);
				}
				// If max y has already been removed but min y hasn't, remove the entry with min y.
				else if (!minYRemoved && maxYRemoved) {
					// Remove the entry at max_y_index given that we have already removed two other entries from the list.
					int third_index_remove = -1;

					// If both indices removed are lesser in value than min_y_index, then subtract 2 from min_y_index
					// to account for the fact that these two entries have already been removed.
					if (min_x_index < min_y_index && max_x_index < min_y_index) {
						third_index_remove = min_y_index - 2;
					}
					// If only one index removed was less than min_y_index, subtract 1 from min_y_index to account for the fact
					// that one entry has already been removed that affects the index of the current min_y match.
					else if ( (min_x_index < min_y_index  && max_x_index > min_y_index ) || (min_x_index > min_y_index  && max_x_index < min_y_index )) {
						third_index_remove = min_y_index  - 1;
					}
					// If both indices removed are greater in value than min_y_index, then use min_y_index as the third index to remove.
					else if (min_x_index > min_y_index  && max_x_index > min_y_index ) {
						third_index_remove = min_y_index;
					}

					//printf("Removing min_y of value %f at iterator index %d\n", min_y, third_index_remove);

					//printf("\n\n");

					// If there is only one element left in the Point2f vector, return this point.
					if (match_coordinates.size() == 1) {
						return match_coordinates[0];
					}

					// Remove the entry with the smallest y value.
					match_coordinates.erase(match_coordinates.begin() + third_index_remove);
				}
				// If neither min y nor max y have been removed, then remove both.
				else if (!minYRemoved && !maxYRemoved) {
					// Calculate the third index we need to remove (for entry min y) given that we have already removed two other entries from the list.
					int third_index_remove = -1;

					// If both indices removed are lesser in value than min_y_index, then subtract 2 from min_y_index
					// to account for the fact that these two entries have already been removed.
					if (min_x_index < min_y_index && max_x_index < min_y_index) {
						third_index_remove = min_y_index - 2;
					}
					// If only one index removed was less than min_y_index, subtract 1 from min_y_index to account for the fact
					// that one entry has already been removed that affects the index of the current min_y match.
					else if ( (min_x_index < min_y_index  && max_x_index > min_y_index ) || (min_x_index > min_y_index  && max_x_index < min_y_index )) {
						third_index_remove = min_y_index  - 1;
					}
					// If both indices removed are greater in value than min_y_index, then use min_y_index as the third index to remove.
					else if (min_x_index > min_y_index  && max_x_index > min_y_index ) {
						third_index_remove = min_y_index;
					}

					//printf("\n\n");

					//printf("Removing min_y of value %f at iterator index %d\n", min_y, third_index_remove);

					// If there is only one element left in the Point2f vector, return this point.
					if (match_coordinates.size() == 1) {
						return match_coordinates[0];
					}

					// Remove the entry with the smallest y value.
					match_coordinates.erase(match_coordinates.begin() + third_index_remove);

					// Calculate the fourth index we need to remove (for entry max y) given that we have already removed three other entries from the list.
					int fourth_index_remove = -1;

					// If all three indices removed are lesser in value than max_y_index, then subtract 3 from max_y_index
					// to account for the fact that these two entries have already been removed.
					if (min_x_index < max_y_index && max_x_index < max_y_index && min_y_index < max_y_index) {
						fourth_index_remove = max_y_index - 3;
					}
					// If two of the indices removed are less than max_y_index, subtract 2 from max_y_index to account for the fact
					// that two entires have already been removed that affect the index of the current max_y match.
					else if ( (min_x_index < max_y_index && max_x_index < max_y_index && min_y_index > max_y_index) ||
								(min_x_index < max_y_index && max_x_index > max_y_index && min_y_index < max_y_index) || 
								(min_x_index > max_y_index && max_x_index < max_y_index && min_y_index < max_y_index) ) {
						fourth_index_remove = max_y_index - 2;
					}
					// If only one index removed was less than max_y_index, subtract 1 from max_y_index to account for the fact
					// that one entry has already been removed that affects the index of the current max_y match.
					else if ( (min_x_index > max_y_index && max_x_index > max_y_index && min_y_index < max_y_index) ||
								(min_x_index > max_y_index && max_x_index < max_y_index && min_y_index > max_y_index) || 
								(min_x_index < max_y_index && max_x_index > max_y_index && min_y_index > max_y_index) ) {
						fourth_index_remove = max_y_index - 1;
					}
					// If both indices removed are greater in value than max_y_index, then use max_y_index as the third index to remove.
					else if (min_x_index > max_y_index && max_x_index > max_y_index && min_y_index > max_y_index ) {
						fourth_index_remove = max_y_index;
					}

					//printf("\n\n");

					//printf("Removing max_y of value %f at iterator index %d\n", min_y, fourth_index_remove);

					// If there is only one element left in the Point2f vector, return this point.
					if (match_coordinates.size() == 1) {
						return match_coordinates[0];
					}

					// Remove the entry with the smallest y value.
					match_coordinates.erase(match_coordinates.begin() + fourth_index_remove);
				}
			}
			/*
			
			// Print out vector contents.
			for (int i = 0; i < match_coordinates.size(); i++) {
				printf("Matches[%d]:	\t x = %f	\t	y = %f\n", i, match_coordinates[i].x, match_coordinates[i].y);
			}

			printf("\n\n");
			*/
		}
		else if (y_range > 45) {
			first_index_remove = min_y_index;

			// TODO: put in check to ensure that min_x_value != min_y_value.

			// Calculate the second iterator to remove based on the first element removed.
			if (min_y_index < max_y_index) {
				second_index_remove = max_y_index - 1;
			} else {
				second_index_remove = max_y_index;
			}
			
			//printf("Removing min_y of value %f at iterator index %d and max_y of value %f at iterator index %d\n", min_y, first_index_remove, max_y, second_index_remove);

			// If there is only one element left in the Point2f vector, return this point.
			if (match_coordinates.size() == 1) {
				return match_coordinates[0];
			}

			// Remove the entry with the smallest x value.
			match_coordinates.erase(match_coordinates.begin() + first_index_remove);

			// If there is only one element left in the Point2f vector, return this point.
			if (match_coordinates.size() == 1) {
				return match_coordinates[0];
			}

			// Remove the entry with the largest x value.
			match_coordinates.erase(match_coordinates.begin() + second_index_remove);


			/*(
			printf("\n\n");

			// Print out vector contents.
			for (int i = 0; i < match_coordinates.size(); i++) {
				printf("Matches[%d]:	\t x = %f	\t	y = %f\n", i, match_coordinates[i].x, match_coordinates[i].y);
			}

			printf("\n\n");
			*/
		}

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

		// Recalculate x_range and y_range.
		for (int i = 0; i < match_coordinates.size(); i++) {
			// Calculate min and max x coordinates.
			if (match_coordinates[i].x > max_x) {
				max_x = match_coordinates[i].x;
				max_x_index = i;
			}
			if (match_coordinates[i].y > max_y) {
				max_y = match_coordinates[i].y;
				max_y_index = i;
			}

			// Calculate min and max y coordinates.
			if (match_coordinates[i].x < min_x) {
				min_x = match_coordinates[i].x;
				min_x_index = i;
			}
			if (match_coordinates[i].y < min_y) {
				min_y = match_coordinates[i].y;
				min_y_index = i;
			}

			//printf("Matches[%d]:	\t x = %f	\t	y = %f\n", i, match_coordinates[i].x, match_coordinates[i].y);
		}

		// Calculate x_range and y_range.
		x_range = max_x - min_x;
		y_range = max_y - min_y;
		
		/*
		printf("\n\n");
		printf("min_x = %f	\t max_x = %f\n", min_x, max_x);
		printf("min_y = %f	\t max_y = %f\n", min_y, max_y);
		printf("x__range = %f	\t y_range = %f\n", x_range, y_range);
		printf("min_x_index = %d	\t max_x_index = %d\n", min_x_index, max_x_index);
		printf("min_y_index = %d	\t max_y_index = %d\n", min_y_index, max_y_index);
		printf("\n\n");
		*/
	}


	// Now print out all matches again.
	//printf("Done eliminating keypoints!\n");

	// Calculate average.
	int size = match_coordinates.size();
	float x_sum = 0; float y_sum = 0;
	float x_val = 0; float y_val = 0;
	for (int i = 0; i < size; i++) {
		//printf("Matches[%d]:	\t x = %f	\t	y = %f\n", i, match_coordinates[i].x, match_coordinates[i].y);
		x_sum += match_coordinates[i].x;
		y_sum += match_coordinates[i].y;
	}

	//printf("\n\n");

	// Calculate and print averages.
	x_val = x_sum / size;
	y_val = y_sum / size;
	//printf("Done calculating!\n\n");
	//printf("x_val = %f	\t	y_val = %f	\n", x_val, y_val);
	//printf("\n\n");

	lego_girl_coordinates.x = x_val;
	lego_girl_coordinates.y = y_val;

	//printf("size = %d\n", size);

	return lego_girl_coordinates;
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

			// Show the images that were matched.

			/*
			std::vector<KeyPoint> matchedKeypoints;

			for (int i = 0; i < good_matches_1.size(); i++) {
				matchedKeypoints.push_back(keypoints_frame[good_matches_1[i].trainIdx]);
			}
			drawKeypoints(frame, matchedKeypoints, img_keypoints_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			imshow("Matched Keypoints 1", img_keypoints_frame);

			matchedKeypoints.clear();

			for (int i = 0; i < good_matches_2.size(); i++) {
				matchedKeypoints.push_back(keypoints_frame[good_matches_2[i].trainIdx]);
			}
			drawKeypoints(frame, matchedKeypoints, img_keypoints_frame, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			imshow("Matched Keypoints 2", img_keypoints_frame);
			*/

			// Print total good matches.
			// printTotalGoodMatches(total_good_matches);

			Point2f lego_girl_location;
			lego_girl_location.x = 0;
			lego_girl_location.y = 0;

			if (all_good_matches.size() >= 35) {
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
			} else {
				//printf("Not on screen.\n");
			}

			// Print out lego_girl location.
			printf("x = %f	\t	y = %f \t\tall_matches = %d\n", lego_girl_location.x, lego_girl_location.y, all_good_matches.size());

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