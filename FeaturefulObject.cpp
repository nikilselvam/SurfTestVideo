#include "FeaturefulObject.h"

// Constructor
FeaturefulObject::FeaturefulObject(std::string objectName, int numImages, std::vector<std::string> images, float minDistance, int minMatches) {
	SurfFeatureDetector detector(400);
	SurfDescriptorExtractor extractor;

	Mat img_to_process;
	std::vector<KeyPoint> keypoints_to_process;
	Mat descriptors_to_process;
	
	name = objectName;
	numberOfImages = numImages;

	previousX = 0;
	previousY = 0;

	for (int i = 0; i < images.size(); i++) {
		// Save the names of target images.
		targetImages.push_back(images[i]);

		// Read in the image to process.
		Mat img_to_process = imread(images[i], CV_LOAD_IMAGE_GRAYSCALE );

		// Compute the keypoints and descriptors for that image.
		detector.detect( img_to_process, keypoints_to_process );
		extractor.compute( img_to_process, keypoints_to_process, descriptors_to_process );

		// Save the keypoints and descriptors for that image.
		keypoints.push_back(keypoints_to_process);
		descriptors.push_back(descriptors_to_process);
	}

	distanceThreshold = minDistance;
	matchesThreshold = minMatches;
}
	
// Getters
std::string FeaturefulObject::get_name() {
	return name;
}

int FeaturefulObject::get_numImages() {
	return numberOfImages;
}

std::vector <std::string> FeaturefulObject::get_images() {
	return targetImages;
}
float FeaturefulObject::get_distanceThreshold() {
	return distanceThreshold;
}

int FeaturefulObject::get_matchesThreshold() {
	return matchesThreshold;
}

std::vector <std::vector<KeyPoint> > FeaturefulObject::get_keypointsVector() {
	return keypoints;
}	

std::vector<KeyPoint> FeaturefulObject::get_keypoints(int keypoint_index) {
	return keypoints[keypoint_index];
}

vector <Mat> FeaturefulObject::get_descriptorsVector() {
	return descriptors;
}

Mat FeaturefulObject::get_descriptor(int descriptor_index) {
	return descriptors[descriptor_index];
}

float FeaturefulObject::get_previousX() {
	return previousX;
}

float FeaturefulObject::get_previousY() {
	return previousY;
}

std::vector <std::vector<DMatch> > FeaturefulObject::get_matches() {
	return matches;
}

std::vector<DMatch> FeaturefulObject::get_match(int index) {
	return matches[index];
}

std::vector <std::vector<DMatch> > FeaturefulObject::get_goodMatches() {
	return good_matches;
}

std::vector<DMatch> FeaturefulObject::get_goodMatch(int index) {
	return good_matches[index];
}


// Setters
void FeaturefulObject::set_name(std::string updatedName) {
	name = updatedName;
}

void FeaturefulObject::set_numImages(int updatedNumber) {
	numberOfImages = updatedNumber;
}


void FeaturefulObject::set_images(std::vector <std::string> images) {
	targetImages.clear();

	for (int i = 0; i < images.size(); i++) {
		targetImages.push_back(images[i]);
	}
}

void FeaturefulObject::get_distanceThreshold (float updatedDistanceThreshold) {
	distanceThreshold = updatedDistanceThreshold;
}
void FeaturefulObject::set_matchesThreshold(int updatedMatchesThreshold) {
	matchesThreshold = updatedMatchesThreshold;
}

void FeaturefulObject::set_previousX(float updatedXValue) {
	previousX = updatedXValue;
}
void FeaturefulObject::set_previousY(float updatedYValue) {
	previousY = updatedYValue;
}

void FeaturefulObject::set_matches(std::vector <std::vector<DMatch> > updatedMatches) {
	matches.clear();
	matches = updatedMatches;
}

void FeaturefulObject::set_goodMatches(std::vector <std::vector<DMatch> > updatedGoodMatches) {
	good_matches.clear();
	good_matches = updatedGoodMatches;
}

// Methods
void FeaturefulObject::findMatches(Mat descriptors_frame) {
	Mat descriptor_to_process;
	std::vector<DMatch> current_matches;
	//FlannBasedMatcher matcher;
	BFMatcher matcherSecond;


	for (int i = 0; i < descriptors.size(); i++) {
		// Clear matches vectors for each new iteration.
		current_matches.clear();

		descriptor_to_process = descriptors[i];

		//matcher.match(descriptors_to_process, descriptors_frame, matches);
		matcherSecond.match(descriptor_to_process, descriptors_frame, current_matches);

		matches.push_back(current_matches);
	}
}

void FeaturefulObject::findGoodMatches() {
	Mat descriptor_to_process;
	std::vector<DMatch> current_matches;
	std::vector<DMatch> current_good_matches;

	for( int i = 0; i < matches.size(); i++ ) {

		// Clear the mathces vectors for each new itearation.
		current_matches.clear();
		current_good_matches.clear();

		current_matches = matches[i];

		descriptor_to_process = descriptors[i];

		for (int j = 0; j < descriptor_to_process.rows; j++) {
			if( current_matches[j].distance <= distanceThreshold ) {
				// Set matches imgIdx to appropriate index and then push match to good_matches.
				current_matches[j].imgIdx = i;

				current_good_matches.push_back(current_matches[j]);
			}		
		}

		good_matches.push_back(current_good_matches);
	}

	return;
}

void FeaturefulObject::clearMatches() {
	matches.clear();
}

void FeaturefulObject::clearGoodMatches() {
	good_matches.clear();
}