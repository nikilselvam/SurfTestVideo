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