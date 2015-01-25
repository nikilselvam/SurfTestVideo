#include "FeaturefulObject.h"


// Constructor
FeaturefulObject::FeaturefulObject(std::string objectName, int numImages, std::vector<std::string> images, float minDistance, int minMatches) {
	name = objectName;
	numberOfImages = numImages;

	for (int i = 0; i < images.size(); i++) {
		targetImages.push_back(images[i]);
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

std::vector <std::vector<KeyPoint> > FeaturefulObject::get_keypoints() {
	return keypoints;
}

vector <Mat> FeaturefulObject::get_descriptors() {
	return descriptors;
}

std::vector <std::vector <DMatch>> FeaturefulObject::get_goodMatches() {
	return good_matches;
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