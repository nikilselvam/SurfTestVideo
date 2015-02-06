#include "FeaturefulObject.h"
#include <fstream>
#include <iostream>

// Constructor
FeaturefulObject::FeaturefulObject(std::string objectName, int numImages, std::vector<std::string> images, float minDistance, int minMatches) {
	SurfFeatureDetector detector(400);
	SurfDescriptorExtractor extractor;

	Mat img_to_process;
	std::vector<KeyPoint> keypoints_to_process;
	Mat descriptors_to_process;
	
	name = objectName;
	numberOfImages = numImages;
	numberOfGoodMatches = 0;

	previousX = 0;
	previousY = 0;

	// Try to open keypoints file for an object.
	std::string nameOfFile = "Keypoints/" + objectName + "-keypoints.txt";
	std::fstream detection_file(nameOfFile);
	
	// If keypoints file does not exist, detect keypoints and extractors.
	// Save keypoints to to the keypoints file.
	if (!detection_file) {
		std::cout << "No existing keypoints file!" << std::endl;

		// Compute keypoints and descriptors.
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


		// Fle does not exist.
		std::ofstream myFile;

		myFile.open(nameOfFile);
		myFile << keypoints.size() << "\n\n";

		for (int i = 0; i < keypoints.size(); i++) {
			vector<KeyPoint> currKeyPointVector = keypoints[i];

			myFile << currKeyPointVector.size() << "\n\n";

			for (int j = 0; j < currKeyPointVector.size(); j++) {
				myFile << j << "\n";
				myFile << currKeyPointVector[j].class_id << "\n";
				myFile << currKeyPointVector[j].octave << "\n";
				myFile << currKeyPointVector[j].response << "\n";
				myFile << currKeyPointVector[j].size << "\n";
				myFile << currKeyPointVector[j].angle << "\n";
				myFile << currKeyPointVector[j].pt.x << "\n";
				myFile << currKeyPointVector[j].pt.y << "\n\n";
			}

			myFile << "\n\n";
		}

		myFile.close();
	}
	// If keypoints file does exist, populate keypoints vector by parsing file.
	// Then extract descriptors.
	else {
		std::cout << "Keypoints file exists!" << std::endl;

		// Read in keypoints.
		std::string line;
		int numberOfVectors = 0;
		int numberOfKeyPoints = 0;

		getline(detection_file, line);
		numberOfVectors = atoi(line.c_str());

		// Burn a line.
		getline(detection_file, line);

		// Get all keypoints.
		for (int i = 0; i < numberOfVectors; i++) {
			// Get number of keypoints in vector.
			getline(detection_file, line);
			numberOfKeyPoints = atoi(line.c_str());

			// Burn a line.
			getline(detection_file, line);

			// Set variables.
			int index = 0;
			int class_id = 0;
			int octave = 0;
			float response = 0;
			float size = 0;
			float angle = 0;
			float x = 0;
			float y = 0;

			vector<KeyPoint> currVector;
			KeyPoint currKeypoint;
			Point_ <float> currPoint;

			// Loop through all of the keypoints.
			for (int j = 0; j < numberOfKeyPoints; j++) {
				// Get current index of keypoint in vector (just for information).
				getline(detection_file, line);
				index = atoi(line.c_str());

				// Get class_id.
				getline(detection_file, line);
				class_id = atoi(line.c_str());
	
				// Get octave.
				getline(detection_file, line);
				octave = atoi(line.c_str());

				// Get response.
				getline(detection_file, line);
				response = atof(line.c_str());

				// Get size.
				getline(detection_file, line);
				size = atof(line.c_str());

				// Get angle.
				getline(detection_file, line);
				angle = atof(line.c_str());

				// Get x.
				getline(detection_file, line);
				x = atof(line.c_str());

				// Get y.
				getline(detection_file, line);
				y = atof(line.c_str());

				//currPoint = new Point (x, y);
				currPoint.x = x;
				currPoint.y = y;

				currKeypoint.class_id = class_id;
				currKeypoint.octave = octave;
				currKeypoint.response = response;
				currKeypoint.size = size;
				currKeypoint.angle = angle;
				currKeypoint.pt = currPoint;

				currVector.push_back(currKeypoint);

				// Burn a line between different keypoints.
				getline(detection_file, line);
			}

			// Push keypoint vector into keypoints vector (which contains vectors of keypoints).
			keypoints.push_back(currVector);

			// Burn two lines between different keypoint vectors.
			getline(detection_file, line);
			getline(detection_file, line);
		}

		detection_file.close();

		// Get descriptors.
		for (int i = 0; i < images.size(); i++) {
			// Save the names of target images.
			targetImages.push_back(images[i]);

			// Read in the image to process.
			Mat img_to_process = imread(images[i], CV_LOAD_IMAGE_GRAYSCALE );
			
			keypoints_to_process = keypoints[i];
			
			extractor.compute( img_to_process, keypoints_to_process, descriptors_to_process );
			descriptors.push_back(descriptors_to_process);

		}
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

int FeaturefulObject::get_numberOfGoodMatches() {
	return numberOfGoodMatches;
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

void FeaturefulObject::set_numberOfgoodMatches(int updatedNumberOfGoodMatches) {
	numberOfGoodMatches = updatedNumberOfGoodMatches;
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

	// Set number of good matches to 0 since we are in the process of determining good matches.
	numberOfGoodMatches = 0;

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

	// Now that all good matches have been found, update the number of good matches.
	for (int i = 0; i < good_matches.size(); i++) {
		numberOfGoodMatches += good_matches[i].size();
	}

	return;
}

void FeaturefulObject::clearMatches() {
	matches.clear();
}

void FeaturefulObject::clearGoodMatches() {
	good_matches.clear();
	numberOfGoodMatches = 0;
}