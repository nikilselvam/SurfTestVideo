#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

class FeaturefulObject {
public:
	// Constructor
	FeaturefulObject(std::string objectName, int numImages, std::vector<std::string> images, float minDistance, int minMatches);
	
	// Getters
	std::string get_name();
	int get_numImages();
	std::vector <std::string> get_images();
	float get_distanceThreshold();
	int get_matchesThreshold();
	std::vector <std::vector<KeyPoint> > get_keypointsVector();
	std::vector<KeyPoint> get_keypoints(int keypoint_index);
	vector <Mat> get_descriptorsVector();
	Mat get_descriptor(int descriptor_index);
	
	// Setters
	void set_name(std::string updatedName);
	void set_numImages(int updatedNumber);
	void set_images(std::vector <std::string> images);
	void get_distanceThreshold (float updatedDistanceThreshold);
	void set_matchesThreshold(int updatedMatchesThreshold);
private:
	// Attributes
	std::string name;
	int numberOfImages;
	std::vector <std::string> targetImages;
	float distanceThreshold;
	float matchesThreshold;
	std::vector <std::vector<KeyPoint> > keypoints;
	std::vector <Mat> descriptors;
};