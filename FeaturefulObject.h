#include <string>
#include <vector>

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
	float total;
};