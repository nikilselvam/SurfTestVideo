class DetectionCoordinates {
public:
	// Constructor
	DetectionCoordinates(float x_val, float y_val);
	
	// Getters
	float get_x();
	float get_y();
	float get_total();

	// Setters
	void set_x(float x_val);
	void set_y(float y_val);

	// Sort
	bool operator< (const DetectionCoordinates &other) const {
		return total < other.total;
	}
private:
	// Attributes
	float x;
	float y;
	float total;
};