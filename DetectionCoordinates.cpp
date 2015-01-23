#include "DetectionCoordinates.h"

DetectionCoordinates::DetectionCoordinates(float x_val, float y_val) {
	x = x_val;
	y = y_val;
	total = x + y;
}

float DetectionCoordinates::get_x(){
	return x;
}

float DetectionCoordinates::get_y(){
	return y;
}

float DetectionCoordinates::get_total(){
	return total;
}

void DetectionCoordinates::set_x(float x_val){
	x = x_val;
	total = x + y;
}

void DetectionCoordinates::set_y(float y_val){
	y = y_val;
	total = x + y;
}