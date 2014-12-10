#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

// Constant to control webcam device number
const int WEBCAM_DEVICE_NUMBER = 0;

// Declare webcamCapture object associated with selected webcam device
VideoCapture webcamCapture(WEBCAM_DEVICE_NUMBER);

void readme();

/** @function main */
int main( int argc, char** argv )
{
	if( argc != 2 ) {
		readme(); return -1;
	}

	Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	
	if(!img_1.data) {
		std::cout<< " --(!) Error reading images " << std::endl; return -1;
	}

	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	SurfDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;
	Mat img_keypoints_1, img_keypoints_2;

	// ============== Lego Girl ===============================
	detector.detect( img_1, keypoints_1 );
	extractor.compute( img_1, keypoints_1, descriptors_1 );
	drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	imshow("Keypoints 1", img_keypoints_1 );

	// set up matching
	double max_dist = 0; double min_dist = 100;

	// ============== Capture Camera Frame ====================
	Mat frame;
	Mat img_matches;
	// Capturing data from camera
		// Caution: It requires webcam to be attached
	VideoCapture cap(0);
		// you can also use
		// cv::VideoCapture cap("video filename");
		// to capture the frame from a video instead of webcam
	if(!cap.isOpened())
		printf("No Camera Detected");
	else {
		while(true) {
			cap >> frame; // get a new frame from camera

			detector.detect(frame, keypoints_2);
			extractor.compute( frame, keypoints_2, descriptors_2 );

			// Step 3: Matching descritor vectors usign FLANN matcher
			FlannBasedMatcher matcher;
			//BFMatcher matcher2(NORM_L1);
			std::vector< vector < DMatch > > matches2;
			matcher.knnMatch(descriptors_1, descriptors_2, matches2, 2);
			vector <DMatch> good_matches;
			good_matches.reserve(matches2.size());
			float nndrRatio = 0.7f;

			for (size_t i = 0; i < matches2.size(); ++i) {
				if (matches2[i].size() < 2)
					continue;

				const DMatch &m1 = matches2[i][0];
				const DMatch &m2 = matches2[i][1];

				if (m1.distance <= nndrRatio * m2.distance)
					good_matches.push_back(m1);
			}

			if (good_matches.size() >= 7) {
				drawMatches( img_1, keypoints_1, frame, keypoints_2, 
					good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

				imshow("Good Matches", img_matches);
			}


			printf("max(2*min_dist, 0.02) is %f\n\n", max(2*min_dist, 0.02));
			for( int i = 0; i < (int)good_matches.size(); i++ )
			{ 
				printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d with distance %f \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance );
			}



			//drawKeypoints(frame, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

			//imshow("Webcam Video", img_keypoints_2);
			if(waitKey(30) >= 0) break;
		}
	}
	return 0;
}

/** @function readme */
void readme() {
	std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl;
}