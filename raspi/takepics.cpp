#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <ctime>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

static const cv::Size frameSize(640,480);
static const double cameraFPS = 4;

static const cv::Size displaySize(320, 240);
static const double displayRatio = double(displaySize.height) / frameSize.height;
static const char* detection_window = "Object Detection";
static cv::Mat display;

std::string date_now()
{
	std::time_t result = std::time(nullptr);
	std::string str(std::asctime(std::localtime(&result)));
	// trim trailing endl
	size_t endpos = str.find_last_not_of(" \t\n");
	if( str.npos != endpos )
	{
	    str = str.substr( 0, endpos+1 );
	}
	return str + " ";
}

int main(int argc, const char** argv)
{
	static int n_file = 0;
	cv::Mat frame, hsv, filtered, buffer1;
	static cv::Vec3i BlobLower(66, 200,  30);
	static cv::Vec3i BlobUpper(94, 255, 255);
	static int dispMode = 2; // 0: none, 1: bw, 2: color

	cv::VideoCapture capture;
	for(;;) {
		capture.open(0);
		capture.set(cv::CAP_PROP_FRAME_WIDTH, frameSize.width);
		capture.set(cv::CAP_PROP_FRAME_HEIGHT, frameSize.height);
		capture.set(cv::CAP_PROP_FPS, cameraFPS);
		capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // Magic! 0.25 means manual exposure, 0.75 = auto
		capture.set(cv::CAP_PROP_EXPOSURE, 0.001);
		capture.set(cv::CAP_PROP_BRIGHTNESS, 0.5);
		capture.set(cv::CAP_PROP_CONTRAST, 0.5);
		capture.set(cv::CAP_PROP_SATURATION, 0.5);
		if(capture.isOpened()) break;
		std::cerr << "Couldn't connect to camera" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(5));
	}
	std::cerr << date_now() << "Camera opened."
			<< " Resolution: " << capture.get(cv::CAP_PROP_FRAME_WIDTH)
			<<             "x" << capture.get(cv::CAP_PROP_FRAME_HEIGHT)
			<< " FPS: "        << capture.get(cv::CAP_PROP_FPS)
			<< std::endl;

	cv::namedWindow(detection_window, cv::WINDOW_NORMAL);
	cv::createTrackbar("Lo H",detection_window, &BlobLower[0], 255);
	cv::createTrackbar("Hi H",detection_window, &BlobUpper[0], 255);
	cv::createTrackbar("Lo S",detection_window, &BlobLower[1], 255);
	cv::createTrackbar("Hi S",detection_window, &BlobUpper[1], 255);
	cv::createTrackbar("Lo V",detection_window, &BlobLower[2], 255);
	cv::createTrackbar("Hi V",detection_window, &BlobUpper[2], 255);

	for(int n; ; ++n) {
		capture >> frame;
		if (frame.empty()) {
			std::cerr << " Error reading from camera, empty frame." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}
		std::vector<long int> timer_values;
		std::vector<std::string> timer_names;
		timer_names.push_back("start"); timer_values.push_back(cv::getTickCount());

		cv::cvtColor(frame, hsv, CV_BGR2HSV);
		inRange(hsv, BlobLower, BlobUpper, filtered);
		timer_names.push_back("inRange applied"); timer_values.push_back(cv::getTickCount());

		switch(dispMode) {
		case 1:
			cv::resize(filtered, display, displaySize);
			break;
		case 2:
			cv::resize(frame, display, displaySize);
			break;
		}
		timer_names.push_back("display resized"); timer_values.push_back(cv::getTickCount());

		if (dispMode > 0 and n%int(cameraFPS) == 0) {
			for(size_t i = 0; i < timer_values.size(); ++i) {
				long int val;
				if(i == 0) val = timer_values[timer_values.size()-1] - timer_values[0];
				else val = timer_values[i] - timer_values[i-1];
				std::ostringstream osst;
				osst << timer_names[i] << ": " << val / cv::getTickFrequency();
				cv::putText(display, osst.str(), cv::Point(20,60+20*i), 0, 0.33, cv::Scalar(0,200,200));
			}

			cv::imshow(detection_window, display);
		}

		int key = cv::waitKey(2);
		if ((key & 255) == 27) break;
		if ((key & 255) == 32) {
			if(++dispMode > 2) dispMode =0;
		}
		if ((key & 255) == 's') cv::waitKey(0);
		if ((key & 255) == 'w') {
			std::ostringstream filename;
			filename << "img" << n_file++ << ".png";
			imwrite(filename.str(), frame);
		}
	}
	return 0;
}

