#define XGUI_ENABLED

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <ctime>
#include "opencv2/opencv.hpp"

static const cv::Size frameSize(1296,972);
//static const cv::Size frameSize(1280, 720);
static const char* default_intrinsic_file = "jetson-camera-720.yml";

#ifdef XGUI_ENABLED
	#include "opencv2/highgui.hpp"
	static const cv::Size displaySize(640, 480);
	static const double displayRatio = double(displaySize.height) / frameSize.height;
	static const char* display_window = "Object Detection";
#endif


bool readIntrinsics(const char *filename, cv::Mat &intrinsic, cv::Mat &distortion)
{
	cv::FileStorage fs( filename, cv::FileStorage::READ );
	if( !fs.isOpened() )
	{
		std::cerr << "Error: Couldn't open intrinsic parameters file "
				<< filename << std::endl;
		return false;
	}
	fs["camera_matrix"] >> intrinsic;
	fs["distortion_coefficients"] >> distortion;
	if( intrinsic.empty() || distortion.empty() )
	{
		std::cerr << "Error: Couldn't load intrinsic parameters from "
				<< filename << std::endl;
		return false;
	}
	fs.release();
	return true;
}

void getAllProps(cv::VideoCapture &capture)
{
struct CamProp { int id; std::string desc; };
	std::vector<CamProp> piProps {
	{ cv::CAP_PROP_FRAME_WIDTH, "CAP_PROP_FRAME_WIDTH" },
	{ cv::CAP_PROP_FRAME_HEIGHT, "CAP_PROP_FRAME_HEIGHT" },
	{ cv::CAP_PROP_FPS, "CAP_PROP_FPS" },
	{ cv::CAP_PROP_FOURCC, "CAP_PROP_FOURCC" },
	{ cv::CAP_PROP_FORMAT, "CAP_PROP_FORMAT" },
	{ cv::CAP_PROP_MODE, "CAP_PROP_MODE" },
	{ cv::CAP_PROP_BRIGHTNESS, "CAP_PROP_BRIGHTNESS" },
	{ cv::CAP_PROP_CONTRAST, "CAP_PROP_CONTRAST" },
	{ cv::CAP_PROP_SATURATION, "CAP_PROP_SATURATION" },
	{ cv::CAP_PROP_EXPOSURE, "CAP_PROP_EXPOSURE" },
	{ cv::CAP_PROP_CONVERT_RGB, "CAP_PROP_CONVERT_RGB" },
	{ cv::CAP_PROP_AUTO_EXPOSURE, "CAP_PROP_AUTO_EXPOSURE" }
	};
	for( auto prop : piProps ) {
		if(prop.id == cv::CAP_PROP_FOURCC or prop.id == cv::CAP_PROP_MODE ) {
			union { char    c[5]; int     i; } myfourcc;
			myfourcc.i = capture.get(CV_CAP_PROP_FOURCC);
			myfourcc.c[4] = '\0';
			std::cout << prop.desc << ": " << myfourcc.c << std::endl;
		}
		else {
			std::cout << prop.desc << ":\t" << capture.get(prop.id) << std::endl;
		}
	}
}

int main(int argc, const char** argv)
{
	const char* intrinsic_file = default_intrinsic_file;
	if(argc > 1) intrinsic_file = argv[1];

	cv::Mat intrinsic, distortion;
	if(!readIntrinsics(intrinsic_file, intrinsic, distortion)) return -1;

	cv::Mat frame, hsv, filtered, display;

	cv::VideoCapture capture;
	for(;;) {
		capture.open("videotestsrc ! video/x-raw, width=640, height=480 ! appsin");
		capture.set(cv::CAP_PROP_CONVERT_RGB, 0);
		capture.set(cv::CAP_PROP_FRAME_WIDTH, frameSize.width);
		capture.set(cv::CAP_PROP_FRAME_HEIGHT, frameSize.height);
		capture.set(cv::CAP_PROP_FPS, 12);
		capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // Magic! 0.25 means manual exposure, 0.75 = auto
		capture.set(cv::CAP_PROP_EXPOSURE, 0.001);
		capture.set(cv::CAP_PROP_BRIGHTNESS, 0.5);
		capture.set(cv::CAP_PROP_CONTRAST, 0.5);
		capture.set(cv::CAP_PROP_SATURATION, 0.5);
		std::cerr << "Resolution: "<< capture.get(cv::CAP_PROP_FRAME_WIDTH)
				<< "x" << capture.get(cv::CAP_PROP_FRAME_HEIGHT)
				<< " FPS: " << capture.get(cv::CAP_PROP_FPS)
				<< std::endl;
		if(capture.isOpened()) break;
		std::cerr << "Couldn't connect to camera" << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(2));
	}
	getAllProps(capture);

#ifdef XGUI_ENABLED
	cv::namedWindow(display_window, cv::WINDOW_NORMAL);
#endif

	for(;;) {
		std::vector<long int> timer_values;
		std::vector<std::string> timer_names;
		timer_names.push_back("start"); timer_values.push_back(cv::getTickCount());
		capture >> frame;
		if (frame.empty()) {
			std::cerr << " Error reading from camera, empty frame." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}
		timer_names.push_back("captured"); timer_values.push_back(cv::getTickCount());

#ifdef XGUI_ENABLED
		cv::resize(frame, display, displaySize);
		timer_names.push_back("display resized"); timer_values.push_back(cv::getTickCount());
		for(size_t i=1; i < timer_values.size(); ++i) {
			long int val = timer_values[i] - timer_values[i-1];
			std::ostringstream osst;
			osst << timer_names[i] << ": " << val / cv::getTickFrequency();
			cv::putText(display, osst.str(), cv::Point(20,40+20*i), 0, 0.4, cv::Scalar(0,200,200));
		}

		cv::imshow(display_window, display);

		int key = cv::waitKey(2);
		if ((key & 255) == 27) break;
		if ((key & 255) == 's') cv::waitKey(0);
#endif
	}
	return 0;
}

