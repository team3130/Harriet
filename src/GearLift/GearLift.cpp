#include <iostream>
#include <thread>
#include <chrono>
#include "networktables/NetworkTable.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"

static const cv::Size frameSize(1280, 720);
static const cv::Size displaySize(640, 360);
static const char* detection_window = "Object Detection";

void CheezyInRange(cv::cuda::GpuMat src, cv::Vec3i BlobLower, cv::Vec3i BlobUpper, cv::cuda::GpuMat dst) {
	cv::cuda::GpuMat channels[3];
	cv::cuda::split(src, channels);
	//threshold, reset to zero everything that is above the upper limit
	cv::cuda::threshold(channels[0], channels[0], BlobUpper[0], 255, cv::THRESH_TOZERO_INV);
	cv::cuda::threshold(channels[1], channels[1], BlobUpper[1], 255, cv::THRESH_TOZERO_INV);
	cv::cuda::threshold(channels[2], channels[2], BlobUpper[2], 255, cv::THRESH_TOZERO_INV);
	//threshold, reset to zero what is below the lower limit, otherwise to 255
	cv::cuda::threshold(channels[0], channels[0], BlobLower[0], 255, cv::THRESH_BINARY);
	cv::cuda::threshold(channels[1], channels[1], BlobLower[1], 255, cv::THRESH_BINARY);
	cv::cuda::threshold(channels[2], channels[2], BlobLower[2], 255, cv::THRESH_BINARY);
	//combine all three channels and collapse them into one B/W image (to channels[0])
	cv::cuda::bitwise_and(channels[0], channels[1], channels[0]);
	cv::cuda::bitwise_and(channels[0], channels[2], dst);
}

int main()
{
#if 0
	NetworkTable::SetClientMode();
	NetworkTable::SetTeam(3130);
	std::shared_ptr<NetworkTable> table = NetworkTable::GetTable("/Jetson");
	std::this_thread::sleep_for(std::chrono::seconds(2));
	std::cout << table->GetString("Command", "Nothing") << std::endl;
	std::this_thread::sleep_for(std::chrono::seconds(2));
	std::cout << table->GetString("Command", "Nothing") << std::endl;
#endif
	cv::Mat frame, filtered, display;
	cv::cuda::GpuMat gpuC, gpu1, gpu2;
	static cv::Vec3i BlobLower(203,  38, 130);
	static cv::Vec3i BlobUpper(244, 239, 233);

	cv::VideoCapture capture;
	std::ostringstream capturePipe;
	capturePipe << "nvcamerasrc ! video/x-raw(memory:NVMM)"
		<< ", width=(int)" << frameSize.width
		<< ", height=(int)" << frameSize.height
		<< ", format=(string)I420, framerate=(fraction)30/1 ! "
		<< "nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! "
		<< "videoconvert ! video/x-raw, format=(string)BGR ! appsink";
	if(!capture.open(capturePipe.str())) {
		capture.open(0);
		capture.set(cv::CAP_PROP_FRAME_WIDTH, frameSize.width);
		capture.set(cv::CAP_PROP_FRAME_HEIGHT, frameSize.height);
		std::cerr << "Resolution: "<< capture.get(cv::CAP_PROP_FRAME_WIDTH)
			<< "x" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
	}
	if(!capture.isOpened()) {
		std::cerr << "Couldn't connect to camera" << std::endl;
		return 1;
	}

	cv::namedWindow(detection_window, cv::WINDOW_NORMAL);
	cv::createTrackbar("Lo H",detection_window, &BlobLower[0], 255);
	cv::createTrackbar("Hi H",detection_window, &BlobUpper[0], 255);
	cv::createTrackbar("Lo S",detection_window, &BlobLower[1], 255);
	cv::createTrackbar("Hi S",detection_window, &BlobUpper[1], 255);
	cv::createTrackbar("Lo V",detection_window, &BlobLower[2], 255);
	cv::createTrackbar("Hi V",detection_window, &BlobUpper[2], 255);

	int elemSize(7);
	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(elemSize+1,elemSize+1));
	cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, gpu1.type(), element);
	cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, gpu1.type(), element);

	gpu1.create(frameSize, CV_8UC1);
	gpu2.create(frameSize, CV_8UC1);

	for(;;) {
		capture >> frame;
		if (frame.empty()) {
			std::cerr << " Error reading from camera, empty frame." << std::endl;
			if(cv::waitKey(5 * 1000) > 0) break;
			continue;
		}
		gpuC.upload(frame);
		cv::cuda::cvtColor(gpuC, gpuC, CV_BGR2HSV);
		CheezyInRange(gpuC, BlobLower, BlobUpper, gpu1);
		erode->apply(gpu1, gpu2);
		dilate->apply(gpu2, gpu1);

		gpu1.download(filtered);
		cv::resize(filtered, display, displaySize);

		cv::imshow(detection_window, display);
		int key = cv::waitKey(20);
		if ((key & 255) == 27) break;
	}
	return 0;
}

