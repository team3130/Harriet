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
	cv::cuda::GpuMat gpu1, gpu2;
	static cv::Vec3i BlobLower(203,  38, 130);
	static cv::Vec3i BlobUpper(244, 239, 233);
	static int elemSize(7);
	frame = cv::imread("LoadingStationBlue.png");
	cv::namedWindow("Object Detection", cv::WINDOW_NORMAL);
	cv::createTrackbar("Low H","Object Detection", &BlobLower[0], 255);
	cv::createTrackbar("High H","Object Detection", &BlobUpper[0], 255);
	cv::createTrackbar("Low S","Object Detection", &BlobLower[1], 255);
	cv::createTrackbar("High S","Object Detection", &BlobUpper[1], 255);
	cv::createTrackbar("Low V","Object Detection", &BlobLower[2], 255);
	cv::createTrackbar("High V","Object Detection", &BlobUpper[2], 255);
	cv::createTrackbar("Elem Size","Object Detection", &elemSize, 63);
	cv::imshow("Test", frame);
	for(;;) {
		cv::inRange(frame, BlobLower, BlobUpper, filtered);
		cv::resize(filtered, display, cv::Size(640, 360));

		gpu1.upload(filtered);
		cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(elemSize+1,elemSize+1));
		cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, gpu1.type(), element);
		cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, gpu1.type(), element);
		erode->apply(gpu1, gpu2);
		dilate->apply(gpu2, gpu1);

		// Return back to the CPU
		gpu1.download(filtered);

		cv::imshow("Object Detection", filtered);
		int key = cv::waitKey(20);
		if ((key & 255) == 27) break;
	}
	return 0;
}

