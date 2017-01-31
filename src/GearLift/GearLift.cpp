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
//static const cv::Size displaySize(1280, 720);
static const cv::Size displaySize(640, 360);
static const double displayRatio = double(displaySize.height) / frameSize.height;
static const char* detection_window = "Object Detection";
static const double MIN_AREA = 0.0002 * frameSize.height * frameSize.width;

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

void FindCorners(
		const std::vector<cv::Point> cont_one,
		const cv::Point2f center,
		const cv::Point2f reference,
		cv::Point &pointPos,
		cv::Point &pointNeg )
{
	double sqDistPos = 0;
	double sqDistNeg = 0;
	cv::Point2f dir((center.x-reference.x)/2, (center.y-reference.y)/2);
	for (cv::Point pnt : cont_one)
	{
		double dx = pnt.x - (reference.x + dir.x);
		double dy = pnt.y - (reference.y + dir.y);
		double sD = dx*dx + dy*dy;
		double cross = cv::Point2f(dx,dy).cross(dir);
		if (cross >= 0 and sD > sqDistPos) {
			sqDistPos = sD;
			pointPos = pnt;
		}
		if (cross < 0 and sD > sqDistNeg) {
			sqDistNeg = sD;
			pointNeg = pnt;
		}
	}

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
	static cv::Vec3i BlobLower( 0, 140,  96);
	static cv::Vec3i BlobUpper(15, 255, 255);
	static int dispMode = 2; // 0: none, 1: bw, 2: color

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
		switch(dispMode) {
		case 1:
			cv::resize(filtered, display, displaySize);
			break;
		case 2:
			cv::resize(frame, display, displaySize);
			break;
		}

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(filtered, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

		std::vector<cv::Point> cont_one, cont_two;
		cv::RotatedRect rect_one, rect_two;
		double biggest = 0, second = 0;
		for (std::vector<cv::Point> cont : contours)
		{
			cv::RotatedRect rect = cv::minAreaRect(cont);
			double ratio = rect.size.height / rect.size.width;
			double angle = fmod((ratio > 1.0 ? rect.angle : rect.angle + 90.0), 90.0);
			if (fabs(angle) >15.0) continue;

			double cont_area = cv::contourArea(cont);
			if (cont_area < MIN_AREA) continue;

			if (cont_area > biggest) {
				cont_two = cont_one;
				rect_two = rect_one;
				second = biggest;
				cont_one = cont;
				rect_one = rect;
				biggest = cont_area;
			}
			else if (cont_area > second) {
				second = cont_area;
				cont_two = cont;
				rect_two = rect;
			}
		}

		if (biggest > 0 && second > 0) {
			cv::Point corners[4];
			FindCorners(cont_one, rect_one.center, rect_two.center, corners[0], corners[1]);
			FindCorners(cont_two, rect_two.center, rect_one.center, corners[2], corners[3]);

			if (dispMode == 2) {
				cv::Point2f vtx1[4], vtx2[4];
				rect_one.points(vtx1);
				rect_two.points(vtx2);
				for( size_t i = 0; i < 4; i++ ) {
					cv::line(display, vtx1[i] * displayRatio, vtx1[(i+1)%4] * displayRatio, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
					cv::line(display, vtx2[i] * displayRatio, vtx2[(i+1)%4] * displayRatio, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
				}
				double angle = rect_one.size.height/rect_one.size.width > 1.0 ? rect_one.angle : rect_one.angle + 90.0;
				angle = fmod(angle, 90.0);
				std::ostringstream oss;
				oss << "cross";
				cv::putText(display, oss.str(), cv::Point(30,30) * displayRatio, 1, 2, cv::Scalar(0,255,0));
				cv::circle(display, corners[0]*displayRatio, 5, cv::Scalar(0,255,0), 2);
				cv::circle(display, corners[1]*displayRatio, 5, cv::Scalar(0,255,255), 2);
				cv::circle(display, corners[2]*displayRatio, 5, cv::Scalar(0,0,255), 2);
				cv::circle(display, corners[3]*displayRatio, 5, cv::Scalar(255,0,0), 2);
			}
		}

		if (dispMode > 0) {
			cv::imshow(detection_window, display);
		}

		int key = cv::waitKey(20);
		if ((key & 255) == 27) break;
		if ((key & 255) == 32) {
			if(++dispMode > 2) dispMode =0;
		}
		if ((key & 255) == 's') cv::waitKey(0);
	}
	return 0;
}

