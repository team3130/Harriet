#include <iostream>
#include <memory>
#include "networktables/NetworkTable.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"

static const cv::Size frameSize(1280, 720);
static const cv::Size displaySize(640, 360);
static const double displayRatio = double(displaySize.height) / frameSize.height;
static const char* detection_window = "Object Detection";
static const double MIN_AREA = 0.0002 * frameSize.height * frameSize.width;
static const char* default_intrinsic_file = "jetson-camera-720.yml";

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

void righten(cv::RotatedRect &rectangle)
{
	if (rectangle.size.height < rectangle.size.width) {
		std::swap(rectangle.size.height, rectangle.size.width);
		rectangle.angle += 90.0;
	}
	rectangle.angle = fmod(rectangle.angle, 180.0);
	if (rectangle.angle >  90.0) rectangle.angle -= 180;
	if (rectangle.angle < -90.0) rectangle.angle += 180;
}

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

void FindMidPoints(std::vector<cv::Point> upCont, std::vector<cv::Point> dnCont, std::vector<cv::Point2f> &imagePoints)
{
	std::vector<cv::Point> new_points = upCont;
	new_points.reserve(upCont.size()+dnCont.size());
	for(size_t i=0; i<dnCont.size(); ++i) new_points.push_back(dnCont[i]);
	cv::RotatedRect big = cv::minAreaRect(new_points);
	if(fabs(big.angle) > 45 and fabs(big.angle) < 135) {
		std::swap(big.size.height, big.size.width);
		big.angle = fmod(big.angle, 180.0);
		if (big.angle >  90.0) big.angle -= 180;
		if (big.angle < -90.0) big.angle += 180;
	}
	imagePoints.push_back(cv::Point(0,0));
	imagePoints.push_back(cv::Point(0,0));
	imagePoints.push_back(cv::Point(0,0));
	imagePoints.push_back(cv::Point(0,0));
}

int main(int argc, const char** argv)
{
	const char* intrinsic_file = default_intrinsic_file;
	if(argc > 1) intrinsic_file = argv[1];

	cv::Mat intrinsic, distortion;
	if(!readIntrinsics(intrinsic_file, intrinsic, distortion)) return -1;

	NetworkTable::SetClientMode();
	NetworkTable::SetTeam(3130);
	std::shared_ptr<NetworkTable> table = NetworkTable::GetTable("/Jetson");

	cv::Mat frame, filtered, display;
	cv::cuda::GpuMat gpuC, gpu1, gpu2;
	static cv::Vec3i BlobLower(24, 128,  55);
	static cv::Vec3i BlobUpper(48, 255, 255);
	static int dispMode = 2; // 0: none, 1: bw, 2: color

	cv::Vec3d camera_offset(-7.0, -4.0, -12);

	static std::vector<cv::Point3f> realPoints;
	realPoints.push_back(cv::Point3f(-5.125,-2.5, 10.5)); // Top left
	realPoints.push_back(cv::Point3f(-5.125, 2.5, 10.5)); // Bottom Left
	realPoints.push_back(cv::Point3f( 5.125,-2.5, 10.5)); // Top right
	realPoints.push_back(cv::Point3f( 5.125, 2.5, 10.5)); // Bottom right

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

	int elemSize(5);
	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(elemSize+1,elemSize+1));
	cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, gpu1.type(), element);
	cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, gpu1.type(), element);

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

		std::vector<cv::Point> cont_one[3];
		cv::RotatedRect rect_one[3];
		float biggest[3] = {0,0,0};
		for (std::vector<cv::Point> cont : contours)
		{
			cv::RotatedRect rect = cv::minAreaRect(cont);
			float area = rect.size.height * rect.size.width;
			if (area < MIN_AREA) continue;
			righten(rect);

			if (area > biggest[0]) {
				cont_one[2] = cont_one[1];
				cont_one[1] = cont_one[0];
				cont_one[0] = cont;
				rect_one[2] = rect_one[1];
				rect_one[1] = rect_one[0];
				rect_one[0] = rect;
				biggest[2] = biggest[1];
				biggest[1] = biggest[0];
				biggest[0] = area;
			}
			else if (area > biggest[1]) {
				cont_one[2] = cont_one[1];
				cont_one[1] = cont;
				rect_one[2] = rect_one[1];
				rect_one[1] = rect;
				biggest[2] = biggest[1];
				biggest[1] = area;
			}
			else if (area > biggest[2]) {
				cont_one[2] = cont;
				rect_one[2] = rect;
				biggest[2] = area;
			}
		}

		if (biggest[0] > 0 && biggest[1] > 0) {
			cv::RotatedRect upRect, dnRect;
			std::vector<cv::Point> upCont, dnCont;
			float mscore = 999999;

			upRect = rect_one[0];
			upCont = cont_one[0];
			dnRect = rect_one[1];
			dnCont = cont_one[1];

			if (upRect.center.x > dnRect.center.x) {
				std::swap(upRect, dnRect);
				std::swap(upCont, dnCont);
			}

			std::vector<cv::Point2f> imagePoints;
			FindMidPoints(upCont, dnCont, imagePoints);

			cv::Vec3d rvec, tvec;
			cv::Matx33d rmat;

			cv::solvePnP(
					realPoints,       // 3-d points in object coordinate
					imagePoints,        // 2-d points in image coordinates
					intrinsic,           // Our camera matrix
					distortion,
					rvec,                // Output rotation *vector*.
					tvec                 // Output translation vector.
			);
			cv::Rodrigues(rvec, rmat);
			cv::Point dispTarget = cv::Point(
					0.5*displaySize.width  + (displaySize.height/150)*tvec[0],
					0.9*displaySize.height - (displaySize.height/150)*tvec[2]
					);

			table->PutNumber("Boiler Distance", cv::norm(tvec));
			table->PutNumber("Boiler Yaw", 180.0*atan2(tvec[0],tvec[2])/CV_PI);

			if (dispMode == 2) {
				cv::circle(display, imagePoints[0]*displayRatio, 8, cv::Scalar(0,125,255), 1);
				cv::circle(display, imagePoints[1]*displayRatio, 8, cv::Scalar(0,0,255), 1);
				cv::circle(display, imagePoints[0]*displayRatio, 8, cv::Scalar(125,255,0), 1);
				cv::circle(display, imagePoints[1]*displayRatio, 8, cv::Scalar(0,255,0), 1);

				cv::line(display,
						dispTarget,
						cv::Point(displaySize.width/2,displaySize.height*0.9),
						cv::Scalar(0,255,255));
				std::ostringstream oss;
				oss << "Yaw: " << 180.0*atan2(tvec[0],tvec[2])/CV_PI;
				cv::putText(display, oss.str(), cv::Point(20,20), 0, 0.33, cv::Scalar(0,200,200));
				std::ostringstream oss1;
				oss1 << "Distance: " << cv::norm(tvec);
				cv::putText(display, oss1.str(), cv::Point(20,40), 0, 0.33, cv::Scalar(0,200,200));
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

