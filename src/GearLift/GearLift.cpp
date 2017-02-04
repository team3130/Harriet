#include <iostream>
#include <thread>
#include <chrono>
#include "networktables/NetworkTable.h"
#include "opencv2/opencv.hpp"
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
static const double MAX_TILT = 15.0;
static const char* calibration_file = "jetson-camera-720.yml";

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

std::vector<cv::Point> FindCorners2(
		const std::vector<cv::Point> cont_one,
		cv::RotatedRect box,
		double dir=1)
{
	std::vector<cv::Point> ret(2, cv::Point(0,0));
	double maxLo = 0;
	double maxHi = 0;
	double r = dir * (box.size.height*box.size.height + box.size.width*box.size.width) / (2 * box.size.width);
	double alpha = CV_PI * box.angle / 180.0;
	cv::Matx22f rot(cos(alpha),-sin(alpha),sin(alpha),cos(alpha));
	cv::Point2f refHi = rot * cv::Point2f(r,  box.size.height/2) + box.center;
	cv::Point2f refLo = rot * cv::Point2f(r, -box.size.height/2) + box.center;
	for (cv::Point pnt : cont_one)
	{
		cv::Point2f lo = cv::Point2f(pnt) - refLo;
		double sLo = cv::norm(lo);
		if (sLo > maxLo) {
			maxLo = sLo;
			ret[1] = pnt;
		}

		cv::Point2f hi = cv::Point2f(pnt) - refHi;
		double sHi = cv::norm(hi);
		if (sHi > maxHi) {
			maxHi = sHi;
			ret[0] = pnt;
		}
	}
	return ret;
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
	static cv::Vec3i BlobLower(33,  56,  78);
	static cv::Vec3i BlobUpper(48, 255, 255);
	static int dispMode = 2; // 0: none, 1: bw, 2: color

	static std::vector<cv::Point3f> realPoints;
	realPoints.push_back(cv::Point3f(-5.125,-2.5, 10.5)); // Top left
	realPoints.push_back(cv::Point3f(-5.125, 2.5, 10.5)); // Bottom Left
	realPoints.push_back(cv::Point3f( 5.125,-2.5, 10.5)); // Top right
	realPoints.push_back(cv::Point3f( 5.125, 2.5, 10.5)); // Bottom right

	cv::FileStorage fs( calibration_file, cv::FileStorage::READ );
	cv::Mat         intrinsic, distortion;
	if( !fs.isOpened() )
	{
		std::cerr << "Error: Couldn't open intrinsic parameters file "
				<< calibration_file << std::endl;
		return -1;
	}
	fs["camera_matrix"] >> intrinsic;
	fs["distortion_coefficients"] >> distortion;
	if( intrinsic.empty() || distortion.empty() )
	{
		std::cerr << "Error: Couldn't load intrinsic parameters from "
				<< calibration_file << std::endl;
		return -1;
	}
	fs.release();

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

			if (rect.size.height < rect.size.width) {
				std::swap(rect.size.height, rect.size.width);
				rect.angle += 90.0;
			}
			rect.angle = fmod(rect.angle, 90.0);
			if (fabs(rect.angle) > MAX_TILT) continue;

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
			if (rect_one.center.x > rect_two.center.x) {
				std::swap(rect_one, rect_two);
				std::swap(cont_one, cont_two);
				std::swap(biggest, second);
			}

			std::vector<cv::Point> lCorn = FindCorners2(cont_one, rect_one, 1);
			std::vector<cv::Point> rCorn = FindCorners2(cont_two, rect_two, -1);
			std::vector<cv::Point2f> imagePoints;
			imagePoints.push_back(lCorn[0]);
			imagePoints.push_back(lCorn[1]);
			imagePoints.push_back(rCorn[0]);
			imagePoints.push_back(rCorn[1]);

			cv::Mat rvec, tvec;
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
			cv::Point3d peg = rmat * cv::Point3d(0,0,displaySize.height/10);
			cv::Point dispTarget = cv::Point(
					0.5*displaySize.width  + (displaySize.height/150)*tvec.at<double>(0),
					0.9*displaySize.height - (displaySize.height/150)*tvec.at<double>(2)
					);
			cv::Point peg2D = cv::Point(peg.x,-peg.z);

			if (dispMode == 2) {
				cv::line(display, imagePoints[0] * displayRatio, imagePoints[1] * displayRatio, cv::Scalar(200, 0, 255), 1, cv::LINE_AA);
				cv::line(display, imagePoints[1] * displayRatio, imagePoints[3] * displayRatio, cv::Scalar(200, 0, 255), 1, cv::LINE_AA);
				cv::line(display, imagePoints[3] * displayRatio, imagePoints[2] * displayRatio, cv::Scalar(200, 0, 255), 1, cv::LINE_AA);
				cv::line(display, imagePoints[2] * displayRatio, imagePoints[0] * displayRatio, cv::Scalar(200, 0, 255), 1, cv::LINE_AA);

				cv::circle(display, lCorn[0]*displayRatio, 8, cv::Scalar(0,125,255), 1);
				cv::circle(display, lCorn[1]*displayRatio, 8, cv::Scalar(0,0,255), 1);
				cv::circle(display, rCorn[0]*displayRatio, 8, cv::Scalar(125,255,0), 1);
				cv::circle(display, rCorn[1]*displayRatio, 8, cv::Scalar(0,255,0), 1);

				cv::line(display,
						dispTarget,
						cv::Point(displaySize.width/2,displaySize.height*0.9),
						cv::Scalar(0,255,255));
				cv::line(display,
						dispTarget,
						dispTarget+peg2D,
						cv::Scalar(0,0,255));
				std::ostringstream oss;
				oss << cv::norm(tvec) << "      ";
				cv::putText(display, oss.str(), dispTarget, cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,200,200),1);
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

