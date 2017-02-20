#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <ctime>
#include "networktables/NetworkTable.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"

static const cv::Size frameSize(1280, 720);
static const double MIN_AREA = 0.0002 * frameSize.height * frameSize.width;
static const char* default_intrinsic_file = "jetson-camera-720.yml";

#ifdef XGUI_ENABLED
	#include "opencv2/highgui.hpp"
	static const cv::Size displaySize(640, 360);
	static const double displayRatio = double(displaySize.height) / frameSize.height;
	static const char* detection_window = "Object Detection";
#endif

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

void righten(cv::RotatedRect &other)
{
	if (other.size.height < other.size.width) {
		std::swap(other.size.height, other.size.width);
		other.angle += 90.0;
	}
	other.angle = fmod(other.angle, 180.0);
	if (other.angle >  90.0) other.angle -= 180;
	if (other.angle < -90.0) other.angle += 180;
}

float rate2rects(cv::RotatedRect one, cv::RotatedRect two)
{
	static const float perf_height = 5.0 / 8.25;
	static const float perf_width  = 2.0 / 8.25;
	float acc = fabs(sin(CV_PI*one.angle/180.0)) + fabs(sin(CV_PI*two.angle/180.0));
	cv::Point2f bar = two.center - one.center;
	if(bar.x == 0) return 999999;
	acc += fabs(bar.y/bar.x);
	float bar_len = cv::norm(bar);
	acc += fabs(one.size.height / bar_len - perf_height);
	acc += fabs(one.size.width  / bar_len - perf_width);
	acc += fabs(two.size.height / bar_len - perf_height);
	acc += fabs(two.size.width  / bar_len - perf_width);
	acc += fabs((one.size.height - two.size.height) / bar_len);
	return acc;
}

float rate3rects(cv::RotatedRect one, cv::RotatedRect two, cv::RotatedRect three)
{
	float acc = 0;
//	cv::Point2f pole = three.center - two.center;
//	if (pole.y == 0) return 99999;
//	acc += fabs(pole.x / pole.y);
	cv::Point2f pts[8];
	two.points(pts);
	three.points(pts+4);
	std::vector<cv::Point2f> new_points;
	new_points.reserve(8);
	for(size_t i=0; i<8; ++i) new_points.push_back(pts[i]);
	cv::RotatedRect other = cv::minAreaRect(new_points);
	righten(other);
	acc += rate2rects(one, other);
	return acc;
}

std::string date_now()
{
    std::time_t result = std::time(nullptr);
    return std::string(std::asctime(std::localtime(&result)));
}

bool readIntrinsics(const char *filename, cv::Mat &intrinsic, cv::Mat &distortion)
{
	cv::FileStorage fs( filename, cv::FileStorage::READ );
	if( !fs.isOpened() )
	{
		std::cerr << date_now() << " Error: Couldn't open intrinsic parameters file "
				<< filename << std::endl;
		return false;
	}
	fs["camera_matrix"] >> intrinsic;
	fs["distortion_coefficients"] >> distortion;
	if( intrinsic.empty() || distortion.empty() )
	{
		std::cerr <<date_now() << " Error: Couldn't load intrinsic parameters from "
				<< filename << std::endl;
		return false;
	}
	fs.release();
	return true;
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
	static cv::Vec3i BlobLower(66, 200,  30);
	static cv::Vec3i BlobUpper(94, 255, 255);
	static int dispMode = 0; // 0: none, 1: bw, 2: color

	cv::Vec3d camera_offset(-13.0, -4.0, 0.0);

	static std::vector<cv::Point3f> realPoints;
	realPoints.push_back(cv::Point3f(-5.125,-2.5, 10.5)); // Top left
	realPoints.push_back(cv::Point3f(-5.125, 2.5, 10.5)); // Bottom Left
	realPoints.push_back(cv::Point3f( 5.125,-2.5, 10.5)); // Top right
	realPoints.push_back(cv::Point3f( 5.125, 2.5, 10.5)); // Bottom right

	cv::VideoCapture capture;
	for(;;) {
		std::ostringstream capturePipe;
#ifdef ICS_CAMERA_PRESENT
		capturePipe << "nvcamerasrc ! video/x-raw(memory:NVMM)"
			<< ", width=(int)" << frameSize.width
			<< ", height=(int)" << frameSize.height
			<< ", format=(string)I420, framerate=(fraction)30/1 ! "
			<< "nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! "
			<< "videoconvert ! video/x-raw, format=(string)BGR ! appsink";
#else
		capturePipe << "NoCSIcamera";
#endif
		if(!capture.open(capturePipe.str())) {
			capture.open(1);
			capture.set(cv::CAP_PROP_FRAME_WIDTH, frameSize.width);
			capture.set(cv::CAP_PROP_FRAME_HEIGHT, frameSize.height);
			capture.set(cv::CAP_PROP_FPS, 7.5);
			capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // Magic! 0.25 means manual exposure, 0.75 = auto
			capture.set(cv::CAP_PROP_EXPOSURE, 0);
			capture.set(cv::CAP_PROP_BRIGHTNESS, 0.5);
			capture.set(cv::CAP_PROP_CONTRAST, 0.5);
			capture.set(cv::CAP_PROP_SATURATION, 0.5);
		}
		if(capture.isOpened()) break;

		std::cerr << date_now() << " Couldn't connect to camera.. sleeping." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(5));
	}
	std::cerr << date_now() << " Camera connected. Resolution: "<< capture.get(cv::CAP_PROP_FRAME_WIDTH)
						<< "x" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

#ifdef XGUI_ENABLED
	cv::namedWindow(detection_window, cv::WINDOW_NORMAL);
	cv::createTrackbar("Lo H",detection_window, &BlobLower[0], 255);
	cv::createTrackbar("Hi H",detection_window, &BlobUpper[0], 255);
	cv::createTrackbar("Lo S",detection_window, &BlobLower[1], 255);
	cv::createTrackbar("Hi S",detection_window, &BlobUpper[1], 255);
	cv::createTrackbar("Lo V",detection_window, &BlobLower[2], 255);
	cv::createTrackbar("Hi V",detection_window, &BlobUpper[2], 255);
#endif

	int elemSize(5);
	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(elemSize+1,elemSize+1));
	cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, gpu1.type(), element);
	cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, gpu1.type(), element);

	gpu1.create(frameSize, CV_8UC1);
	gpu2.create(frameSize, CV_8UC1);

	for(;;) {
		capture >> frame;
		if (frame.empty()) {
			std::cerr << date_now() << " Error reading from camera, empty frame." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}
		gpuC.upload(frame);
		cv::cuda::cvtColor(gpuC, gpuC, CV_BGR2HSV);
		CheezyInRange(gpuC, BlobLower, BlobUpper, gpu1);
		erode->apply(gpu1, gpu2);
		dilate->apply(gpu2, gpu1);

		gpu1.download(filtered);

#ifdef XGUI_ENABLED
		switch(dispMode) {
		case 1:
			cv::resize(filtered, display, displaySize);
			break;
		case 2:
			cv::resize(frame, display, displaySize);
			break;
		}
#endif

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
			cv::RotatedRect lRect, rRect;
			std::vector<cv::Point> lCont, rCont;
			float mscore = 999999;

			if(biggest[2] > 0) {
				for(size_t i=0; i < 3; ++i) {
					float score = rate2rects(rect_one[i], rect_one[(i+1)%3]);
					if(score < mscore) {
						mscore = score;
						lRect = rect_one[i];
						lCont = cont_one[i];
						rRect = rect_one[(i+1)%3];
						rCont = cont_one[(i+1)%3];
					}
				}
				for(size_t i=0; i < 3; ++i) {
					float score = rate3rects(rect_one[i], rect_one[(i+1)%3], rect_one[(i+2)%3]);
					if(score < mscore) {
						mscore = score;
						lRect = rect_one[i];
						lCont = cont_one[i];
						rCont = cont_one[(i+1)%3];
						rCont.reserve(rCont.size() + cont_one[(i+2)%3].size());
						rCont.insert(rCont.end(), cont_one[(i+2)%3].begin(), cont_one[(i+2)%3].end());
						rRect = cv::minAreaRect(rCont);
						righten(rRect);
					}
				}
			}
			else {
				lRect = rect_one[0];
				lCont = cont_one[0];
				rRect = rect_one[1];
				rCont = cont_one[1];
			}

			if (lRect.center.x > rRect.center.x) {
				std::swap(lRect, rRect);
				std::swap(lCont, rCont);
			}

			std::vector<cv::Point> lCorn = FindCorners2(lCont, lRect,  1);
			std::vector<cv::Point> rCorn = FindCorners2(rCont, rRect, -1);
			std::vector<cv::Point2f> imagePoints;
			imagePoints.push_back(lCorn[0]);
			imagePoints.push_back(lCorn[1]);
			imagePoints.push_back(rCorn[0]);
			imagePoints.push_back(rCorn[1]);

			cv::Vec3d rvec, tvec, lvec;
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
			// tvec is where the target is in the camera coordinates
			// We offset it with the camera_offset vector and rotate opposite to the target's rotation
			lvec = rmat.t() * -(tvec + camera_offset);

			table->PutNumber("Peg Crossrange", lvec[0]);
			table->PutNumber("Peg Downrange", -lvec[2]); // Robot is in negative Z area. We expect positive down range
			table->PutNumber("Peg Yaw", atan2(tvec[0],tvec[2]));

#ifdef XGUI_ENABLED
			cv::Point dispTarget = cv::Point(
					0.5*displaySize.width  + (displaySize.height/150)*tvec[0],
					0.9*displaySize.height - (displaySize.height/150)*tvec[2]
					);
			cv::Vec3d peg = rmat.t() * (tvec + camera_offset);
			cv::Point peg2D = displaySize.height/10 * (cv::Point2d(peg[0],peg[2]) / cv::norm(peg));
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
						dispTarget - peg2D,
						cv::Scalar(0,0,255));
				std::ostringstream oss;
				oss << "Yaw:  " << 180.0*atan2(tvec[0],tvec[2])/CV_PI << " (" << rvec[0] << " : " << rvec[1] << " : " << rvec[2] << ")";
				cv::putText(display, oss.str(), cv::Point(20,20), 0, 0.33, cv::Scalar(0,200,200));
				std::ostringstream oss1;
				oss1 << "Downrange: " << lvec[2];
				cv::putText(display, oss1.str(), cv::Point(20,40), 0, 0.33, cv::Scalar(0,200,200));
				std::ostringstream oss2;
				oss2 << "Crossrange: " << lvec[0];
				cv::putText(display, oss2.str(), cv::Point(20,60), 0, 0.33, cv::Scalar(0,200,200));
			}
#endif
		}

#ifdef XGUI_ENABLED
		if (dispMode > 0) {
			cv::imshow(detection_window, display);
		}

		int key = cv::waitKey(20);
		if ((key & 255) == 27) break;
		if ((key & 255) == 32) {
			if(++dispMode > 2) dispMode =0;
		}
		if ((key & 255) == 's') cv::waitKey(0);
#endif
	}
	return 0;
}

