#define XGUI_ENABLED

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <ctime>
#include "networktables/NetworkTable.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

static const cv::Size frameSize(640,480);
static const double cameraFPS = 8;
static const double MIN_AREA = 0.0002 * frameSize.height * frameSize.width;
static const double BOILER_TAPE_RATIO = 2.5;
static const double BOILER_TAPE_RATIO2 = BOILER_TAPE_RATIO/2;
static const char* default_intrinsic_file = "jetson-camera-720.yml";
static const double CAMERA_GOAL_HEIGHT = 69; //!<- Tower height is 97" and the camera is 19" above the floor
static const double CAMERA_ZERO_DIST = 130; //!<- Tower height is 97" and the camera is 12" above the floor

#ifdef XGUI_ENABLED
	#include "opencv2/highgui.hpp"
	static const cv::Size displaySize(320, 240);
	static const double displayRatio = double(displaySize.height) / frameSize.height;
	static const char* detection_window = "Object Detection";
#endif

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

struct RingRelation {
	double rating;
	std::vector<cv::Point> *my_cont;
	std::vector<cv::Point> *other_cont;
	cv::RotatedRect my_rect;
	cv::RotatedRect other_rect;
	double rate2rings(const RingRelation &other)
	{
		double rate = 0;
		rate += fabs(my_rect.angle)/90.0; // angle is in (-90,+90), 90 is the best
		rate += 1.0 - fabs(BOILER_TAPE_RATIO - my_rect.size.height/my_rect.size.width);
		rate += 1.0 - fabs((my_rect.center.x-other_rect.center.x) / (my_rect.center.y-other_rect.center.y));
		rate += 1.0 - fabs(BOILER_TAPE_RATIO2 - my_rect.size.height / norm(my_rect.center-other_rect.center));
		return rate;
	};
	RingRelation(std::vector<cv::Point> *cont, cv::RotatedRect rect, const std::vector<RingRelation> &chain)
		: rating(0), my_cont(cont), my_rect(rect), other_cont(cont), other_rect(rect)
	{
		for(auto&& other : chain) {
			double temp = rate2rings(other);
			if(temp > rating) {
				rating = temp;
				other_rect = other.my_rect;
				other_cont = other.my_cont;
			}
		}
	};
	// This comparison operator is for sort. Reversed for descending order.
	bool operator<(RingRelation &other) { return rating > other.rating; };
};


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
		std::cerr << date_now() << " Error: Couldn't open intrinsic parameters file "
				<< filename << std::endl;
		return false;
	}
	fs["camera_matrix"] >> intrinsic;
	fs["distortion_coefficients"] >> distortion;
	if( intrinsic.empty() || distortion.empty() )
	{
		std::cerr << date_now() << " Error: Couldn't load intrinsic parameters from "
				<< filename << std::endl;
		return false;
	}
	fs.release();
	return true;
}

cv::Point2d intersect(cv::Point2d pivot, cv::Matx22d rotation, cv::Point one, cv::Point two)
{
	cv::Point2d ret(-1,-1);
	cv::Point2d one_f(one), two_f(two);
	cv::Vec2d one_v = rotation * (one_f - pivot);
	cv::Vec2d two_v = rotation * (two_f - pivot);
	if(one_v[0] > 0 and two_v[0] > 0) return ret;
	if(one_v[0] < 0 and two_v[0] < 0) return ret;
	if(one_v[0] == 0) return one_f;
	if(two_v[0] == 0) return two_f;
	double y0 = two_v[1] - two_v[0] * (one_v[1]-two_v[1])/(one_v[0]-two_v[0]);
	ret = rotation.t() * cv::Point2d(0, y0) + pivot;
	return ret;
}

bool compPoints(const cv::Point2d a, const cv::Point2d b) { return (a.y < b.y); }

void FindMidPoints(std::vector<cv::Point> *upCont, std::vector<cv::Point> *dnCont, std::vector<cv::Point2d> &imagePoints)
{
	std::vector<cv::Point> new_points = *upCont;
	new_points.reserve(upCont->size()+dnCont->size());
	for(size_t i=0; i<dnCont->size(); ++i) new_points.push_back((*dnCont)[i]);
	cv::RotatedRect big = cv::minAreaRect(new_points);
	if(fabs(big.angle) > 45 and fabs(big.angle) < 135) {
		std::swap(big.size.height, big.size.width);
		big.angle = fmod(big.angle+90.0, 180.0);
		if (big.angle >  90.0) big.angle -= 180;
		if (big.angle < -90.0) big.angle += 180;
	}
	double cosT = cos(CV_PI*big.angle/180.0);
	double sinT = -sin(CV_PI*big.angle/180.0); // RotatedRect::angle is clockwise, so negative
	cv::Matx22d rmat(cosT, -sinT, sinT, cosT);

	std::vector<cv::Point2d> pointsUp;
	for(size_t i = 0; i < upCont->size(); ++i) {
		cv::Point2d point = intersect(big.center, rmat, (*upCont)[i], (*upCont)[(i+1)%upCont->size()]);
		if(point != cv::Point2d(-1,-1)) pointsUp.push_back(point);
	}
	if(pointsUp.size() > 1) {
		std::sort(pointsUp.begin(), pointsUp.end(), compPoints);
		imagePoints.push_back(pointsUp.front());
		imagePoints.push_back(pointsUp.back());
	}

	std::vector<cv::Point2d> pointsDn;
	for(size_t i = 0; i < dnCont->size(); ++i) {
		cv::Point2d point = intersect(big.center, rmat, (*dnCont)[i], (*dnCont)[(i+1)%dnCont->size()]);
		if(point != cv::Point2d(-1,-1)) pointsDn.push_back(point);
	}
	if(pointsDn.size() > 1) {
		std::sort(pointsDn.begin(), pointsDn.end(), compPoints);
		imagePoints.push_back(pointsDn.front());
		imagePoints.push_back(pointsDn.back());
	}
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
	std::shared_ptr<NetworkTable> preferences = NetworkTable::GetTable("/Preferences");

	cv::Mat frame, hsv, filtered, display;
	static cv::Vec3i BlobLower(66, 200,  30);
	static cv::Vec3i BlobUpper(94, 255, 255);
	static int dispMode = 2; // 0: none, 1: bw, 2: color

	cv::Vec3d camera_offset(-7.0, -4.0, -12);

	static std::vector<cv::Point3d> realPoints;
	realPoints.push_back(cv::Point3d(0,-4, 0));
	realPoints.push_back(cv::Point3d(0, 0, 0.3));
	realPoints.push_back(cv::Point3d(0, 4, 0.3));
	realPoints.push_back(cv::Point3d(0, 6, 0));

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

#ifdef XGUI_ENABLED
	cv::namedWindow(detection_window, cv::WINDOW_NORMAL);
	cv::createTrackbar("Lo H",detection_window, &BlobLower[0], 255);
	cv::createTrackbar("Hi H",detection_window, &BlobUpper[0], 255);
	cv::createTrackbar("Lo S",detection_window, &BlobLower[1], 255);
	cv::createTrackbar("Hi S",detection_window, &BlobUpper[1], 255);
	cv::createTrackbar("Lo V",detection_window, &BlobLower[2], 255);
	cv::createTrackbar("Hi V",detection_window, &BlobUpper[2], 255);
#endif

	// Initialize an element (reusable) for morphology filters
	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4,4));

	for(;;) {
		// System timer always runs so we can see the coprocessor is online
		// and can compare the last measurement time against the current time.
		table->PutNumber("Boiler Sys Time", cv::getTickCount()/cv::getTickFrequency());

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
		erode(filtered, filtered, element);
		dilate(filtered, filtered, element);
		timer_names.push_back("de-noise applied"); timer_values.push_back(cv::getTickCount());

#ifdef XGUI_ENABLED
		switch(dispMode) {
		case 1:
			cv::resize(filtered, display, displaySize);
			break;
		case 2:
			cv::resize(frame, display, displaySize);
			break;
		}
		timer_names.push_back("display resized"); timer_values.push_back(cv::getTickCount());
#endif

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(filtered, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		timer_names.push_back("contours found"); timer_values.push_back(cv::getTickCount());

		std::vector<RingRelation> graph;
		for (auto&& cont : contours)
		{
			cv::RotatedRect rect = cv::minAreaRect(cont);
			double area = rect.size.height * rect.size.width;
			if (area < MIN_AREA) continue;
			righten(rect);
			if (fabs(rect.angle) < 45) continue;
			graph.push_back(RingRelation(&cont, rect, graph));
		}
		std::sort(graph.begin(), graph.end());

		if (graph.size() > 1) {
			double distance, yaw;

			std::vector<cv::Point2d> imagePoints;
			if (graph[0].my_rect.center.y < graph[1].my_rect.center.y) {
				FindMidPoints(graph[0].my_cont, graph[1].my_cont, imagePoints);
			}
			else {
				FindMidPoints(graph[1].my_cont, graph[0].my_cont, imagePoints);
			}


			cv::Vec3d rvec, tvec;
			if(imagePoints.size() == 4) {
				std::vector<cv::Point2d> undistortedPoints;
				cv::undistortPoints(imagePoints, undistortedPoints, intrinsic, distortion, cv::noArray(), intrinsic);
				double cam_tilt = preferences->GetNumber("Front Camera Tilt", 40);
				cv::Vec3d cam_offset(-18, 0, -18);
				double cam_cos = cos(CV_PI*cam_tilt/180.0);
				double dee = cv::norm(undistortedPoints[0] - undistortedPoints[3]);
				distance = cam_cos * intrinsic.at<double>(1,1) * fabs(realPoints[3].y-realPoints[0].y) / dee;

				double m_zenith = intrinsic.at<double>(0,0) * preferences->GetNumber("CameraZeroDist", CAMERA_ZERO_DIST) / preferences->GetNumber("CameraHeight", CAMERA_GOAL_HEIGHT);
				double m_horizon = intrinsic.at<double>(0,0) * preferences->GetNumber("CameraHeight", CAMERA_GOAL_HEIGHT) / preferences->GetNumber("CameraZeroDist", CAMERA_ZERO_DIST);
				double m_flat = sqrt(intrinsic.at<double>(0,0)*intrinsic.at<double>(0,0) + m_horizon*m_horizon);

				// dX is the offset of the target from the focal center to the right
				float dX = undistortedPoints[0].x - intrinsic.at<double>(0,2);
				// dY is the distance from the zenith to the target on the image
				float dY = m_zenith + undistortedPoints[0].y - intrinsic.at<double>(1,2);
				// The real azimuth to the target is on the horizon, so scale it accordingly
				float azimuth = dX * ((m_zenith + m_horizon) / dY);
				// Vehicle's yaw is negative arc tangent from the current heading to the target
				yaw = -atan2(azimuth, m_flat);

				table->PutNumber("Boiler Distance", distance);
				table->PutNumber("Boiler Yaw", yaw);
				table->PutNumber("Boiler Time", cv::getTickCount()/cv::getTickFrequency());
			}
			timer_names.push_back("calcs done"); timer_values.push_back(cv::getTickCount());

#ifdef XGUI_ENABLED
			if (dispMode == 2 and imagePoints.size() == 4) {
				cv::circle(display, imagePoints[0]*displayRatio, 8, cv::Scalar(  0,  0,200), 1);
				cv::circle(display, imagePoints[1]*displayRatio, 8, cv::Scalar(  0,200,200), 1);
				cv::circle(display, imagePoints[2]*displayRatio, 8, cv::Scalar(  0,200,  0), 1);
				cv::circle(display, imagePoints[3]*displayRatio, 8, cv::Scalar(200,  0,  0), 1);

				double targetScale = 1.0 / 240.0; // 240 inches is about max distance
				cv::Point dispTarget(
						displaySize.width * (0.5 - distance * sin(yaw) *targetScale),
						displaySize.height* (0.9 - distance * cos(yaw) *targetScale)
						);

				cv::line(display,
						dispTarget,
						cv::Point(displaySize.width/2,displaySize.height*0.9),
						cv::Scalar(0,255,255));
				std::ostringstream oss;
				oss << "Yaw: " << yaw << " Tvec: " << imagePoints;
				cv::putText(display, oss.str(), cv::Point(20,20), 0, 0.33, cv::Scalar(0,200,200));
				std::ostringstream oss1;
				oss1 << "Distance: " << distance;
				cv::putText(display, oss1.str(), cv::Point(20,40), 0, 0.33, cv::Scalar(0,200,200));
			}
#endif
		}

#ifdef XGUI_ENABLED
		if (dispMode > 0) {
			for(size_t i=1; i < timer_values.size(); ++i) {
				long int val = timer_values[i] - timer_values[i-1];
				std::ostringstream osst;
				osst << timer_names[i] << ": " << val / cv::getTickFrequency();
				cv::putText(display, osst.str(), cv::Point(20,40+20*i), 0, 0.33, cv::Scalar(0,200,200));
			}

			cv::imshow(detection_window, display);
		}

		int key = cv::waitKey(2);
		if ((key & 255) == 27) break;
		if ((key & 255) == 32) {
			if(++dispMode > 2) dispMode =0;
		}
		if ((key & 255) == 's') cv::waitKey(0);
#endif
	}
	return 0;
}

