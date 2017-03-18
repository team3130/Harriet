//#define XGUI_ENABLED
#define NETWORKTABLES_ENABLED

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <ctime>
#include "networktables/NetworkTable.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

static const cv::Size frameSize(640,480);
static const double cameraFPS = 10;
static const double MIN_AREA = 0.0002 * frameSize.height * frameSize.width;
static const double BOILER_TAPE_RATIO = 2.5;
static const double BOILER_TAPE_RATIO2 = BOILER_TAPE_RATIO/2;
static const double CAMERA_GOAL_HEIGHT = 69; //!<- Top tape height is 88" and the camera is 19" above the floor
static const double CAMERA_ZERO_DIST = 77;

static cv::Mat intrinsic, distortion;

#ifdef NETWORKTABLES_ENABLED
	static std::shared_ptr<NetworkTable> table;
	static std::shared_ptr<NetworkTable> preferences;
#else
	struct PREFS {
		static double GetNumber(std::string name, double deflt) {return deflt;};
	} *preferences;
#endif

static const std::vector<cv::Point3d> realBoiler {{0,-4, 0}, {0, 0, 0.3}, {0, 4, 0.3}, {0, 6, 0}};
static const std::vector<cv::Point3f> realLift {
	{-5.125,-2.5, 10.5}, {-5.125, 2.5, 10.5}, // Left, top then bottom
	{ 5.125,-2.5, 10.5}, { 5.125, 2.5, 10.5}  // Right, top then bottom
};
static const cv::Vec3d boiler_camera_offset(8.0, 0.0, 12);
static const cv::Vec3d lift_camera_offset(-13.0, -1.0, 0.0);

#ifdef XGUI_ENABLED
	#include "opencv2/highgui.hpp"
	static const cv::Size displaySize(320, 240);
	static const double displayRatio = double(displaySize.height) / frameSize.height;
	static const char* detection_window = "Object Detection";
	static cv::Mat display;
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

bool readIntrinsics(const char *filename)
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

bool ProcessGearLift(std::vector<std::vector<cv::Point>> &contours)
{
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

		cv::Vec3d lift_camera_turn(0, (preferences->GetNumber("Peg Camera Bias", 27) * (CV_PI/180.0)), 0);
		cv::Vec3d rvec, tvec, robot_loc, target_loc;
		cv::Matx33d rmat, crmat;
		cv::Rodrigues(lift_camera_turn, crmat);

		cv::solvePnP(
				realLift,       // 3-d points in object coordinate
				imagePoints,        // 2-d points in image coordinates
				intrinsic,           // Our camera matrix
				distortion,
				rvec,                // Output rotation *vector*.
				tvec                 // Output translation vector.
		);
		cv::Rodrigues(rvec + lift_camera_turn, rmat);
		// tvec is where the target is in the camera coordinates
		// We offset it with the camera_offset vector and rotate opposite to the target's rotation
		target_loc = crmat * (tvec + (crmat.t() * lift_camera_offset));
		robot_loc = rmat.t() * -(target_loc);
		// Yaw of the robot is positive if the target is to the left less camera own turn
		double yaw = -atan2(target_loc[0],target_loc[2]) - lift_camera_turn[1];

#ifdef NETWORKTABLES_ENABLED
		table->PutNumber("Peg Distance", cv::norm(target_loc));
		table->PutNumber("Peg Crossrange", robot_loc[0]);
		table->PutNumber("Peg Downrange", -robot_loc[2]); // Robot is in negative Z area. We expect positive down range
		table->PutNumber("Peg Yaw", yaw);
		table->PutNumber("Peg Time", cv::getTickCount() / cv::getTickFrequency());
#endif

#ifdef XGUI_ENABLED
		cv::line(display, imagePoints[0] * displayRatio, imagePoints[1] * displayRatio, cv::Scalar(200, 0, 255), 1, cv::LINE_AA);
		cv::line(display, imagePoints[1] * displayRatio, imagePoints[3] * displayRatio, cv::Scalar(200, 0, 255), 1, cv::LINE_AA);
		cv::line(display, imagePoints[3] * displayRatio, imagePoints[2] * displayRatio, cv::Scalar(200, 0, 255), 1, cv::LINE_AA);
		cv::line(display, imagePoints[2] * displayRatio, imagePoints[0] * displayRatio, cv::Scalar(200, 0, 255), 1, cv::LINE_AA);

		cv::circle(display, lCorn[0]*displayRatio, 8, cv::Scalar(0,125,255), 1);
		cv::circle(display, lCorn[1]*displayRatio, 8, cv::Scalar(0,0,255), 1);
		cv::circle(display, rCorn[0]*displayRatio, 8, cv::Scalar(125,255,0), 1);
		cv::circle(display, rCorn[1]*displayRatio, 8, cv::Scalar(0,255,0), 1);

		cv::Point dispTarget = cv::Point(
				0.5*displaySize.width  + (displaySize.height/150)*(target_loc[0]),
				0.9*displaySize.height - (displaySize.height/150)*(target_loc[2])
				);
		cv::line(display,
				dispTarget,
				cv::Point(displaySize.width/2,displaySize.height*0.9),
				cv::Scalar(0,255,255));

		cv::Point dispRobot = cv::Point(
				0.5*displaySize.width  - (displaySize.height/150)*(lift_camera_offset[0]),
				0.9*displaySize.height + (displaySize.height/150)*(lift_camera_offset[2])
				);
		cv::line(display,
				dispTarget,
				dispRobot,
				cv::Scalar(0,255,255));

		cv::Vec3d peg = robot_loc / cv::norm(robot_loc);
		cv::Point peg2D = displaySize.height/10 * cv::Point2d(peg[0],peg[2]);
		cv::line(display,
				dispTarget,
				dispTarget + peg2D,
				cv::Scalar(0,0,255));

		std::ostringstream oss;
		oss << "Yaw: " << 180.0*yaw/CV_PI << " (" << rvec[0] << " " << rvec[1] << " " << rvec[2] << ")";
		cv::putText(display, oss.str(), cv::Point(20,20), 0, 0.33, cv::Scalar(0,200,200));
		std::ostringstream oss1;
		oss1 << "Down: " << -robot_loc[2] << "  Cross: " << robot_loc[0];
		cv::putText(display, oss1.str(), cv::Point(20,40), 0, 0.33, cv::Scalar(0,200,200));
#endif
		return true;
	}
	return false;
}

bool ProcessHighGoal(std::vector<std::vector<cv::Point>> &contours)
{
	// First build a graph of relationships between contours
	std::vector<RingRelation> graph;
	for (auto&& cont : contours)
	{
		cv::RotatedRect rect = cv::minAreaRect(cont);
		double area = rect.size.height * rect.size.width;
		if (area < MIN_AREA) continue;
		// All small noise is ignored at this point
		righten(rect);
		// Righten rectangle makes the longer side be "vertical" (virtually, to measure angle)
		if (fabs(rect.angle) < 45) continue;
		// We are only interested in horizontal-ish shapes for the boiler detection
		graph.push_back(RingRelation(&cont, rect, graph));
	}
	std::sort(graph.begin(), graph.end());

	if (graph.size() > 1) {
		double distance, yaw;

		std::vector<cv::Point2d> imagePoints;
		// Take the two best relationships and find the mid points
		// choosing which one is on the top first (by the y coordinate)
		if (graph[0].my_rect.center.y < graph[1].my_rect.center.y) {
			FindMidPoints(graph[0].my_cont, graph[1].my_cont, imagePoints);
		}
		else {
			FindMidPoints(graph[1].my_cont, graph[0].my_cont, imagePoints);
		}

		if(imagePoints.size() == 4) {
			// Now as we have only 4 points we can undistort their coordinates using camera intrinsics
			std::vector<cv::Point2d> undistortedPoints;
			cv::undistortPoints(imagePoints, undistortedPoints, intrinsic, distortion, cv::noArray(), intrinsic);

			double cam_bias = preferences->GetNumber("Boiler Camera Bias", 0) * (CV_PI/180.0);
			double tangent = CAMERA_GOAL_HEIGHT / preferences->GetNumber("Boiler Camera ZeroDist", CAMERA_ZERO_DIST);
			double focal = intrinsic.at<double>(1,1);
			double zenith = focal / tangent;
			double horizon = focal * tangent;
			double flat = sqrt(focal*focal + horizon*horizon);
			cv::Point2d Zen = cv::Point2d(intrinsic.at<double>(0,2), intrinsic.at<double>(1,2)-zenith);

			double pixels = cv::norm(undistortedPoints[3] - undistortedPoints[0]);
			double inches = fabs(realBoiler[3].y - realBoiler[0].y);

			// dX is the offset of the target from the focal center to the right
			double dX = undistortedPoints[0].x - intrinsic.at<double>(0,2);
			// dY is the distance from the zenith to the target on the image
			double dY = zenith + undistortedPoints[0].y - intrinsic.at<double>(1,2);
			// The real azimuth to the target is on the horizon, so scale it accordingly
			double azimuth = dX * ((zenith + horizon) / dY);
			// Vehicle's yaw is negative arc tangent from the current heading to the target
			yaw = -atan2(azimuth, flat) + cam_bias;

			// TODO This is again an approximation of the distance between the camera and the target
			// TODO It's an OK approximation for OK aimed targets but won't work if the target is off
			cv::Point2d q(dX, undistortedPoints[0].y - intrinsic.at<double>(1,2));
			double q_len = cv::norm(q);
			double alpha = atan(tangent);
			double theta = atan2(q_len, focal);
			double height = pixels * sin(CV_PI/2 - theta) / sin(CV_PI/2 + theta - alpha);
			double a = pixels * sin(alpha) / sin(CV_PI/2 + theta - alpha);
			distance = (sqrt(focal*focal + q_len*q_len) + a) * inches / height;

			// Do further adjustments only if distance makes sense
			if(distance > CAMERA_GOAL_HEIGHT) {
				double level_dist = sqrt(distance*distance - CAMERA_GOAL_HEIGHT*CAMERA_GOAL_HEIGHT);
				cv::Vec3d D(-level_dist*sin(yaw), 0, level_dist*cos(yaw));
				cv::Vec3d A = boiler_camera_offset + D;
				yaw = -atan2(A[0], A[2]);
#ifdef NETWORKTABLES_ENABLED
				table->PutNumber("Boiler Zero", zenith - cv::norm(Zen - undistortedPoints[0]));
				table->PutNumber("Boiler Groundrange", cv::norm(A));
				table->PutNumber("Boiler Distance", distance);
				table->PutNumber("Boiler Yaw", yaw);
				table->PutNumber("Boiler Time", cv::getTickCount()/cv::getTickFrequency());
#endif
			}

#ifdef XGUI_ENABLED
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
			oss << "Yaw: " << yaw;
			cv::putText(display, oss.str(), cv::Point(20,20), 0, 0.33, cv::Scalar(0,200,200));
			std::ostringstream oss1;
			oss1 << "Distance: " << distance;
			cv::putText(display, oss1.str(), cv::Point(20,40), 0, 0.33, cv::Scalar(0,200,200));
#endif
			return true;
		}
	}
	return false;
}

enum TaskID {
	kLift,
	kBoiler
};

int main(int argc, const char** argv)
{
	if(argc != 3) {
		std::cerr << "Format: this-program task intrinsics.yml" << std::endl
				<< "Tasks: Peg or Boiler" << std::endl;
	}
	const char* task_argv = argv[1];
	const char* intrinsic_file = argv[2];
	static std::string taskSysTime;
	static TaskID task;

	if(std::strcmp(task_argv, "Peg") == 0) {
		taskSysTime = "Peg Sys Time";
		task = kLift;
	}
	else
	if(std::strcmp(task_argv, "Boiler") == 0) {
		taskSysTime = "Boiler Sys Time";
		task = kBoiler;
	}
	else {
		std::cerr << "The task can only be either Peg or Boiler" << std::endl;
	}
	if(!readIntrinsics(intrinsic_file)) return -1;

#ifdef NETWORKTABLES_ENABLED
	NetworkTable::SetClientMode();
	NetworkTable::SetTeam(3130);
	table = NetworkTable::GetTable("/Jetson");
	preferences = NetworkTable::GetTable("/Preferences");
#else
	preferences = NULL;
#endif

	cv::Mat frame, hsv, filtered, buffer1;
	static cv::Vec3i BlobLower(50,  80,  50);
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

	long int n = 0;
	double total_time = 0;
	for(;;) {
#ifdef NETWORKTABLES_ENABLED
		// System timer always runs so we can see the coprocessor is online
		// and can compare the last measurement time against the current time.
		table->PutNumber(taskSysTime, cv::getTickCount()/cv::getTickFrequency());
#endif

		std::vector<long int> timer_values;
		std::vector<std::string> timer_names;
		timer_names.push_back("start"); timer_values.push_back(cv::getTickCount());

		capture >> frame;
		if (frame.empty()) {
			std::cerr << " Error reading from camera, empty frame." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(2));
			continue;
		}
		timer_names.push_back("frame taken"); timer_values.push_back(cv::getTickCount());

		cv::cvtColor(frame, hsv, CV_BGR2HSV);
		inRange(hsv, BlobLower, BlobUpper, filtered);
		timer_names.push_back("inRange applied"); timer_values.push_back(cv::getTickCount());
		erode(filtered, buffer1, element);
		dilate(buffer1, filtered, element);
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

		switch (task) {
		case kBoiler:
			ProcessHighGoal(contours);
			break;
		case kLift:
			ProcessGearLift(contours);
			break;
		}
		timer_names.push_back("calcs done"); timer_values.push_back(cv::getTickCount());
		total_time += (cv::getTickCount() - timer_values[1]) / cv::getTickFrequency();

		if(++n > cameraFPS * 120) {
			std::cerr << date_now() << std::endl;
			std::cerr << " Time per frame = " << total_time / n << " at freq: " << cv::getTickFrequency() << std::endl;
			std::cerr << " Prefs: Peg Camera Bias = " << preferences->GetNumber("Peg Camera Bias", 9999) << std::endl;
			std::cerr << " Prefs: Boiler Camera Bias = " << preferences->GetNumber("Boiler Camera Bias", 9999) << std::endl;
			std::cerr << " Prefs: Boiler Camera ZeroDist = " << preferences->GetNumber("Boiler Camera ZeroDist", 9999) << std::endl;
			total_time = 0;
			n = 0;
		}

#ifdef XGUI_ENABLED
		if (dispMode > 0) {
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
#endif
	}
	return 0;
}

