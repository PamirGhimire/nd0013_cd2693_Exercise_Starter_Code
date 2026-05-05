#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Client.h>
#include <carla/client/Map.h>
#include <carla/client/Sensor.h>
#include <carla/client/Vehicle.h>
#include <carla/geom/Location.h>
#include <carla/geom/Transform.h>
#include <carla/sensor/data/LidarMeasurement.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/console/time.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "helper.h"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std;
using namespace std::chrono_literals;

namespace {

constexpr bool kUseNdt = true;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegreesToRadians = kPi / 180.0;

constexpr float kSteeringStep = 0.02f;
constexpr float kThrottleStep = 0.1f;
constexpr float kMaxThrottle = 1.0f;
constexpr float kMaxSteer = 1.0f;
constexpr float kSteerAngleScale = kPi / 4.0f;
constexpr float kDefaultBrake = 1.0f;

constexpr float kCarLength = 4.0f;
constexpr float kCarWidth = 2.0f;
constexpr float kCarHeight = 2.0f;
constexpr double kSteerRayStartDistance = 2.0;
constexpr double kSteerRayEndDistance = 4.0;
constexpr double kCameraHeight = 60.0;
constexpr double kCameraLookAhead = 1.0;

constexpr double kInitialLidarX = -0.5;
constexpr double kInitialLidarY = 0.0;
constexpr double kInitialLidarZ = 1.8;
constexpr double kLidarYawCorrection = kPi / 2.0;
constexpr double kMinEgoPointDistanceSquared = 8.0;
constexpr size_t kMinScanPointCount = 5000;
constexpr float kVoxelLeafSize = 0.5f;

const string kLidarUpperFov = "15";
const string kLidarLowerFov = "-25";
const string kLidarChannels = "32";
const string kLidarRange = "30";
const string kLidarRotationFrequency = "60";
const string kLidarPointsPerSecond = "500000";

constexpr double kMaxAllowedError = 1.2;
constexpr double kPassDistance = 170.0;
constexpr size_t kSpeedHistorySize = 6;

constexpr int kIcpMaxIterations = 4;
constexpr double kIcpTransformationEpsilon = 1e-6;
constexpr double kIcpMaxCorrespondenceDistance = 20.0;

constexpr int kNdtMaxIterations = 4;
constexpr double kNdtTransformationEpsilon = 1e-6;
constexpr double kNdtResolution = 1.0;

constexpr int kTextX = 200;
constexpr int kMaxErrorTextY = 100;
constexpr int kPoseErrorTextY = 150;
constexpr int kDistanceTextY = 200;
constexpr int kEvalTextY = 50;
constexpr int kTextSize = 32;

PointCloudT pclCloud;
cc::Vehicle::Control control;
vector<ControlState> pendingControls;
bool refreshView = false;

struct RegistrationResult {
	Eigen::Matrix4d transform;
	double fitnessScore;
	double processTimeMs;
	bool hasConverged;
};

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void*) {
	if (!event.keyDown()) {
		return;
	}

	const string key = event.getKeySym();
	if (key == "Right") {
		pendingControls.emplace_back(0.0f, -kSteeringStep, 0.0f);
	} else if (key == "Left") {
		pendingControls.emplace_back(0.0f, kSteeringStep, 0.0f);
	} else if (key == "Up") {
		pendingControls.emplace_back(kThrottleStep, 0.0f, 0.0f);
	} else if (key == "Down") {
		pendingControls.emplace_back(-kThrottleStep, 0.0f, 0.0f);
	} else if (key == "a") {
		refreshView = true;
	}
}

void Actuate(ControlState command, cc::Vehicle::Control& state) {
	if (command.t > 0.0f) {
		state.throttle = state.reverse
			? min(command.t, kMaxThrottle)
			: min(state.throttle + command.t, kMaxThrottle);
		state.reverse = false;
	} else if (command.t < 0.0f) {
		const float reverseThrottle = -command.t;
		state.throttle = state.reverse
			? min(state.throttle + reverseThrottle, kMaxThrottle)
			: min(reverseThrottle, kMaxThrottle);
		state.reverse = true;
	}

	state.steer = min(max(state.steer + command.s, -kMaxSteer), kMaxSteer);
	state.brake = command.b;
}

void drawCar(Pose pose, int id, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr& viewer) {
	BoxQ box;
	box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
	box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
	box.cube_length = kCarLength;
	box.cube_width = kCarWidth;
	box.cube_height = kCarHeight;
	renderBox(viewer, box, id, color, alpha);
}

Pose getVehiclePose(cc::Vehicle& vehicle) {
	const auto transform = vehicle.GetTransform();
	return Pose(
		Point(transform.location.x, transform.location.y, transform.location.z),
		Rotate(
			transform.rotation.yaw * kDegreesToRadians,
			transform.rotation.pitch * kDegreesToRadians,
			transform.rotation.roll * kDegreesToRadians));
}

Eigen::Matrix4d getLidarToVehicleTransform() {
	return transform3D(
		kLidarYawCorrection,
		0.0,
		0.0,
		kInitialLidarX,
		kInitialLidarY,
		kInitialLidarZ);
}

Eigen::Matrix4d getVehicleToMapTransform(Pose pose, double xOffset = 0.0) {
	return transform3D(
		pose.rotation.yaw,
		pose.rotation.pitch,
		pose.rotation.roll,
		pose.position.x + xOffset,
		pose.position.y,
		pose.position.z);
}

double computeAverageXStep(const vector<double>& xPositions) {
	if (xPositions.size() < 2) {
		return 0.0;
	}

	double totalStep = 0.0;
	for (size_t i = 1; i < xPositions.size(); ++i) {
		totalStep += xPositions[i] - xPositions[i - 1];
	}
	return totalStep / static_cast<double>(xPositions.size() - 1);
}

RegistrationResult runIcp(PointCloudT::Ptr mapCloud, PointCloudT::Ptr scanCloud, Pose pose, double xOffset) {
	const Eigen::Matrix4d vehicleToMapGuess = getVehicleToMapTransform(pose, xOffset);
	const Eigen::Matrix4d lidarToMapGuess = vehicleToMapGuess * getLidarToVehicleTransform();

	PointCloudT::Ptr scanInMapGuess(new PointCloudT);
	pcl::transformPointCloud(*scanCloud, *scanInMapGuess, lidarToMapGuess);

	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setInputSource(scanInMapGuess);
	icp.setInputTarget(mapCloud);
	icp.setMaximumIterations(kIcpMaxIterations);
	icp.setTransformationEpsilon(kIcpTransformationEpsilon);
	icp.setMaxCorrespondenceDistance(kIcpMaxCorrespondenceDistance);

	PointCloudT alignedScan;
	pcl::console::TicToc timer;
	timer.tic();
	icp.align(alignedScan);

	return {
		icp.getFinalTransformation().cast<double>() * vehicleToMapGuess,
		icp.getFitnessScore(),
		timer.toc(),
		icp.hasConverged()
	};
}

RegistrationResult runNdt(PointCloudT::Ptr mapCloud, PointCloudT::Ptr scanCloud, Pose pose, double xOffset) {
	const Eigen::Matrix4d vehicleToMapGuess = getVehicleToMapTransform(pose, xOffset);

	PointCloudT::Ptr scanInVehicle(new PointCloudT);
	pcl::transformPointCloud(*scanCloud, *scanInVehicle, getLidarToVehicleTransform());

	pcl::NormalDistributionsTransform<PointT, PointT> ndt;
	ndt.setInputSource(scanInVehicle);
	ndt.setInputTarget(mapCloud);
	ndt.setMaximumIterations(kNdtMaxIterations);
	ndt.setTransformationEpsilon(kNdtTransformationEpsilon);
	ndt.setResolution(kNdtResolution);

	PointCloudT alignedScan;
	pcl::console::TicToc timer;
	timer.tic();
	ndt.align(alignedScan, vehicleToMapGuess.cast<float>());

	return {
		ndt.getFinalTransformation().cast<double>(),
		ndt.getFitnessScore(),
		timer.toc(),
		ndt.hasConverged()
	};
}

RegistrationResult registerScan(PointCloudT::Ptr mapCloud, PointCloudT::Ptr scanCloud, Pose pose, double xOffset) {
	return kUseNdt
		? runNdt(mapCloud, scanCloud, pose, xOffset)
		: runIcp(mapCloud, scanCloud, pose, xOffset);
}

double horizontalDistance(Point a, Point b) {
	const double dx = a.x - b.x;
	const double dy = a.y - b.y;
	return sqrt(dx * dx + dy * dy);
}

void updateEvalText(pcl::visualization::PCLVisualizer::Ptr& viewer, double maxError) {
	viewer->removeShape("eval");
	if (maxError > kMaxAllowedError) {
		viewer->addText(
			"Try Again",
			kTextX,
			kEvalTextY,
			kTextSize,
			1.0,
			0.0,
			0.0,
			"eval",
			0);
		return;
	}

	viewer->addText("Passed!", kTextX, kEvalTextY, kTextSize, 0.0, 1.0, 0.0, "eval", 0);
}

void updateMetricText(
	pcl::visualization::PCLVisualizer::Ptr& viewer,
	const string& id,
	const string& label,
	double value,
	int y) {
	viewer->removeShape(id);
	viewer->addText(label + to_string(value) + " m", kTextX, y, kTextSize, 1.0, 1.0, 1.0, id, 0);
}

} // namespace

int main() {
	auto client = cc::Client("localhost", 2000);
	client.SetTimeout(2s);
	auto world = client.GetWorld();

	auto blueprintLibrary = world.GetBlueprintLibrary();
	auto vehicles = blueprintLibrary->Filter("vehicle");
	auto map = world.GetMap();
	auto spawnTransform = map->GetRecommendedSpawnPoints()[1];
	auto egoActor = world.SpawnActor((*vehicles)[12], spawnTransform);

	auto lidarBlueprint = *(blueprintLibrary->Find("sensor.lidar.ray_cast"));
	lidarBlueprint.SetAttribute("upper_fov", kLidarUpperFov);
	lidarBlueprint.SetAttribute("lower_fov", kLidarLowerFov);
	lidarBlueprint.SetAttribute("channels", kLidarChannels);
	lidarBlueprint.SetAttribute("range", kLidarRange);
	lidarBlueprint.SetAttribute("rotation_frequency", kLidarRotationFrequency);
	lidarBlueprint.SetAttribute("points_per_second", kLidarPointsPerSecond);

	auto lidarTransform = cg::Transform(cg::Location(kInitialLidarX, kInitialLidarY, kInitialLidarZ));
	auto lidarActor = world.SpawnActor(lidarBlueprint, lidarTransform, egoActor.get());
	auto lidar = boost::static_pointer_cast<cc::Sensor>(lidarActor);
	bool newScan = true;

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->registerKeyboardCallback(keyboardEventOccurred, static_cast<void*>(&viewer));

	auto vehicle = boost::static_pointer_cast<cc::Vehicle>(egoActor);
	Pose pose(Point(0, 0, 0), Rotate(0, 0, 0));
	const Pose poseRef = getVehiclePose(*vehicle);

	PointCloudT::Ptr mapCloud(new PointCloudT);
	pcl::io::loadPCDFile("map.pcd", *mapCloud);
	cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
	renderPointCloud(viewer, mapCloud, "map", Color(0, 0, 1));

	PointCloudT::Ptr filteredScan(new PointCloudT);
	PointCloudT::Ptr scanCloud(new PointCloudT);
	PointCloudT::Ptr transformedScan(new PointCloudT);

	lidar->Listen([&newScan, &scanCloud](auto data) {
		if (!newScan) {
			return;
		}

		auto scan = boost::static_pointer_cast<csd::LidarMeasurement>(data);
		for (const auto& detection : *scan) {
			const double distanceSquared =
				detection.x * detection.x + detection.y * detection.y + detection.z * detection.z;
			if (distanceSquared > kMinEgoPointDistanceSquared) {
				pclCloud.points.emplace_back(detection.x, detection.y, detection.z);
			}
		}

		if (pclCloud.points.size() >= kMinScanPointCount) {
			*scanCloud = pclCloud;
			newScan = false;
		}
	});

	double maxError = 0.0;
	size_t scanCount = 0;
	vector<double> xHistory;

	while (!viewer->wasStopped()) {
		while (newScan) {
			std::this_thread::sleep_for(0.1s);
			world.Tick(1s);
		}

		if (refreshView) {
			viewer->setCameraPosition(
				pose.position.x,
				pose.position.y,
				kCameraHeight,
				pose.position.x + kCameraLookAhead,
				pose.position.y + kCameraLookAhead,
				0,
				0,
				0,
				1);
			refreshView = false;
		}

		viewer->removeShape("box0");
		viewer->removeShape("boxFill0");
		Pose truePose = getVehiclePose(*vehicle) - poseRef;
		drawCar(truePose, 0, Color(1, 0, 0), 0.7, viewer);

		const double steerTheta = control.steer * kSteerAngleScale + truePose.rotation.yaw;
		viewer->removeShape("steer");
		renderRay(
			viewer,
			Point(
				truePose.position.x + kSteerRayStartDistance * cos(truePose.rotation.yaw),
				truePose.position.y + kSteerRayStartDistance * sin(truePose.rotation.yaw),
				truePose.position.z),
			Point(
				truePose.position.x + kSteerRayEndDistance * cos(steerTheta),
				truePose.position.y + kSteerRayEndDistance * sin(steerTheta),
				truePose.position.z),
			"steer",
			Color(0, 1, 0));

		ControlState command(0.0f, 0.0f, kDefaultBrake);
		if (!pendingControls.empty()) {
			command = pendingControls.back();
			pendingControls.clear();
			Actuate(command, control);
			vehicle->ApplyControl(control);
		}

		viewer->spinOnce();

		newScan = true;
		if (scanCount == 0) {
			pose = truePose;
		}
		++scanCount;

		pcl::VoxelGrid<PointT> voxelFilter;
		voxelFilter.setInputCloud(scanCloud);
		voxelFilter.setLeafSize(kVoxelLeafSize, kVoxelLeafSize, kVoxelLeafSize);
		voxelFilter.filter(*filteredScan);

		const double xOffset = computeAverageXStep(xHistory);
		const RegistrationResult result = registerScan(mapCloud, filteredScan, pose, xOffset);
		pose = getPose(result.transform);

		xHistory.push_back(pose.position.x);
		if (xHistory.size() > kSpeedHistorySize) {
			xHistory.erase(xHistory.begin());
		}

		const Eigen::Matrix4f lidarToMap =
			(getVehicleToMapTransform(pose) * getLidarToVehicleTransform()).cast<float>();
		pcl::transformPointCloud(*filteredScan, *transformedScan, lidarToMap);

		viewer->removePointCloud("scan");
		renderPointCloud(viewer, transformedScan, "scan", Color(1, 0, 0));

		viewer->removeAllShapes();
		drawCar(pose, 1, Color(0, 1, 0), 0.35, viewer);

		const double poseError = horizontalDistance(truePose.position, pose.position);
		maxError = max(maxError, poseError);
		const double distanceDriven = horizontalDistance(truePose.position, Point(0, 0, truePose.position.z));

		updateMetricText(viewer, "maxE", "Max Error: ", maxError, kMaxErrorTextY);
		updateMetricText(viewer, "derror", "Pose error: ", poseError, kPoseErrorTextY);
		updateMetricText(viewer, "dist", "Distance: ", distanceDriven, kDistanceTextY);

		if (maxError > kMaxAllowedError || distanceDriven >= kPassDistance) {
			updateEvalText(viewer, maxError);
		}

		pclCloud.points.clear();
	}

	return 0;
}
