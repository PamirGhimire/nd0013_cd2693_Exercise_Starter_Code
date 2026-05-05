
#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Map.h>
#include <carla/geom/Location.h>
#include <carla/geom/Transform.h>
#include <carla/client/Sensor.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <thread>

#include <carla/client/Vehicle.h>

//pcl code
namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;
using namespace std;

#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include "helper.h"
#include <sstream>
#include <chrono> 
#include <ctime> 
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/console/time.h>   // TicToc

// Store accumulated lidar points globally.
PointCloudT pclCloud;
// Store the current vehicle control command.
cc::Vehicle::Control control;
// Declare timestamps used for scan timing.
std::chrono::time_point<std::chrono::system_clock> currentTime;
// Queue keyboard control inputs.
vector<ControlState> cs;

// Track whether the camera should recenter.
bool refresh_view = false;
// Handle keyboard input from the visualizer.
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer)
// Begin block.
{

  	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
	// Check for a right steering key press.
	if (event.getKeySym() == "Right" && event.keyDown()){
		// Add the requested control change.
		cs.push_back(ControlState(0, -0.02, 0));
  	// End block.
  	}
	// Check for a left steering key press.
	else if (event.getKeySym() == "Left" && event.keyDown()){
		// Add the requested control change.
		cs.push_back(ControlState(0, 0.02, 0)); 
  	// End block.
  	}
  	// Check for a throttle key press.
  	if (event.getKeySym() == "Up" && event.keyDown()){
		// Add the requested control change.
		cs.push_back(ControlState(0.1, 0, 0));
  	// End block.
  	}
	// Check for a reverse key press.
	else if (event.getKeySym() == "Down" && event.keyDown()){
		// Add the requested control change.
		cs.push_back(ControlState(-0.1, 0, 0)); 
  	// End block.
  	}
	// Check for a camera-refresh key press.
	if(event.getKeySym() == "a" && event.keyDown()){
		// Request camera recentering.
		refresh_view = true;
	// End block.
	}
// End block.
}

// Apply a requested control change i.e., ACTUATE
void Accuate(ControlState response, cc::Vehicle::Control& state){

	// Handle forward throttle input.
	if(response.t > 0){
		// Check whether the car is already driving forward.
		if(!state.reverse){
			// Clamp throttle to the legal range.
			state.throttle = min(state.throttle+response.t, 1.0f);
		// End block.
		}
		// Handle the alternative branch.
		else{
			// Switch out of reverse mode.
			state.reverse = false;
			// Clamp throttle to the legal range.
			state.throttle = min(response.t, 1.0f);
		// End block.
		}
	// End block.
	}
	// Handle reverse throttle input.
	else if(response.t < 0){
		// Convert reverse input magnitude to positive.
		response.t = -response.t;
		// Check whether the car is already reversing.
		if(state.reverse){
			// Clamp throttle to the legal range.
			state.throttle = min(state.throttle+response.t, 1.0f);
		// End block.
		}
		// Handle the alternative branch.
		else{
			// Switch into reverse mode.
			state.reverse = true;
			// Clamp throttle to the legal range.
			state.throttle = min(response.t, 1.0f);

		// End block.
		}
	// End block.
	}
	// Update and clamp steering.
	state.steer = min( max(state.steer+response.s, -1.0f), 1.0f);
	// Apply brake input.
	state.brake = response.b;
// End block.
}

// Render a car-shaped box for a pose.
void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr& viewer){

	// Create a render box description.
	BoxQ box;
	// Set the box position.
	box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
    // Set the box orientation.
    box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
    // Set the car box length.
    box.cube_length = 4;
    // Set the car box width.
    box.cube_width = 2;
    // Set the car box height.
    box.cube_height = 2;
	// Draw the box in the viewer.
	renderBox(viewer, box, num, color, alpha);
// End block.
}

struct Result {
	Eigen::Matrix4d transform;
	double fitness_score;
	double process_time;
	int has_converged;
};

const bool USE_NDT = true;

// lidar is rotated 90 degrees clockwise
Eigen::Matrix4d getLidarToVehicleTransform() {
	return transform3D(pi / 2, 0, 0, -0.5, 0, 1.8);
}

Result ICP(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose, double x_inc) {
	Eigen::Matrix4d vehicleToMapGuess = transform3D(
		startingPose.rotation.yaw,
		startingPose.rotation.pitch,
		startingPose.rotation.roll,
		startingPose.position.x + x_inc,
		startingPose.position.y,
		startingPose.position.z);
	Eigen::Matrix4d initTransform = vehicleToMapGuess * getLidarToVehicleTransform();

	PointCloudT::Ptr transformSource(new PointCloudT); //source transformed to map coordinates will be stored here
	pcl::transformPointCloud(*source, *transformSource, initTransform);

	pcl::console::TicToc time;
	time.tic();
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setTransformationEpsilon(1e-6);
	icp.setMaximumIterations(4);
	icp.setInputSource(transformSource);
	icp.setInputTarget(target);
	icp.setMaxCorrespondenceDistance(20);

	PointCloudT::Ptr cloudIcp(new PointCloudT);
	icp.align(*cloudIcp);

	Eigen::Matrix4d transformationMatrix = icp.getFinalTransformation().cast<double>() * vehicleToMapGuess;
	return Result {
		transformationMatrix,
		icp.getFitnessScore(),
		time.toc(),
		icp.hasConverged()
	};
}

Result NDT(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose, double x_inc) {
	Eigen::Matrix4d vehicleToMapGuess = transform3D(
		startingPose.rotation.yaw,
		startingPose.rotation.pitch,
		startingPose.rotation.roll,
		startingPose.position.x + x_inc,
		startingPose.position.y,
		startingPose.position.z);

	PointCloudT::Ptr vehicleSource(new PointCloudT);
	pcl::transformPointCloud(*source, *vehicleSource, getLidarToVehicleTransform());

	pcl::console::TicToc time;
	time.tic();
	pcl::NormalDistributionsTransform<PointT, PointT> ndt;
	ndt.setTransformationEpsilon(1e-6);
	ndt.setResolution(1.0);
	ndt.setMaximumIterations(4);
	ndt.setInputSource(vehicleSource);
	ndt.setInputTarget(target);

	PointCloudT::Ptr cloudNdt(new PointCloudT);
	ndt.align(*cloudNdt, vehicleToMapGuess.cast<float>());

	return Result {
		ndt.getFinalTransformation().cast<double>(),
		ndt.getFitnessScore(),
		time.toc(),
		ndt.hasConverged()
	};
}

double compute_average_speed_per_iteration(vector<double> xs) {
	double avg_speed = 0;
	int n = xs.size() - 1;
	if(n == 0)
		return 0.0;
	for(int i = 0; i < n; i++)
		avg_speed += xs[i + 1] - xs[i];
	return avg_speed / n;
}

// Start the program entry point.
int main(){

	// Connect to the local CARLA server.
	auto client = cc::Client("localhost", 2000);
	// Set CARLA RPC timeout.
	client.SetTimeout(2s);
	// Get the active CARLA world.
	auto world = client.GetWorld();

	// Get all spawnable actor blueprints.
	auto blueprint_library = world.GetBlueprintLibrary();
	// Filter blueprints to vehicle types.
	auto vehicles = blueprint_library->Filter("vehicle");

	// Get the active CARLA map.
	auto map = world.GetMap();
	// Select a recommended spawn pose.
	auto transform = map->GetRecommendedSpawnPoints()[1];
	// Spawn the ego vehicle actor.
	auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

	//Create lidar
	// Fetch the lidar sensor blueprint.
	auto lidar_bp = *(blueprint_library->Find("sensor.lidar.ray_cast"));
	// Existing note for this section.
	// CANDO: Can modify lidar values to get different scan resolutions
	// Set lidar upper vertical angle.
	lidar_bp.SetAttribute("upper_fov", "15");
    // Set lidar lower vertical angle.
    lidar_bp.SetAttribute("lower_fov", "-25");
    // Set lidar vertical scan channels.
    lidar_bp.SetAttribute("channels", "32");
    // Set lidar max range.
    lidar_bp.SetAttribute("range", "30");
	// Set lidar rotations per second.
	lidar_bp.SetAttribute("rotation_frequency", "60");
	// Set lidar point density.
	lidar_bp.SetAttribute("points_per_second", "500000");

	// Define optional lidar placement offset.
	auto user_offset = cg::Location(0, 0, 0);
	// Place lidar relative to the ego vehicle.
	auto lidar_transform = cg::Transform(cg::Location(-0.5, 0, 1.8) + user_offset);
	// Spawn lidar attached to the ego vehicle.
	auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, ego_actor.get());
	// Cast the lidar actor to a sensor.
	auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);
	// Track whether the program is waiting for a scan.
	bool new_scan = true;
	// Declare timestamps used for scan timing.
	std::chrono::time_point<std::chrono::system_clock> lastScanTime, startTime;

	// Create the PCL 3D viewer.
	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  	// Set the viewer background color.
  	viewer->setBackgroundColor (0, 0, 0);
	// Register keyboard input handling.
	viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

	// Cast the ego actor to a vehicle.
	auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);
	// Initialize the estimated ego pose.
	Pose pose(Point(0,0,0), Rotate(0,0,0));

	// Existing note for this section.
	// Load map
	// Allocate the map point cloud.
	PointCloudT::Ptr mapCloud(new PointCloudT);
  	// Load the saved map from disk.
  	pcl::io::loadPCDFile("map.pcd", *mapCloud);
  	// Report how many map points were loaded.
  	cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
	// Render the map in blue.
	renderPointCloud(viewer, mapCloud, "map", Color(0,0,1)); 

	// Allocate a filtered scan cloud.
	typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
	// Allocate the current scan cloud.
	typename pcl::PointCloud<PointT>::Ptr scanCloud (new pcl::PointCloud<PointT>);
	// Reuse the transformed scan cloud across frames to avoid per-scan heap allocation.
	typename pcl::PointCloud<PointT>::Ptr transformedScan (new pcl::PointCloud<PointT>);

	// Register the lidar callback.
	lidar->Listen([&new_scan, &lastScanTime, &scanCloud](auto data){

		// Only collect points when a new scan is needed.
		if(new_scan){
			// Cast incoming data to lidar measurements.
			auto scan = boost::static_pointer_cast<csd::LidarMeasurement>(data);
			// Iterate over lidar returns.
			for (auto detection : *scan){
				// Ignore points too close to the ego vehicle.
				if((detection.x*detection.x + detection.y*detection.y + detection.z*detection.z) > 8.0){
					// Add the lidar point to the scan.
					pclCloud.points.push_back(PointT(detection.x, detection.y, detection.z));
				// End block.
				}
			// End block.
			}
			// Stop collecting once enough points exist.
			if(pclCloud.points.size() > 5000){ // CANDO: Can modify this value to get different scan resolutions
				// Record when the scan completed.
				lastScanTime = std::chrono::system_clock::now();
				// Copy accumulated points into the scan cloud.
				*scanCloud = pclCloud;
				// Mark the scan as ready for processing.
				new_scan = false;
			// End block.
			}
		// End block.
		}
	// Finish the lidar callback registration.
	});
	
	// Initialize the estimated ego pose.
	Pose poseRef(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180));
	// Track the largest localization error.
	double maxError = 0;
	int n_scans = 0;
	vector<double> xs;
	long unsigned int xs_max_size = 6;

	// Run until the viewer closes.
	while (!viewer->wasStopped())
  	// Begin block.
  	{
		// Wait until the callback fills a scan.
		while(new_scan){
			// Pause briefly while waiting.
			std::this_thread::sleep_for(0.1s);
			// Advance the CARLA simulation.
			world.Tick(1s);
		// End block.
		}
		// Handle a requested camera recenter.
		if(refresh_view){
			// Move the camera above the estimated pose.
			viewer->setCameraPosition(pose.position.x, pose.position.y, 60, pose.position.x+1, pose.position.y+1, 0, 0, 0, 1);
			// Clear the camera refresh request.
			refresh_view = false;
		// End block.
		}
		
		// Remove the previous true-pose box outline.
		viewer->removeShape("box0");
		// Remove the previous true-pose box fill.
		viewer->removeShape("boxFill0");
		// Compute true pose relative to the start.
		Pose truePose = Pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180)) - poseRef;
		// Draw the true car pose in red.
		drawCar(truePose, 0,  Color(1,0,0), 0.7, viewer);
		// Cache the true yaw angle.
		double theta = truePose.rotation.yaw;
		// Compute the steered heading angle.
		double stheta = control.steer * pi/4 + theta;
		// Remove the old steering ray.
		viewer->removeShape("steer");
		// Draw the steering direction ray.
		renderRay(viewer, Point(truePose.position.x+2*cos(theta), truePose.position.y+2*sin(theta),truePose.position.z),  Point(truePose.position.x+4*cos(stheta), truePose.position.y+4*sin(stheta),truePose.position.z), "steer", Color(0,1,0));


		// Default to braking if no command exists.
		ControlState accuate(0, 0, 1);
		// Check whether keyboard input is queued.
		if(cs.size() > 0){
			// Use the most recent keyboard command.
			accuate = cs.back();
			// Discard older queued commands.
			cs.clear();

			// Update the vehicle control state.
			Accuate(accuate, control);
			// Send control command to CARLA.
			vehicle->ApplyControl(control);
		// End block.
		}

  		// Refresh the viewer once.
  		viewer->spinOnce ();
		
		// Process a completed scan.
		if(!new_scan){
			
			// Allow the next scan to be collected.
			// Downsample the raw lidar scan to reduce ICP input size and improve matching speed.
			new_scan = true;
			if(n_scans == 0){
				pose.position = truePose.position;
				pose.rotation = truePose.rotation;
			}
			n_scans++;

			const double leafSize = 0.5f;
			pcl::VoxelGrid<PointT> voxelFilter;
			voxelFilter.setInputCloud(scanCloud);
			voxelFilter.setLeafSize(leafSize, leafSize, leafSize);
			voxelFilter.filter(*cloudFiltered);
			
			double avg_speed = compute_average_speed_per_iteration(xs);
			Result result = USE_NDT ? NDT(mapCloud, cloudFiltered, pose, avg_speed) : ICP(mapCloud, cloudFiltered, pose, avg_speed);
			pose = getPose(result.transform);
			xs.push_back(pose.position.x);
			if(xs.size() > xs_max_size)
				xs.erase(xs.begin());

			// Existing note for this section.
			//Transform scan so it aligns with ego's actual pose and render that scan
			Eigen::Matrix4f lidarToMap = (transform3D(
				pose.rotation.yaw,
				pose.rotation.pitch,
				pose.rotation.roll,
				pose.position.x,
				pose.position.y,
				pose.position.z) * getLidarToVehicleTransform()).cast<float>();
			pcl::transformPointCloud(*cloudFiltered, *transformedScan, lidarToMap);

			// Remove the previous rendered scan.
			viewer->removePointCloud("scan");
			// Existing note for this section.
			// TODO: Change `scanCloud` below to your transformed scan
			// Render the current scan in red.
			renderPointCloud(viewer, transformedScan, "scan", Color(1,0,0) );

			// Clear all temporary shapes.
			viewer->removeAllShapes();
			// Draw the estimated car pose in green.
			drawCar(pose, 1,  Color(0,1,0), 0.35, viewer);
          
          	// Compute horizontal localization error.
          	double poseError = sqrt( (truePose.position.x - pose.position.x) * (truePose.position.x - pose.position.x) + (truePose.position.y - pose.position.y) * (truePose.position.y - pose.position.y) );
			// Check for a new maximum error.
			if(poseError > maxError)
				// Update maximum observed error.
				maxError = poseError;
			// Compute distance from the start.
			double distDriven = sqrt( (truePose.position.x) * (truePose.position.x) + (truePose.position.y) * (truePose.position.y) );
			// Remove old max-error text.
			viewer->removeShape("maxE");
			// Display maximum error.
			viewer->addText("Max Error: "+to_string(maxError)+" m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE",0);
			// Remove old current-error text.
			viewer->removeShape("derror");
			// Display current pose error.
			viewer->addText("Pose error: "+to_string(poseError)+" m", 200, 150, 32, 1.0, 1.0, 1.0, "derror",0);
			// Remove old distance text.
			viewer->removeShape("dist");
			// Display distance driven.
			viewer->addText("Distance: "+to_string(distDriven)+" m", 200, 200, 32, 1.0, 1.0, 1.0, "dist",0);

			// Check whether evaluation should finish.
			if(maxError > 1.2 || distDriven >= 170.0 ){
				// Remove old evaluation label.
				viewer->removeShape("eval");
			// Check whether evaluation should finish.
			if(maxError > 1.2){
				// Display failure result.
				viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval",0);
			// End block.
			}
			// Handle the alternative branch.
			else{
				// Display success result.
				viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval",0);
			// End block.
			}
		// End block.
		}

			// Clear accumulated lidar points.
			pclCloud.points.clear();
		// End block.
		}
  	// End block.
  	}
	// Exit successfully.
	return 0;
// End block.
}
