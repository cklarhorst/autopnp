#include <ros/ros.h>
#include <ros/package.h>
#include <ros/console.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <ipa_building_msgs/MapSegmentationAction.h>
#include <ipa_room_segmentation/dynamic_reconfigure_client.h>
#include <nav_msgs/OccupancyGrid.h>

/* Used by the mapCallback */
actionlib::SimpleActionClient<ipa_building_msgs::MapSegmentationAction>* ac;


/**
 * Convert an OccupancyGrid to a CVImage based on an occupancy threshold
 */
void convertOccupancyGrid2CvImage(const nav_msgs::OccupancyGridConstPtr& msg, cv_bridge::CvImage& image)
{
	cv::Mat img_data(msg->info.height, msg->info.width, CV_8UC1);
	image.encoding = "mono8";
	std::vector<signed char>::const_iterator mapDataIter = msg->data.begin();
	
	/* Min, Max only used for debugging (to tune the threshold) */
	unsigned char min=255;
	unsigned char max=0;
	unsigned char map_occ_thres = 200;
	
	for(unsigned int j = 0; j < msg->info.height; ++j)
	{ // (0,0) is lower left corner of OccupancyGrid
		for(unsigned int i = 0; i < msg->info.width; ++i)
		{
			const unsigned char tmp = (unsigned char)*mapDataIter;
			min = min<tmp?min:tmp;
			max = max>tmp?max:tmp;
			if (tmp < map_occ_thres)
			{
				img_data.at<uchar>(j,i) = 255;
			} else
			{
				img_data.at<uchar>(j,i) = 0;
			}
			++mapDataIter;
		}
	}
	ROS_INFO_STREAM("Occupancy range: " << (int)min << ", " << (int)max);
	image.image = img_data;
}

/**
 * Called when a new map is published, starts the segmentation process
 */
void mapCallback(const nav_msgs::OccupancyGridConstPtr& msg)
{ 
	ROS_INFO("Receive new map");
	ROS_DEBUG_STREAM("Header     : " << msg->header);
	ROS_DEBUG_STREAM("MapMetaData: " << msg->info);
	
	sensor_msgs::Image map;
	cv_bridge::CvImage cv_image;
	convertOccupancyGrid2CvImage(msg, cv_image);
	cv_image.toImageMsg(map);
	
	
	ipa_building_msgs::MapSegmentationGoal goal;
	goal.input_map = map;
	goal.map_origin = msg->info.origin;
	goal.map_resolution = msg->info.resolution;
	goal.return_format_in_meter = false;
	goal.return_format_in_pixel = true;
	goal.robot_radius = 0.4; //todo: set roboter size
	ac->sendGoal(goal);
	
	//wait for the action to return
	bool finished_before_timeout = ac->waitForResult(ros::Duration());
	
	if (finished_before_timeout)
	{
		ROS_INFO("Finished successfully!");
	}
	
	
 }

int main(int argc, char **argv)
{
	const char* map_publish_topic = "/map_merge/map";
	
	ros::init(argc, argv, "room_segmentation_client");
	ros::NodeHandle nh;
	
	ROS_INFO("Waiting for room segmentation server to start.");
	actionlib::SimpleActionClient<ipa_building_msgs::MapSegmentationAction> segmentation_server("room_segmentation_server", true);
	ac=&segmentation_server;
	
	ac->waitForServer(); //will wait for infinite time
	ROS_INFO("segmentation server started, reconfigure.");

	DynamicReconfigureClient drc(nh, "room_segmentation_server/set_parameters", "room_segmentation_server/parameter_updates");
	drc.setConfig("room_segmentation_algorithm", 3);
	drc.setConfig("display_segmented_map", false);
	drc.setConfig("publish_segmented_map", true);
	drc.setConfig("max_iterations", 25);
	drc.setConfig("max_area_for_merging", 5.0);
	//drc.setConfig("room_lower_limit_voronoi", 1.5);
	drc.setConfig("room_area_factor_lower_limit_voronoi", 3.0);	
	//drc.setConfig("min_critical_point_distance_factor", 1);	
	drc.setConfig("min_critical_point_distance_factor", 1.5);	
	drc.setConfig("voronoi_neighborhood_index", 800);	
	
	//drc.setConfig("room_area_factor_upper_limit_voronoi", 120.0);
	ROS_INFO("Waiting");
	ros::Subscriber sub = nh.subscribe(map_publish_topic, 1, mapCallback);
	ros::spin();
	return 0;
}
