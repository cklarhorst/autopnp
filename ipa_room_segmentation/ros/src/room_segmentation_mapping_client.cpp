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
#include <ipa_room_segmentation/A_star_pathplanner.h>
#include <iostream>
#include <filesystem>
#include <boost/algorithm/string.hpp>

struct Poi {
	double centerX, centerY;
	double size;
	std::string label;
	cv::Scalar color;
	Poi(double centerX, double centerY, double size, std::string label, cv::Scalar color)
    {
		this->centerX = centerX;
		this->centerY = centerY;
		this->size  = size;
		this->label = label;
		this->color = color;
    }
};


/* Used by the mapCallback */
actionlib::SimpleActionClient<ipa_building_msgs::MapSegmentationAction>* ac;
bool writeDebugImages = false;
bool enableAStar      = false;
bool printTaskTable   = false;
std::vector<Poi> pois;
std::string poiDescription; //Used to define the pois
using namespace cv;

/**
 * Convert an OccupancyGrid to a CVImage based on an occupancy threshold
 */
static void convertOccupancyGrid2CvImageTrinary(const nav_msgs::OccupancyGridConstPtr& msg, cv_bridge::CvImage& wall, cv_bridge::CvImage& wall_and_unknown)
{
	cv::Mat img_wall(msg->info.height, msg->info.width, CV_8UC1);
	cv::Mat img_wall_unknown(msg->info.height, msg->info.width, CV_8UC1);
	wall.encoding = "mono8";
	wall_and_unknown.encoding = "mono8";
	std::vector<signed char>::const_iterator mapDataIter = msg->data.begin();
	
	/* Min, Max only used for debugging (to tune the threshold) */
	unsigned char min=255;
	unsigned char max=0;
	const unsigned char map_occ_thres_walls=100;
	const unsigned char map_occ_thres_unknown=255;
	
	for(unsigned int j = 0; j < msg->info.height; ++j)
	{ // (0,0) is lower left corner of OccupancyGrid
		for(unsigned int i = 0; i < msg->info.width; ++i)
		{
			const unsigned char tmp = (unsigned char)*mapDataIter;
			min = min<tmp?min:tmp;
			max = max>tmp?max:tmp;
			if (tmp==map_occ_thres_walls) 
			{
				img_wall.at<uchar>(j,i) = 0;
				img_wall_unknown.at<uchar>(j,i) = 255; //was 0
			} else if (tmp==map_occ_thres_unknown) 
			{
				img_wall.at<uchar>(j,i) = 255;
				img_wall_unknown.at<uchar>(j,i) = 0;
			} else 
			{
				img_wall.at<uchar>(j,i) = 255;
				img_wall_unknown.at<uchar>(j,i) = 255;

			}
			++mapDataIter;
		}
	}
	ROS_INFO_STREAM("Occupancy range: " << (int)min << ", " << (int)max);
	wall.image = img_wall;
	wall_and_unknown.image = img_wall_unknown;
}

/**
 * Preprocess map
 * - Multiple Erode/Dilate steps
 * - Remove small wall noise
 */
static void preprocess_map(const nav_msgs::OccupancyGridConstPtr& msg, cv::Mat& cv_image_walls_and_unknown_cleaned_combined)
{
	cv_bridge::CvImage cv_image_walls;
	cv_bridge::CvImage cv_image_walls_and_unknown;
	convertOccupancyGrid2CvImageTrinary(msg, cv_image_walls, cv_image_walls_and_unknown);

	if (writeDebugImages) 
	{
		cv::imwrite("walls.png", cv_image_walls.image);
		cv::imwrite("walls_and_unknown.png", cv_image_walls_and_unknown.image);
	}

	cv::Mat cv_image_walls_and_unknown_cleaned;
	cv::Mat cv_image_tmp;
	cv::Mat cv_image_walls_expanded;
	cv::Mat cv_image_tmp2;
	cv::Mat cv_image_tmp3;
	cv::Mat cv_image_tmp4;
	cv::erode(cv_image_walls_and_unknown.image, cv_image_tmp2, cv::Mat(), cv::Point(-1, -1), 1);
	cv::dilate(cv_image_tmp2, cv_image_tmp4, cv::Mat(), cv::Point(-1, -1), 2);
	cv::erode(cv_image_tmp4, cv_image_walls_and_unknown_cleaned, cv::Mat(), cv::Point(-1, -1), 1);	
        cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                       cv::Size( 2*1 + 1, 2*1+1 ),
                       cv::Point( 1, 1 ) ); 

	cv::erode(cv_image_walls.image, cv_image_tmp3, element, cv::Point(-1, -1), 4);
	cv::dilate(cv_image_tmp3, cv_image_walls_expanded, element, cv::Point(-1, -1), 2);
	if (writeDebugImages)
		cv::imwrite("walls_and_unknown_cleaned.png", cv_image_walls_and_unknown_cleaned);
	
	//Remove small wall noise
	SimpleBlobDetector::Params params;
	params.minThreshold = 0;
	params.maxThreshold = 200;
	params.blobColor = 0;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByCircularity = false;
	params.filterByColor = true;
	params.filterByArea = true;
	params.minArea = 1;
	params.maxArea = 400;
	params.minRepeatability = 1;
	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);;
	std::vector<KeyPoint> keypoints;
	detector->detect( cv_image_walls_expanded, keypoints);
	cv_image_tmp = cv_image_walls_expanded.clone();
	for (KeyPoint k : keypoints) 
	{
		circle(cv_image_tmp, k.pt,k.size, cv::Scalar(255), CV_FILLED, 0);
	}
	
	cv::bitwise_and(cv_image_walls_and_unknown_cleaned,cv_image_walls_and_unknown_cleaned,cv_image_walls_and_unknown_cleaned_combined,cv_image_tmp);

	if (writeDebugImages) 
	{
		ROS_INFO_STREAM("Wall noise remove size: " << keypoints.size());
		cv::imwrite("remove_noise.png", cv_image_tmp);
		cv::imwrite("walls_expanded.png", cv_image_walls_expanded);	
		cv::imwrite("walls_unkown_cleaned_combined.png", cv_image_walls_and_unknown_cleaned_combined);
		cv::imwrite("used_for_segmentation.png", cv_image_walls_and_unknown_cleaned_combined);
	}
}

static void runAStarOnSegmentedMap(ipa_building_msgs::MapSegmentationResultConstPtr result_seg, cv_bridge::CvImage cv_image, cv::Mat& segmented_map, double map_resolution)
{
	AStarPlanner a_star_path_planner;
	int unknown=0;
	double min, max;
	cv::Mat colour_segmented_map = segmented_map.clone();
	colour_segmented_map.convertTo(colour_segmented_map, CV_8U);
	colour_segmented_map.setTo(0, colour_segmented_map==255);
	cv::normalize(colour_segmented_map,colour_segmented_map,255,0,cv::NORM_MINMAX);
	cv::minMaxLoc(colour_segmented_map,&min,&max);
	cv::cvtColor(colour_segmented_map, colour_segmented_map, CV_GRAY2BGR);
	//colour_segmented_map = Scalar(0,0,0);
	cv::Mat cleaned_map;
	cv::Mat downsampled_map;
	cv::dilate(cv_image.image, cleaned_map, cv::Mat(), cv::Point(-1, -1), 1);
	
	size_t nb_rooms = result_seg->room_information_in_pixel.size()+1;
	unsigned int room_sizes[nb_rooms]={0};
	for(size_t u = 0; u < segmented_map.rows; ++u)
	{
		for(size_t v = 0; v < segmented_map.cols; ++v)
		{
			unsigned int id = segmented_map.at<int>(u,v);
			if (id>result_seg->room_information_in_pixel.size())
				++unknown;
				//ROS_ERROR_STREAM("Room size calculator: Out of range: " << id << std::endl);
			else
				++room_sizes[id];
		}
	}
	ROS_INFO_STREAM("#"<<unknown<<" px belong to no rooms?!?");
	/* Which center should we use?
	for (size_t src = 0; src < result_seg->room_information_in_pixel.size(); ++src) { // not nb_rooms
		std::ostringstream str;
		str << "R" << src;
		pointNames[src] = str.str();
		cv::Mat tmp;
		cv::Vec3b c = colour_segmented_map.at<cv::Vec3b>(result_seg->room_information_in_pixel[src].room_center.y,result_seg->room_information_in_pixel[src].room_center.x);
		cv::inRange(colour_segmented_map,c,c, tmp );
		cv::Moments m = moments(tmp,true);
		centers[src].x = m.m10/m.m00;
		centers[src].y = m.m01/m.m00;
		std::cout << "Room: " << src << ":" << c << ":" << centers[src].x << "," << centers[src].y << std::endl;
	}*/

	a_star_path_planner.downsampleMap(cleaned_map, downsampled_map, 1.0, 4.0, 1.0);
	if (writeDebugImages)
		cv::imwrite("downsampled_map_for_a_star.png", downsampled_map);
	std::cout << "#" << result_seg->room_information_in_pixel.size() << " final rooms" << std::endl;
	
	std::vector<Poi> roomsAndPois;
	
	// Add rooms
	for(size_t i = 0; i < result_seg->room_information_in_pixel.size(); ++i)
	{
		roomsAndPois.emplace_back(result_seg->room_information_in_pixel[i].room_center.x, result_seg->room_information_in_pixel[i].room_center.y, room_sizes[i+1] * map_resolution * map_resolution, std::string("room"), CV_RGB (255,0,0));
	}
	ROS_INFO_STREAM("#Rooms: "<<roomsAndPois.size());
	// Add Pois
	for(Poi poi : pois)
	{
		roomsAndPois.push_back(poi);
	}
	ROS_INFO_STREAM("#Poi: "<<pois.size());
	
	for(size_t src = 0; src < roomsAndPois.size(); ++src)
	{
		int src_x = roomsAndPois[src].centerX;
		int src_y = roomsAndPois[src].centerY;
		for(size_t dst = 0; dst < roomsAndPois.size(); ++dst)
		{
			if (src==dst)
			{
				if (printTaskTable)
				{
					std::cout << std::fixed << std::setprecision(2) << roomsAndPois[src].size << ",";
				}
				continue;
			}
			int dst_x = roomsAndPois[dst].centerX;
			int dst_y = roomsAndPois[dst].centerY;
			a_star_path_planner.m = downsampled_map.rows;// horizontal size of the map
			a_star_path_planner.n = downsampled_map.cols;// vertical size size of the map
			std::string path = a_star_path_planner.pathFind(src_x, src_y, dst_x, dst_y, downsampled_map);
			a_star_path_planner.drawRoute(colour_segmented_map,cv::Point(src_x,src_y),path,1.0);
			if (printTaskTable)
				std::cout << std::fixed << std::setprecision(2) << (path.size() *  map_resolution) << ",";
		}
		cv::circle(colour_segmented_map, cv::Point(src_x,src_y), 4, roomsAndPois[src].color, 2);
		if (printTaskTable)
			std::cout << std::endl;
	}
	if (writeDebugImages)
	{
		cv::imwrite("segmentated_map_with_pathes.png",colour_segmented_map);	
	}
}

static void processSegmentedMap(cv_bridge::CvImage cv_image, double map_resolution)
{
	ipa_building_msgs::MapSegmentationResultConstPtr result_seg = ac->getResult();
	cv_bridge::CvImagePtr cv_ptr_obj;
	cv_ptr_obj = cv_bridge::toCvCopy(result_seg->segmented_map, sensor_msgs::image_encodings::TYPE_32SC1);
	double min, max;
	cv::minMaxLoc(cv_ptr_obj->image,&min,&max);
	cv::Mat segmented_map = cv_ptr_obj->image;
	
	if (writeDebugImages)
		cv::imwrite("segmented_map.png",segmented_map);
	if (enableAStar) 
	{
		runAStarOnSegmentedMap(result_seg, cv_image, segmented_map, map_resolution);
	}
}



/**
 * Called when a new map is published, starts the segmentation process
 */
void mapCallback(const nav_msgs::OccupancyGridConstPtr& msg)
{ 
	ROS_INFO("Receive new map");
	
	pois.clear();
	std::vector<std::string> poisString;
	boost::split(poisString,poiDescription,boost::is_any_of(";"));
	for (std::string poiString : poisString)
	{
		std::vector<std::string> poisData;
		boost::split(poisData,poiString,boost::is_any_of(","));
		if (poisData.size()!=7) 
		{
			ROS_INFO_STREAM("Error: no valid POI Data: " << poiString << "size: " << poisData.size());
			continue;
		}
		pois.emplace_back(std::stod(poisData[0]),std::stod(poisData[1]),std::stod(poisData[2]),poisData[3],CV_RGB(std::stoi(poisData[4]),std::stoi(poisData[5]),std::stoi(poisData[6])));	
	}
	ROS_INFO_STREAM("#pois loaded: "<<pois.size());
	ROS_INFO_STREAM("poi string: " <<poiDescription);
	ROS_DEBUG_STREAM("Header     : " << msg->header);
	ROS_DEBUG_STREAM("MapMetaData: " << msg->info);
	if (writeDebugImages)
		ROS_INFO_STREAM("Current path is " << getcwd(NULL, 0) << "\n"); //will leak mem
	sensor_msgs::Image map;

	cv::Mat cv_image_walls_and_unknown_cleaned_combined;
	preprocess_map(msg,cv_image_walls_and_unknown_cleaned_combined);
	
	cv_bridge::CvImage finalImage;
	finalImage.encoding = "mono8";
	finalImage.image = cv_image_walls_and_unknown_cleaned_combined;
	finalImage.toImageMsg(map);
	cv_bridge::CvImage cv_image = finalImage;
	
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
		processSegmentedMap(cv_image, msg->info.resolution);
	}
	

 }

/* Main: Starts the Room Segmentation Client */
int main(int argc, char **argv)
{
	ros::init(argc, argv, "room_segmentation_client");
	ros::NodeHandle nh;
	
	std::string map_publish_topic;
	nh.param<std::string>("/room_segmentation_client/map_publish_topic", map_publish_topic, "/map");
	ROS_INFO("start");
	nh.param("/room_segmentation_client/write_debug_images", writeDebugImages, true);
	nh.param("/room_segmentation_client/enable_a_star", enableAStar, true);
	nh.param("/room_segmentation_client/print_task_table", printTaskTable, true);
	nh.param("/room_segmentation_client/poi_description", poiDescription, std::string(""));
	ROS_INFO_STREAM("test" << writeDebugImages);

	ROS_INFO("Waiting for room segmentation server to start.");
	actionlib::SimpleActionClient<ipa_building_msgs::MapSegmentationAction> segmentation_server("room_segmentation_server", true);
	ac=&segmentation_server;
	
	ac->waitForServer(); //will wait for infinite time
	ROS_INFO("segmentation server started, reconfigure.");

	DynamicReconfigureClient drc(nh, "room_segmentation_server/set_parameters", "room_segmentation_server/parameter_updates");
	drc.setConfig("room_segmentation_algorithm", 3);
	drc.setConfig("display_segmented_map", false);
	drc.setConfig("publish_segmented_map", true);
	drc.setConfig("max_iterations", 200);
	drc.setConfig("max_area_for_merging", 100.0); //100.0 citec 
	drc.setConfig("room_area_factor_lower_limit_voronoi", 2.5);
	drc.setConfig("min_critical_point_distance_factor", 1.5);
	drc.setConfig("voronoi_neighborhood_index", 100);	//100 citec-single-corridor; 500 citec-split-corridors
	ROS_INFO("Waiting");
	ros::Subscriber sub = nh.subscribe(map_publish_topic, 1, mapCallback);
	ros::spin();
	return 0;
}
