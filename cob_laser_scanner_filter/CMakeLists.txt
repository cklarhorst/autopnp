cmake_minimum_required(VERSION 2.8.3)
project(cob_laser_scanner_filter)

set(catkin_RUN_PACKAGES
	roscpp
	sensor_msgs
)

set(catkin_BUILD_PACKAGES
	${catkin_RUN_PACKAGES}
	cmake_modules
)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
## Creates a bunch of environment variables that may be used later in the script
## e.g. catkin_INCLUDE_DIRS
find_package(catkin REQUIRED COMPONENTS
	${catkin_BUILD_PACKAGES}
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
	INCLUDE_DIRS
	LIBRARIES
	CATKIN_DEPENDS
		${catkin_RUN_PACKAGES}
	DEPENDS
)

###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
include_directories(
	${catkin_INCLUDE_DIRS}
)

add_executable(filter src/filter.cpp)
target_link_libraries(filter
	${catkin_LIBRARIES}
)
add_dependencies(filter ${catkin_EXPORTED_TARGETS})


#add_executable(check src/check.cpp)
#target_link_libraries(check
#	${catkin_LIBRARIES}
#)
#add_dependencies(check ${catkin_EXPORTED_TARGETS})


#############
## Install ##
#############
## Mark executables and/or libraries for installation
install(TARGETS filter #check
	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#install(DIRECTORY common/include/${PROJECT_NAME}/
#	DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#	FILES_MATCHING PATTERN "*.h"
#	PATTERN ".svn" EXCLUDE
#)

install(DIRECTORY launch
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
	PATTERN ".svn" EXCLUDE
)

#install(DIRECTORY common/files
#	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/common
#	PATTERN ".svn" EXCLUDE
#)
