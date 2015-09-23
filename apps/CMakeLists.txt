PROJECT(apps)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#Set where executables end up
IF(NOT EXECUTABLE_OUTPUT_PATH)
	SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
ENDIF(NOT EXECUTABLE_OUTPUT_PATH)

FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem system thread program_options)
FIND_PACKAGE(OpenCL REQUIRED)
FIND_PACKAGE(NetCDF QUIET)

OPTION(apps_USE_NETCDF "Enable or disable NetCDF-support" ${NetCDF_FOUND})
OPTION(apps_USE_NETCDF_COMPRESSION "Enable or disable NetCDF compression-support (Requires new version of netcdf)" FALSE)

# - Find the mgr include dir
FIND_PATH(mgr_INCLUDE_DIR NAMES manager.h DOC "Path to directory where manager.h can be found")
FIND_LIBRARY(mgr_LIBRARIES NAMES mgr.lib mgr.a DOC "Path to directory where mgr libraries can be found")

# - Find the sim include dir
FIND_PATH(sim_INCLUDE_DIR NAMES simulator.h DOC "Path to directory where simulator.h can be found")
FIND_LIBRARY(sim_LIBRARIES NAMES sim.lib sim.a DOC "Path to directory where sim libraries can be found")

INCLUDE_DIRECTORIES(
	${apps_SOURCE_DIR}/src
	${mgr_INCLUDE_DIR}
	${sim_INCLUDE_DIR}
	${OpenCL_INCLUDE_DIRS}
	${Boost_INCLUDE_DIR}
	${CMAKE_CURRENT_BINARY_DIR})
	
IF(apps_USE_NETCDF)
	INCLUDE_DIRECTORIES(${NetCDF_INCLUDE_DIR})
ENDIF()

LINK_DIRECTORIES(
	${Boost_LIBRARY_DIRS})

# - Create objects needed
FILE(GLOB apps_src "src/*.cpp")
ADD_LIBRARY(apps STATIC ${apps_src})

# - Create applications
FILE(GLOB apps_app src/*.cpp)
# MLS: No netCDF-reader/writer yet...
#LIST(REMOVE_ITEM apps_app "${apps_SOURCE_DIR}/src/NetCDFHandler.cpp")
FOREACH(filename ${apps_app})
  GET_FILENAME_COMPONENT(target ${filename} NAME_WE)
  SET(name "app_${target}")
  ADD_EXECUTABLE(${name} ${filename})
  TARGET_LINK_LIBRARIES(${name} 
	apps
	${sim_LIBRARIES}
	${mgr_LIBRARIES}
	${OPENCL_LIBRARIES})
	
  IF(apps_USE_NETCDF)
    TARGET_LINK_LIBRARIES(${name} ${NetCDF_LIBRARIES})
  ENDIF()

  IF(NOT CMAKE_HOST_WIN32)
    TARGET_LINK_LIBRARIES(${name}	${Boost_LIBRARIES})
  ENDIF()
  SET_TARGET_PROPERTIES(${name} PROPERTIES OUTPUT_NAME "${target}")
ENDFOREACH(filename ${apps_app})
