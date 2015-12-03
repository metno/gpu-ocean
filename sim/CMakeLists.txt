PROJECT(sim)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(OpenCL REQUIRED)
	
IF(NOT LIBRARY_OUTPUT_PATH)
	SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
ENDIF(NOT LIBRARY_OUTPUT_PATH)

INCLUDE_DIRECTORIES(
	${sim_SOURCE_DIR}/src
	${mgr_INCLUDE_DIR}
	${Boost_INCLUDE_DIR}
	${OpenCL_INCLUDE_DIRS}
	${CMAKE_CURRENT_BINARY_DIR})

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
	SET(sim_TARGET_CPU_ARCH "64")
ELSEIF(CMAKE_SIZEOF_VOID_P EQUAL 4)
	SET(sim_TARGET_CPU_ARCH "32")
ELSE()
	MESSAGE(ERROR "Could not deduce target architecture: sizeof(void*) is neither 4 or 8...")
ENDIF()

OPTION(sim_PROFILING "Enable profiling" FALSE)

SET(sim_WGNX 16 CACHE STRING "Width of work group")
SET(sim_WGNY 16 CACHE STRING "Height of work group")

#Get the working branch and revision hash (short) from git
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND git rev-parse --short --verify HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

SET(sim_SIMULATOR_MAJOR_VERSION "0" CACHE STRING "Specify the major version" FORCE)
SET(sim_SIMULATOR_MINOR_VERSION "1" CACHE STRING "Specify the minor version" FORCE)

#Create config.h from config.h.in
#file(REMOVE "${KPSIMULATOR_SOURCE_DIR}/configure.h")
configure_file("src/config.h.in" "src/config.h" ESCAPE_QUOTES)

ADD_LIBRARY(sim STATIC src/oclutils.cpp src/simbase.cpp src/simulator.cpp src/testsim.cpp)
# MLS: Do need this next line?
TARGET_LINK_LIBRARIES(sim ${OPENCL_LIBRARIES})
SET_TARGET_PROPERTIES(sim PROPERTIES DEBUG_POSTFIX "d")
INSTALL(TARGETS sim DESTINATION sim/lib)
INSTALL(DIRECTORY include DESTINATION sim PATTERN ".svn" EXCLUDE)
