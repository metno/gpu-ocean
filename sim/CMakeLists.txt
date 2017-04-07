PROJECT(sim)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

if (COVERALLS)
    set(COVERAGE_SRCS src/simulator.cpp)

    # Create the coveralls target.
    coveralls_setup(
        "${COVERAGE_SRCS}" # The source files.
        ON                 # If we should upload.
        )
endif()

FIND_PACKAGE(OpenCL REQUIRED)

SET(sim_OPENCL_VERSION_MAJOR ${OpenCL_VERSION_MAJOR} CACHE STRING "OpenCL (major) version")

INCLUDE(GetGitRevisionDescription)

IF(NOT LIBRARY_OUTPUT_PATH)
	SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
ENDIF(NOT LIBRARY_OUTPUT_PATH)

INCLUDE_DIRECTORIES(
	${sim_SOURCE_DIR}/src
	${mgr_INCLUDE_DIR}
	${Boost_INCLUDE_DIR}
	${OpenCL_INCLUDE_DIRS}
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_BINARY_DIR})

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
#execute_process(
#  COMMAND git rev-parse --short --verify HEAD
#  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#  OUTPUT_VARIABLE GIT_COMMIT_HASH
#  OUTPUT_STRIP_TRAILING_WHITESPACE
#)
# using this module ensures cmake reconfigure
get_git_head_revision(GIT_REFSPEC GIT_COMMIT_HASH)

SET(sim_SIMULATOR_MAJOR_VERSION "0" CACHE STRING "Specify the major version" FORCE)
SET(sim_SIMULATOR_MINOR_VERSION "1" CACHE STRING "Specify the minor version" FORCE)

#Create config.h from config.h.in
configure_file("src/config.h.in" ${CMAKE_CURRENT_SOURCE_DIR}/src/config.h ESCAPE_QUOTES)

# copy OpenCL kernel source code to build directory (will be compiled on run)
if (NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    # install all OpenCL-kernels
    file(GLOB sim_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cl")
    foreach(item IN LISTS sim_KERNELS)
    	get_filename_component(itemName ${item} NAME)
        message(STATUS ${itemName})
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kernels/${itemName}"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/${itemName}" "${CMAKE_CURRENT_BINARY_DIR}/kernels/${itemName}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/${itemName}"
        )
        
    	add_custom_target("install_kernel_${itemName}" ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/kernels/${itemName}")
    endforeach()
    
    # install all OpenCL-kernels' type definitions
    file(GLOB sim_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/src/*_types.h")
    foreach(item IN LISTS sim_KERNELS)
    	get_filename_component(itemName ${item} NAME)
        message(STATUS ${itemName})
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${itemName}"
            COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/src/${itemName}" "${CMAKE_CURRENT_BINARY_DIR}/${itemName}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${itemName}"
        )
        
    	add_custom_target("install_kernel_${itemName}" ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${itemName}")
    endforeach()
    
    # install config.h for out-of-source builds
    message(STATUS "config.h")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/config.h"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h" "${CMAKE_CURRENT_BINARY_DIR}/config.h"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h"
    )
    add_custom_target("install_config_h" ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/config.h")
endif()

ADD_LIBRARY(sim STATIC src/oclutils.cpp src/simbase.cpp src/simulator.cpp src/testsim.cpp)
# MLS: Do we need this next line?
TARGET_LINK_LIBRARIES(sim ${OPENCL_LIBRARIES})
SET_TARGET_PROPERTIES(sim PROPERTIES DEBUG_POSTFIX "d")
INSTALL(TARGETS sim DESTINATION sim/lib)
INSTALL(DIRECTORY include DESTINATION sim PATTERN ".svn" EXCLUDE)
