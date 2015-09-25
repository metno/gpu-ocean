#include "oclutils.h"
#include <iostream>

///XXX
//#define NDEBUG

using namespace std;

/**
 * Finds all available OpenCL platforms on the current node.
 * @param clPlatformIDs Output: Vector of detected platforms
 * @return Number of detected platforms
 */
cl_uint OpenCLUtils::getOCLPlatforms(vector<cl_platform_id> &clPlatformIDs)
{
	cl_uint countPlatforms;
	CL_CHECK(clGetPlatformIDs(0, NULL, &countPlatforms));

	clPlatformIDs.resize(countPlatforms);

	CL_CHECK(clGetPlatformIDs(countPlatforms, clPlatformIDs.data(), NULL));

#ifndef NDEBUG
    cout << "Number of available OpenCL platforms: " << countPlatforms << endl;
#endif

    return countPlatforms;
}

/**
 * Finds number of available devices for a given OpenCL platform.
 * @param clPlatformID Input: The platform to search
 * @return Number of detected devices for given platform
 */
cl_uint OpenCLUtils::countOCLDevices(cl_platform_id clPlatformID)
{
	cl_uint countDevices;

	CL_CHECK(clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &countDevices));

#ifndef NDEBUG
	cout << "Number of available OpenCL devices: " << countDevices << endl;
#endif

    return countDevices;
}
