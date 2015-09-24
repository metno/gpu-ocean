#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "programoptions.h"
#include "initconditions.h"
#include <boost/shared_ptr.hpp>

#include <vector>

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

// This class manages the low-level aspects of a simulation.
class Simulator
{
public:
    Simulator(const OptionsPtr &, const InitCondPtr &);
    virtual ~Simulator();
    void init();
    int nextStep() const;
    int finalStep() const;
    void execNextStep();
    void printStatus() const;

    /**
     * Find all available OpenCL platforms on the current node.
     * XXX: Extract this to OCL utils class
     * @param clPlatformIDs Output: Vector of detected platforms
     * @return Number of detected platforms
     */
    cl_uint getOCLPlatforms(std::vector<cl_platform_id> &clPlatformIDs);

    /**
     * Find number of available devices for a given OpenCL platform
     * XXX: Extract this to OCL utils class
     * @param clPlatformID Input: The platform to search
     * @return Number of detected devices for given platform
     */
    cl_uint countOCLDevices(cl_platform_id clPlatformID) const;

private:
    struct SimulatorImpl;
    SimulatorImpl *pimpl;
};

typedef boost::shared_ptr<Simulator> SimPtr;

#endif // SIMULATOR_H
