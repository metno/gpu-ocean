#include "oclutils.h"
#include "TMP-defines.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "boost/format.hpp"

#undef NDEBUG
//#define NDEBUG

using namespace std;

/**
 * Public function for multiplying two NxN matrices.
 * Throws std::runtime_error if something goes wrong.
 * @param size Input: The size of N
 * @param execOnCpu Input: Whether to execute the kernel on the CPU
 */
void matmul(size_t size)
{
}
