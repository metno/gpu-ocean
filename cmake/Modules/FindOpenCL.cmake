# This module looks for OpenCL support.
# It will define the following variables
#  OPENCL_FOUND	      = System has OpenCL
#  OPENCL_INCLUDE_DIR = Where cl.h can be found
#  OPENCL_LIBRARIES   = The libraries to link with

# ( Note: On pc4131, set OPENCL_ROOT to /disk1/usr/local/cuda )

# Find include path
FIND_PATH(OPENCL_INCLUDE_DIR 
  NAMES cl.h
  PATHS
  ${OPENCL_ROOT}/include/CL
  /usr/include/CL
)

# Find libraries
FIND_LIBRARY(OPENCL_LIBRARIES
  NAMES OpenCL
  PATHS 
  ${OPENCL_ROOT}/lib64
  /usr/lib
  /usr/lib64
)

SET( OPENCL_FOUND "NO" )

IF(OPENCL_LIBRARIES )
	SET( OPENCL_FOUND "YES" )
ENDIF(OPENCL_LIBRARIES)

MARK_AS_ADVANCED(
  OPENCL_INCLUDE_DIR
)
