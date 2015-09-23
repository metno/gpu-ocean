#############################################################################
#                                                                           #
#                                                                           #
# (c) Copyright 2010, 2011, 2012 by                                         #
#     SINTEF, Oslo, Norway                                                  #
#     All rights reserved.                                                  #
#                                                                           #
#  THIS SOFTWARE IS FURNISHED UNDER A LICENSE AND MAY BE USED AND COPIED    #
#  ONLY IN  ACCORDANCE WITH  THE  TERMS  OF  SUCH  LICENSE  AND WITH THE    #
#  INCLUSION OF THE ABOVE COPYRIGHT NOTICE. THIS SOFTWARE OR  ANY  OTHER    #
#  COPIES THEREOF MAY NOT BE PROVIDED OR OTHERWISE MADE AVAILABLE TO ANY    #
#  OTHER PERSON.  NO TITLE TO AND OWNERSHIP OF  THE  SOFTWARE IS  HEREBY    #
#  TRANSFERRED.                                                             #
#                                                                           #
#  SINTEF  MAKES NO WARRANTY  OF  ANY KIND WITH REGARD TO THIS SOFTWARE,    #
#  INCLUDING,   BUT   NOT   LIMITED   TO,  THE  IMPLIED   WARRANTIES  OF    #
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.                    #
#                                                                           #
#  SINTEF SHALL NOT BE  LIABLE  FOR  ERRORS  CONTAINED HEREIN OR DIRECT,    #
#  SPECIAL,  INCIDENTAL  OR  CONSEQUENTIAL  DAMAGES  IN  CONNECTION WITH    #
#  FURNISHING, PERFORMANCE, OR USE OF THIS MATERIAL.                        #
#                                                                           #
#                                                                           #
#############################################################################

# This module looks for NetCDF support.
# It will define the following variables
#  NetCDF_INCLUDE_DIR = Where atlas_level1.h can be found
#  NetCDF_LIBRARIES   = The libraries to link with

#Find include path
FIND_PATH(NetCDF_INCLUDE_DIR 
  NAMES netcdf.h
  PATHS
  ${NetCDF_ROOT}/include
  /usr/include
)

#Find libraries
FIND_LIBRARY(NetCDF_C_LIBRARY
  NAMES netcdf
  PATHS 
  ${NetCDF_ROOT}
  ${NetCDF_ROOT}/lib
  ${NetCDF_ROOT}/lib64
  /usr/lib
  /usr/lib64
)
FIND_LIBRARY(NetCDF_CXX_LIBRARY
  NAMES netcdf_c++
  PATHS 
  ${NetCDF_ROOT}
  ${NetCDF_ROOT}/lib
  ${NetCDF_ROOT}/lib64
  /usr/lib
  /usr/lib64
)
SET(NetCDF_LIBRARIES ${NetCDF_C_LIBRARY} ${NetCDF_CXX_LIBRARY} CACHE STRING "Netcdf libraries" FORCE)

IF(NetCDF_INCLUDE_DIR AND NetCDF_LIBRARIES)
  SET(NetCDF_FOUND TRUE)
ELSE()
  SET(NetCDF_FOUND FALSE)
ENDIF()

MARK_AS_ADVANCED(
  NetCDF_INCLUDE_DIR
  NetCDF_C_LIBRARY
  NetCDF_CXX_LIBRARY
  NetCDF_LIBRARIES
)
