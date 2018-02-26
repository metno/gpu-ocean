#ifndef CONFIG_H
#define CONFIG_H

#define SIMULATOR_MAJOR_VERSION "0"
#define SIMULATOR_MINOR_VERSION "0"
#define GIT_BRANCH ""
#define GIT_COMMIT_HASH ""

#define PROFILE ""

#if ! defined(WGNX) || ! defined(WGNY)

#define WGNX 16
#define WGNY 16
 
#if ! defined(block_width) || ! defined(block_height)
#define block_width WGNX
#define block_height WGNY
#endif

#endif

#define OPENCL_VERSION_MAJOR 2

#endif // CONFIG_H
