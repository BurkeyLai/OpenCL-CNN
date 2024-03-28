#pragma once

#ifndef THESIS_INCLUDE_H
#define THESIS_INCLUDE_H

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200


//#include "CL/cl2.hpp"
//#include "OpenCL.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>



#endif //THESIS_INCLUDE_H
