#pragma once

#define NOMINMAX					// Use standard library min/max
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>				// For writing debug info to the output windows, remove if you do not need it.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <ppl.h>
#include <string>
#include <vector>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <thrust\copy.h>
#include <thrust\device_vector.h>
#include <thrust\extrema.h>
#include <thrust\functional.h>
#include <thrust\iterator\constant_iterator.h>
#include <thrust\reduce.h>
#include <thrust\scan.h>
#include <thrust\sequence.h>
#include <thrust\sort.h>
#include <thrust\transform.h>

// My headers
#include "GpuTimer.h"
#include "Utilities.h"

// Load libraries
#pragma comment(lib, "cudart")

// opencv requires debug libraries when running indebug mode
#if _DEBUG
#pragma comment(lib, "opencv_core243d")
#pragma comment(lib, "opencv_imgproc243d")
#pragma comment(lib, "opencv_highgui243d")
#pragma comment(lib, "opencv_ml243d")
#pragma comment(lib, "opencv_video243d")
#pragma comment(lib, "opencv_features2d243d")
#pragma comment(lib, "opencv_calib3d243d")
#pragma comment(lib, "opencv_objdetect243d")
#pragma comment(lib, "opencv_contrib243d")
#pragma comment(lib, "opencv_legacy243d")
#pragma comment(lib, "opencv_flann243d")
#else
#pragma comment(lib, "opencv_core243")
#pragma comment(lib, "opencv_imgproc243")
#pragma comment(lib, "opencv_highgui243")
#pragma comment(lib, "opencv_ml243")
#pragma comment(lib, "opencv_video243")
#pragma comment(lib, "opencv_features2d243")
#pragma comment(lib, "opencv_calib3d243")
#pragma comment(lib, "opencv_objdetect243")
#pragma comment(lib, "opencv_contrib243")
#pragma comment(lib, "opencv_legacy243")
#pragma comment(lib, "opencv_flann243")
#endif