#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "dev_noise.cuh"
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

class CudaUtils{
public:
	virtual ~CudaUtils()=0;
};