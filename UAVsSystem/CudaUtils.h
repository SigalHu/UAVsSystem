#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "dev_noise.cuh"
#include "ComplexArray.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <cstdio>
using namespace std;

class CudaUtils{
public:
	static const int THREAD_NUM_PER_BLOCK = 256;
public:
	virtual ~CudaUtils()=0;

	static bool setDevice(int device);
	static bool resetDevice();
	static int getDeviceCount();
	static bool getDeviceProperties(int device, cudaDeviceProp *prop);
	static void printDeviceProperties();
};