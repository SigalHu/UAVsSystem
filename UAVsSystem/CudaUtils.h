#pragma once

#include "cuda_task.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <cstdio>
#include <stdexcept>
using namespace std;

#define varName(x) #x

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