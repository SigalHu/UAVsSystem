#pragma once

#include "cuda_api.h"
#include "common.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <cstdio>
#include <stdexcept>
using namespace std;

class CudaUtils{
public:
	static const int THREAD_NUM_PER_BLOCK = 256;
protected:
	static string lastError;
public:
	virtual ~CudaUtils()=0;

	const string& getLastError() const;

	static bool setDevice(int device);
	static bool resetDevice();
	static int getDeviceCount();
	static bool getDeviceProperties(int device, cudaDeviceProp &prop);
	static void printDeviceProperties();
};