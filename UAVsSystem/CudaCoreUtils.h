#pragma once
#include "CudaUtils.h"

class CudaCoreUtils :protected CudaUtils{
public:
	static void setDevice(int device);
	static void resetDevice();
	static int getDeviceCount();
	static void getDeviceProperties(int device, cudaDeviceProp &prop);
	static void printDeviceProperties();
};