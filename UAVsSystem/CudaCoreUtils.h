#pragma once
#include "CudaUtils.h"

class CudaCoreUtils :protected CudaUtils{
private:
	static std::string getClassName();
public:
	static void setDevice(int device);
	static void resetDevice();
	static int getDeviceCount();
	static void getDeviceProperties(int device, cudaDeviceProp &prop);
	static void printDeviceProperties();
};