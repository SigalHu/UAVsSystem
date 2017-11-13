#pragma once
#include <map>
#include "CudaUtils.h"

class CudaCoreUtils :protected CudaUtils{
private:
	static const dim3 DEFAULT_GRID_DIM;

	static std::map<unsigned int, const dim3> gridDimMap;
private:
	static std::string getClassName();
	static std::map<unsigned int, const dim3> getGridDimList();
public:
	static const dim3 getGridDim(const unsigned int &deviceId);

	static void setDevice(int device);
	static void resetDevice();
	static int getDeviceCount();
	static void getDeviceProperties(int device, cudaDeviceProp &prop);
	static void printDeviceProperties();
};