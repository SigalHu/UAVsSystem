#include "CudaUtils.h"

string CudaUtils::lastError = "";

const string& CudaUtils::getLastError() const{
	return CudaUtils::lastError;
}

bool CudaUtils::setDevice(int device){
	cudaError error = cudaSetDevice(device);
	CudaUtils::lastError = cudaGetErrorString(error);
	return cudaSuccess == error;
}

bool CudaUtils::resetDevice(){
	cudaError error = cudaDeviceReset();
	CudaUtils::lastError = cudaGetErrorString(error);
	return cudaSuccess == error;
}

int CudaUtils::getDeviceCount(){
	int count = 0;
	cudaError error = cudaGetDeviceCount(&count);
	CudaUtils::lastError = cudaGetErrorString(error);
	return count;
}

bool CudaUtils::getDeviceProperties(int device, cudaDeviceProp &prop){
	cudaError error = cudaGetDeviceProperties(&prop, device);
	CudaUtils::lastError = cudaGetErrorString(error);
	return cudaSuccess == error;
}

void CudaUtils::printDeviceProperties(){
	int count = getDeviceCount();
	cudaDeviceProp prop;
	for (int ii = 0; ii < count; ii++){
		if (getDeviceProperties(ii, prop)){
			printf("###############################################\n");
			printf("Device Name : %s.\n", prop.name);
			printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
			printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
			printf("regsPerBlock : %d.\n", prop.regsPerBlock);
			printf("warpSize : %d.\n", prop.warpSize);
			printf("memPitch : %d.\n", prop.memPitch);
			printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
			printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
			printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
			printf("totalConstMem : %d.\n", prop.totalConstMem);
			printf("major.minor : %d.%d.\n", prop.major, prop.minor);
			printf("clockRate : %d.\n", prop.clockRate);
			printf("textureAlignment : %d.\n", prop.textureAlignment);
			printf("deviceOverlap : %d.\n", prop.deviceOverlap);
			printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
			printf("###############################################\n");
		}
	}
}