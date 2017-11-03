#include "common.h"
#include "cuda_runtime.h"
#include "CudaCoreUtils.h"

std::string CudaCoreUtils::getClassName(){
	return MacroUtils_ClassName(CudaCoreUtils);
}

void CudaCoreUtils::setDevice(int device){
	cudaError_t error = cudaSetDevice(device);
	if (cudaSuccess != error)
		throw SystemException(SystemCodeEnum::CUDA_RUNTIME_ERROR,
		getClassName(), MacroUtils_CurFunctionName(),
		MacroUtils_FunctionName(cudaSetDevice), cudaGetErrorString(error));
}

void CudaCoreUtils::resetDevice(){
	cudaError_t error = cudaDeviceReset();
	if (cudaSuccess != error)
		throw SystemException(SystemCodeEnum::CUDA_RUNTIME_ERROR,
		getClassName(), MacroUtils_CurFunctionName(),
		MacroUtils_FunctionName(cudaDeviceReset), cudaGetErrorString(error));
}

int CudaCoreUtils::getDeviceCount(){
	int count = 0;
	cudaError_t error = cudaGetDeviceCount(&count);
	if (cudaSuccess != error)
		throw SystemException(SystemCodeEnum::CUDA_RUNTIME_ERROR,
		getClassName(), MacroUtils_CurFunctionName(),
		MacroUtils_FunctionName(cudaGetDeviceCount), cudaGetErrorString(error));
	return count;
}

void CudaCoreUtils::getDeviceProperties(int device, cudaDeviceProp &prop){
	cudaError_t error = cudaGetDeviceProperties(&prop, device);
	if (cudaSuccess != error)
		throw SystemException(SystemCodeEnum::CUDA_RUNTIME_ERROR,
		getClassName(), MacroUtils_CurFunctionName(),
		MacroUtils_FunctionName(cudaGetDeviceProperties), cudaGetErrorString(error));
}

void CudaCoreUtils::printDeviceProperties(){
	int count = getDeviceCount();
	cudaDeviceProp prop;
	for (int ii = 0; ii < count; ii++){
		getDeviceProperties(ii, prop);
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