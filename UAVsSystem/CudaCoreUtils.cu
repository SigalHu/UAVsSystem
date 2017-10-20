#include "CudaCoreUtils.h"

bool CudaCoreUtils::malloc(void **devPtr, size_t size){
	return cudaSuccess == cudaMalloc(devPtr, size);
}

bool CudaCoreUtils::mallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height){
	return cudaSuccess == cudaMallocPitch(devPtr, pitch, width, height);
}

void CudaCoreUtils::free(void *devPtr){
	if (devPtr)
		cudaFree(devPtr);
}

bool CudaCoreUtils::memcpyDeviceToHost(void *dst, const void *src, size_t count){
	return cudaSuccess == cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}