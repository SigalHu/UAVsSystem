#pragma once

#include "CudaUtils.h"

class CudaCoreUtils :protected CudaUtils{
public:
	static bool malloc(void **devPtr, size_t size);
	static bool mallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
	static void free(void *devPtr);
	static bool memcpyDeviceToHost(void *dst, const void *src, size_t count);
};