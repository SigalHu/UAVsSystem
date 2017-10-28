#pragma once

class CudaUtils{
public:
	static const int THREAD_NUM_PER_BLOCK = 256;
public:
	virtual ~CudaUtils() = 0;
};