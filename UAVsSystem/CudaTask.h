#pragma once
#include <vector>
#include "thrust\device_ptr.h"

class CudaTask{
private:
	dim3 blockDim;
	dim3 threadDim;
public:
	CudaTask(const dim3 &blockDim, const dim3 &threadDim) :blockDim(blockDim), threadDim(threadDim){}
	virtual ~CudaTask(){}

	virtual void operator()(const thrust::device_ptr<void> &devPtr, const std::vector<const thrust::device_ptr<void>> &otherPtrs = {}) = 0;
};

