#pragma once

#include "cuda_api.h"
#include <vector>
using namespace std;

class CudaTask{
private:
	dim3 blockDim;
	dim3 threadDim;
public:
	CudaTask(const dim3 &blockDim, const dim3 &threadDim) :blockDim(blockDim), threadDim(threadDim){}
	virtual ~CudaTask(){}

	virtual void operator()(void* const &devPtr, const vector<void* const> &otherPtrs = {}) = 0;
};

