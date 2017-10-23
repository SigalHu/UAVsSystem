#pragma once

#include "cuda_task_api.h"
#include <vector>
using namespace std;

class CudaTask{
private:
	dim3 threadDim;
	dim3 blockDim;
public:
	CudaTask(const dim3 &blockDim, const dim3 &threadDim) :blockDim(blockDim), threadDim(threadDim){}
	virtual ~CudaTask(){}

	virtual void operator()(void* const &devPtr, const vector<void* const> &otherPtrs = {}) = 0;
};

