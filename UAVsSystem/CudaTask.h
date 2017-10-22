#pragma once

#include "UtilsDeclaration.h"

class CudaTask{
private:
	dim3 threadDim;
	dim3 blockDim;
public:
	CudaTask(const dim3 &blockDim, const dim3 &threadDim) :blockDim(blockDim), threadDim(threadDim){}
	virtual ~CudaTask(){}

	virtual void operator()() = 0;
};

