#pragma once
#include <string>
#include <vector>
#include "thrust\device_ptr.h"

class CudaTask{
private:
	dim3 blockDim;
	dim3 threadDim;
private:
	std::string getClassName() const;

	std::string  getBlockDim() const;
	std::string  getThreadDim() const;
public:
	CudaTask(const dim3 &blockDim, const dim3 &threadDim);
	virtual ~CudaTask();
	std::string toString() const;

	virtual void operator()(const thrust::device_ptr<void> &devPtr, const std::vector<const thrust::device_ptr<void>> &otherPtrs = {}) = 0;
};

