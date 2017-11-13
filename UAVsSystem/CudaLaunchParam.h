#pragma once
#include "device_launch_parameters.h"
#include <string>

class CudaLaunchParam{
private:
	dim3 threadNum;
	dim3 blockNum;
	size_t sharedMemSize;
private:
	static std::string getClassName();

	std::string formatThreadNum() const;
	std::string formatBlockNum() const;
	std::string formatSharedMemSize() const;
public:
	CudaLaunchParam();
	CudaLaunchParam(const dim3 &threadNum, const dim3 &blockNum);
	CudaLaunchParam(const dim3 &threadNum, const dim3 &blockNum, const size_t &sharedMemSize);
	~CudaLaunchParam();

	void setThreadNum(const dim3 &threadNum);
	dim3 getThreadNum() const;
	void setBlockNum(const dim3 &blockNum);
	dim3 getBlockNum() const;
	void setSharedMemSize(const size_t &sharedMemSize);
	size_t getSharedMemSize() const;

	std::string toString() const;
};

