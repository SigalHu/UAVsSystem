#include "common.h"
#include "CudaTask.h"

CudaTask::CudaTask(const dim3 &blockDim, const dim3 &threadDim) 
:blockDim(blockDim), threadDim(threadDim){
}

CudaTask::~CudaTask(){
}

std::string CudaTask::getClassName() const{
	return MacroUtils_ClassName(CudaTask);
}

std::string  CudaTask::getBlockDim() const{
	return StringUtils::format(MacroUtils_VariableName(blockDim)
		.append("[x = %d, y = %d, z = %d]"),
		blockDim.x, blockDim.y, blockDim.z);
}

std::string  CudaTask::getThreadDim() const{
	return StringUtils::format(MacroUtils_VariableName(threadDim)
		.append("[x = %d, y = %d, z = %d]"),
		threadDim.x, threadDim.y, threadDim.z);
}

std::string CudaTask::toString() const{
	return MacroUtils_ClassName(CudaTask).append("[").append(getBlockDim()).append(", ").append(getThreadDim()).append("]");
}