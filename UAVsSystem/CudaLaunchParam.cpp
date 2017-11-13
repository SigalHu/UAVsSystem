#include "common.h"
#include "CudaLaunchParam.h"

std::string CudaLaunchParam::getClassName(){
	return MacroUtils_ClassName(CudaLaunchParam);
}

CudaLaunchParam::CudaLaunchParam(){
}

CudaLaunchParam::CudaLaunchParam(const dim3 &threadNum, const dim3 &gridNum){
	setThreadNum(threadNum);
	setBlockNum(gridNum);
}

CudaLaunchParam::CudaLaunchParam(const dim3 &threadNum, const dim3 &gridNum, const size_t &sharedMemSize){
	setThreadNum(threadNum);
	setBlockNum(gridNum);
	setSharedMemSize(sharedMemSize);
}

CudaLaunchParam::~CudaLaunchParam(){
}

void CudaLaunchParam::setThreadNum(const dim3 &threadNum){
	this->threadNum = threadNum;
}

dim3 CudaLaunchParam::getThreadNum() const{
	return this->threadNum;
}

void CudaLaunchParam::setBlockNum(const dim3 &gridNum){
	this->blockNum = gridNum;
}

dim3 CudaLaunchParam::getBlockNum() const{
	return this->blockNum;
}

void CudaLaunchParam::setSharedMemSize(const size_t &sharedMemSize){
	this->sharedMemSize = sharedMemSize;
}

size_t CudaLaunchParam::getSharedMemSize() const{
	return this->sharedMemSize;
}

std::string CudaLaunchParam::formatThreadNum() const{
	return StringUtils::format(MacroUtils_VariableName(threadNum)
		.append("[x = %d, y = %d, z = %d]"), 
		threadNum.x, threadNum.y, threadNum.z);
}

std::string CudaLaunchParam::formatBlockNum() const{
	return StringUtils::format(MacroUtils_VariableName(blockNum)
		.append("[x = %d, y = %d, z = %d]"),
		blockNum.x, blockNum.y, blockNum.z);
}

std::string CudaLaunchParam::formatSharedMemSize() const{
	return StringUtils::format(MacroUtils_VariableName(sharedMemSize)
		.append(" = %d"), sharedMemSize);
}

std::string CudaLaunchParam::toString() const{
	return getClassName().append("[")
		.append(formatThreadNum()).append(", ")
		.append(formatBlockNum()).append(", ")
		.append(formatSharedMemSize()).append("]");
}