#include "CudaTask.h"

CudaTask::CudaTask(const dim3 &blockDim, const dim3 &threadDim) 
:blockDim(blockDim), threadDim(threadDim){
}

CudaTask::~CudaTask(){
}