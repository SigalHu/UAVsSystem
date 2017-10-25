#include "SystemCodeEnum.h"

const SystemCode SystemCodeEnum::CUDA_RUNTIME_ERROR(10000, "[%s::%s][Function %s] Cuda run error.\n[Error Detail] %s");
const SystemCode SystemCodeEnum::CUDA_CALL_ERROR(10001, "[%s::%s][Function %s] Cuda call error.\n[Error Detail] %s");

const SystemCode SystemCodeEnum::NULL_POINTER(10002, "[%s::%s][Variable %s] The pointer is null.\n[Error Detail] %s");
const SystemCode SystemCodeEnum::OUT_OF_RANGE(10003, "[%s::%s][Variable %s] The variable is out of range.\n[Error Detail] %s");
