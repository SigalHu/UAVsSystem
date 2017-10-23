#include "SystemCode.h"

const SystemCode SystemCode::CUDA_RUNTIME_ERROR(10000, "[%s::%s][Function %s] Cuda run error.\n[Error Detail] %s");
const SystemCode SystemCode::CUDA_CALL_ERROR(10001, "[%s::%s][Function %s] Cuda call error.\n[Error Detail] %s");

const SystemCode SystemCode::NULL_POINTER(10002, "[%s::%s][Variable %s] The pointer is null.\n[Error Detail] %s");
const SystemCode SystemCode::OUT_OF_RANGE(10003,"[%s::%s][Variable %s] The Variable is out of range.\n[Error Detail] %s");

SystemCode::SystemCode(const unsigned int &code, const string &info)
:code(code),info(info){
}


SystemCode::~SystemCode() {
}

const unsigned int& SystemCode::getCode() const{
	return this->code;
}

const string& SystemCode::getInfo() const{
	return this->info;
}
