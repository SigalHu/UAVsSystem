#include "common.h"
#include "CudaService.h"

const unsigned int CudaService::DEFAULT_DEVICE_ID = 0;

CudaService::CudaService():deviceId(DEFAULT_DEVICE_ID){
}

CudaService::CudaService(const unsigned int &deviceId) : deviceId(deviceId){
}

CudaService::~CudaService(){
}

std::string CudaService::getClassName(){
	return MacroUtils_ClassName(CudaService);
}

unsigned int CudaService::getDeviceId() const{
	return this->deviceId;
}

std::string CudaService::getDeviceIdStr() const{
	return MacroUtils_VariableName(deviceId);
}

std::string CudaService::toString() const{
	return StringUtils::format(getClassName().append("[").append(getDeviceIdStr()).append(" = %d]"),getDeviceId());
}