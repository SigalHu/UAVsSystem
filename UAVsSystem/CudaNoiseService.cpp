#include "common.h"
#include "CudaNoiseService.h"

const float CudaNoiseService::DEFAULT_NOISE_POWER = 1;

CudaNoiseService::CudaNoiseService()
:CudaSoSService(){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(const int &deviceId)
:CudaSoSService(deviceId){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend)
:CudaSoSService(deviceId, fs, timeSpend){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega)
: CudaSoSService(deviceId, fs, timeSpend, pathNum, maxFd, deltaOmega){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega, const float &noisePower)
: CudaSoSService(deviceId, fs, timeSpend, pathNum, maxFd, deltaOmega){
	setNoisePower(noisePower);
}

CudaNoiseService::~CudaNoiseService(){
}

std::string CudaNoiseService::getClassName(){
	return MacroUtils_ClassName(CudaNoiseService);
}

std::string CudaNoiseService::getNoisePowerStr() const{
	return MacroUtils_VariableName(noisePower);
}

void CudaNoiseService::setNoisePower(const float &noisePower){
	this->noisePower = noisePower;
}

float CudaNoiseService::getNoisePower() const{
	return this->noisePower;
}

std::string CudaNoiseService::toString() const{
	return StringUtils::format(getClassName().append("[").append(CudaSoSService::toString()).append(", ")
		.append(getNoisePowerStr()).append(" = %f]"),
		getNoisePower());
}