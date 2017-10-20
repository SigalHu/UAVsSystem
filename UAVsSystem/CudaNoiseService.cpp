#include "CudaNoiseService.h"

CudaNoiseService::CudaNoiseService(int deviceId)
:CudaService(deviceId){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(int deviceId, float fs, float timeSpend)
:CudaService(deviceId, fs, timeSpend){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(int deviceId, unsigned int pathNum, float fs, float timeSpend, float maxFd, float deltaOmega)
: CudaService(deviceId, pathNum, fs, timeSpend, maxFd, deltaOmega){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(int deviceId, unsigned int pathNum, float fs, float timeSpend, float maxFd, float deltaOmega, float noisePower)
: CudaService(deviceId, pathNum, fs, timeSpend, maxFd, deltaOmega){
	setNoisePower(noisePower);
}

CudaNoiseService::~CudaNoiseService(){
}

void CudaNoiseService::setNoisePower(float noisePower){
	this->noisePower = noisePower;
}

float CudaNoiseService::getNoisePower(){
	return this->noisePower;
}

string CudaNoiseService::toString(){
	string str;
	ostringstream ss;
	ss << typeid(*this).name();
	ss << "{";
	ss << varName(deviceId) << "=" << getDeviceId();
	ss << ",";
	ss << varName(pathNum) << "=" << getPathNum();
	ss << ",";
	ss << varName(fs) << "=" << getFs();
	ss << ",";
	ss << varName(timeSpend) << "=" << getTimeSpend();
	ss << ",";
	ss << varName(maxFd) << "=" << getMaxFd();
	ss << ",";
	ss << varName(deltaOmega) << "=" << getDeltaOmega();
	ss << ",";
	ss << varName(noisePower) << "=" << getNoisePower();
	ss << "}";
	return ss.str();
}