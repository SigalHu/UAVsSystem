#include "CudaNoiseService.h"

const float CudaNoiseService::DEFAULT_NOISE_POWER = 1;

CudaNoiseService::CudaNoiseService(const int &deviceId)
:CudaAlgorithmService(deviceId){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend)
:CudaAlgorithmService(deviceId, fs, timeSpend){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega)
: CudaAlgorithmService(deviceId, fs, timeSpend, pathNum, maxFd, deltaOmega){
	setNoisePower(DEFAULT_NOISE_POWER);
}

CudaNoiseService::CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega, const float &noisePower)
: CudaAlgorithmService(deviceId, fs, timeSpend, pathNum, maxFd, deltaOmega){
	setNoisePower(noisePower);
}

CudaNoiseService::~CudaNoiseService(){
}

void CudaNoiseService::setNoisePower(const float &noisePower){
	this->noisePower = noisePower;
}

float CudaNoiseService::getNoisePower() const{
	return this->noisePower;
}

string CudaNoiseService::toString() const{
	string str;
	ostringstream ss;
	ss << typeid(*this).name();
	ss << "{";
	ss << MacroUtils_VariableName(deviceId) << "=" << getDeviceId();
	ss << ",";
	ss << MacroUtils_VariableName(pathNum) << "=" << getPathNum();
	ss << ",";
	ss << MacroUtils_VariableName(fs) << "=" << getFs();
	ss << ",";
	ss << MacroUtils_VariableName(timeSpend) << "=" << getTimeSpend();
	ss << ",";
	ss << MacroUtils_VariableName(maxFd) << "=" << getMaxFd();
	ss << ",";
	ss << MacroUtils_VariableName(deltaOmega) << "=" << getDeltaOmega();
	ss << ",";
	ss << MacroUtils_VariableName(noisePower) << "=" << getNoisePower();
	ss << "}";
	return ss.str();
}