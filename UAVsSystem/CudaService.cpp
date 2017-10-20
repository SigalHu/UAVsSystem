#include "CudaService.h"

CudaService::CudaService(int deviceId){
	setDeviceId(deviceId);
	setFs(DEFAULT_FS);
	setTimeSpend(DEFAULT_TIME_SPEND);
	setPathNum(MIN_PATH_NUM);
	setMaxFd(DEFAULT_MAX_FD);
	setDeltaOmega(DEFAULT_DELTA_OMEGA);
}

CudaService::CudaService(int deviceId, float fs, float timeSpend){
	setDeviceId(deviceId);
	setFs(fs);
	setTimeSpend(timeSpend);
	setPathNum(MIN_PATH_NUM);
	setMaxFd(DEFAULT_MAX_FD);
	setDeltaOmega(DEFAULT_DELTA_OMEGA);
}

CudaService::CudaService(int deviceId, unsigned int pathNum, float fs, float timeSpend, float maxFd, float deltaOmega){
	setDeviceId(deviceId);
	setFs(fs);
	setTimeSpend(timeSpend);
	setPathNum(pathNum);
	setDeltaOmega(deltaOmega);
	setMaxFd(maxFd);
}

CudaService::~CudaService(){
}

void CudaService::setDeviceId(int deviceId){
	this->deviceId = deviceId;
}

int CudaService::getDeviceId(){
	return this->deviceId;
}

unsigned int CudaService::setPathNum(unsigned int pathNum){
	unsigned int tmp = MIN_PATH_NUM;
	while (tmp < pathNum)
		tmp <<= 1;
	return this->pathNum = tmp;
}

unsigned int CudaService::getPathNum(){
	return this->pathNum;
}

void CudaService::setFs(float fs){
	this->fs = fs;
}

void CudaService::setFs(float fs, FrequencyUnit unit){
	this->fs = fs * unit;
}

float CudaService::getFs(){
	return this->fs;
}

void CudaService::setTimeSpend(float timeSpend){
	this->timeSpend = timeSpend;
}

float CudaService::getTimeSpend(){
	return this->timeSpend;
}

void CudaService::setMaxFd(float maxFd){
	this->maxFd = maxFd;
}

void CudaService::setMaxFd(float maxFd, FrequencyUnit unit){
	this->maxFd = maxFd * unit;
}

float CudaService::getMaxFd(){
	return this->maxFd;
}

void CudaService::setDeltaOmega(float deltaOmega){
	this->deltaOmega = deltaOmega;
}

float CudaService::getDeltaOmega(){
	return this->deltaOmega;
}

string CudaService::toString(){
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
	ss << "}";
	return ss.str();
}