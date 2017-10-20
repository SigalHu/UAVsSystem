#include "CudaService.h"

const unsigned int CudaService::MIN_PATH_NUM = 8;
const float CudaService::DEFAULT_FS = 1000000;
const float CudaService::DEFAULT_TIME_SPEND = 1;
const float CudaService::DEFAULT_MAX_FD = 50;
const float CudaService::DEFAULT_DELTA_OMEGA = 0;

CudaService::CudaService(const int &deviceId){
	setDeviceId(deviceId);
	setFs(DEFAULT_FS);
	setTimeSpend(DEFAULT_TIME_SPEND);
	setPathNum(MIN_PATH_NUM);
	setMaxFd(DEFAULT_MAX_FD);
	setDeltaOmega(DEFAULT_DELTA_OMEGA);
}

CudaService::CudaService(const int &deviceId, const float &fs, const float &timeSpend){
	setDeviceId(deviceId);
	setFs(fs);
	setTimeSpend(timeSpend);
	setPathNum(MIN_PATH_NUM);
	setMaxFd(DEFAULT_MAX_FD);
	setDeltaOmega(DEFAULT_DELTA_OMEGA);
}

CudaService::CudaService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega){
	setDeviceId(deviceId);
	setFs(fs);
	setTimeSpend(timeSpend);
	setPathNum(pathNum);
	setDeltaOmega(deltaOmega);
	setMaxFd(maxFd);
}

CudaService::~CudaService(){
}

void CudaService::setDeviceId(const int &deviceId){
	this->deviceId = deviceId;
}

int CudaService::getDeviceId() const{
	return this->deviceId;
}

unsigned int CudaService::setPathNum(const unsigned int &pathNum){
	unsigned int tmp = MIN_PATH_NUM;
	while (tmp < pathNum)
		tmp <<= 1;
	return this->pathNum = tmp;
}

unsigned int CudaService::getPathNum() const{
	return this->pathNum;
}

void CudaService::setFs(const float &fs){
	this->fs = fs;
}

void CudaService::setFs(const float &fs, const FrequencyUnit &unit){
	this->fs = fs * unit;
}

float CudaService::getFs() const{
	return this->fs;
}

void CudaService::setTimeSpend(const float &timeSpend){
	this->timeSpend = timeSpend;
}

float CudaService::getTimeSpend() const{
	return this->timeSpend;
}

void CudaService::setMaxFd(const float &maxFd){
	this->maxFd = maxFd;
}

void CudaService::setMaxFd(const float &maxFd, const FrequencyUnit &unit){
	this->maxFd = maxFd * unit;
}

float CudaService::getMaxFd() const{
	return this->maxFd;
}

void CudaService::setDeltaOmega(const float &deltaOmega){
	this->deltaOmega = deltaOmega;
}

float CudaService::getDeltaOmega() const{
	return this->deltaOmega;
}

string CudaService::toString() const{
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