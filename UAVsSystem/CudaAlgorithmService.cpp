#include "CudaAlgorithmService.h"

const unsigned int CudaAlgorithmService::MIN_PATH_NUM = 8;
const float CudaAlgorithmService::DEFAULT_FS = 1000000;
const float CudaAlgorithmService::DEFAULT_TIME_SPEND = 1;
const float CudaAlgorithmService::DEFAULT_MAX_FD = 50;
const float CudaAlgorithmService::DEFAULT_DELTA_OMEGA = 0;

CudaAlgorithmService::CudaAlgorithmService(const int &deviceId){
	setDeviceId(deviceId);
	setFs(DEFAULT_FS);
	setTimeSpend(DEFAULT_TIME_SPEND);
	setPathNum(MIN_PATH_NUM);
	setMaxFd(DEFAULT_MAX_FD);
	setDeltaOmega(DEFAULT_DELTA_OMEGA);
}

CudaAlgorithmService::CudaAlgorithmService(const int &deviceId, const float &fs, const float &timeSpend){
	setDeviceId(deviceId);
	setFs(fs);
	setTimeSpend(timeSpend);
	setPathNum(MIN_PATH_NUM);
	setMaxFd(DEFAULT_MAX_FD);
	setDeltaOmega(DEFAULT_DELTA_OMEGA);
}

CudaAlgorithmService::CudaAlgorithmService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega){
	setDeviceId(deviceId);
	setFs(fs);
	setTimeSpend(timeSpend);
	setPathNum(pathNum);
	setDeltaOmega(deltaOmega);
	setMaxFd(maxFd);
}

CudaAlgorithmService::~CudaAlgorithmService(){
}

void CudaAlgorithmService::setDeviceId(const int &deviceId){
	this->deviceId = deviceId;
}

int CudaAlgorithmService::getDeviceId() const{
	return this->deviceId;
}

unsigned int CudaAlgorithmService::setPathNum(const unsigned int &pathNum){
	unsigned int tmp = MIN_PATH_NUM;
	while (tmp < pathNum)
		tmp <<= 1;
	return this->pathNum = tmp;
}

unsigned int CudaAlgorithmService::getPathNum() const{
	return this->pathNum;
}

void CudaAlgorithmService::setFs(const float &fs){
	this->fs = fs;
}

void CudaAlgorithmService::setFs(const float &fs, const FrequencyUnit &unit){
	this->fs = fs * unit;
}

float CudaAlgorithmService::getFs() const{
	return this->fs;
}

void CudaAlgorithmService::setTimeSpend(const float &timeSpend){
	this->timeSpend = timeSpend;
}

float CudaAlgorithmService::getTimeSpend() const{
	return this->timeSpend;
}

void CudaAlgorithmService::setMaxFd(const float &maxFd){
	this->maxFd = maxFd;
}

void CudaAlgorithmService::setMaxFd(const float &maxFd, const FrequencyUnit &unit){
	this->maxFd = maxFd * unit;
}

float CudaAlgorithmService::getMaxFd() const{
	return this->maxFd;
}

void CudaAlgorithmService::setDeltaOmega(const float &deltaOmega){
	this->deltaOmega = deltaOmega;
}

float CudaAlgorithmService::getDeltaOmega() const{
	return this->deltaOmega;
}

string CudaAlgorithmService::toString() const{
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
	ss << "}";
	return ss.str();
}