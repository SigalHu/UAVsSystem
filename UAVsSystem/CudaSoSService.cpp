#include "common.h"
#include "CudaSoSService.h"

const unsigned int CudaSoSService::MIN_PATH_NUM = 8;
const unsigned int CudaSoSService::DEFAULT_PATH_NUM = 32;
const float CudaSoSService::DEFAULT_FS = 1000000;
const float CudaSoSService::DEFAULT_TIME_SPEND = 1;
const float CudaSoSService::DEFAULT_MAX_FD = 50;
const float CudaSoSService::DEFAULT_DELTA_OMEGA = 0;

CudaSoSService::CudaSoSService()
:CudaService(){
	setFs(DEFAULT_FS);
	setTimeSpend(DEFAULT_TIME_SPEND);
	setPathNum(DEFAULT_PATH_NUM);
	setMaxFd(DEFAULT_MAX_FD);
	setDeltaOmega(DEFAULT_DELTA_OMEGA);
}

CudaSoSService::CudaSoSService(const unsigned int &deviceId)
:CudaService(deviceId){
	setFs(DEFAULT_FS);
	setTimeSpend(DEFAULT_TIME_SPEND);
	setPathNum(DEFAULT_PATH_NUM);
	setMaxFd(DEFAULT_MAX_FD);
	setDeltaOmega(DEFAULT_DELTA_OMEGA);
}

CudaSoSService::CudaSoSService(const unsigned int &deviceId, const float &fs, const float &timeSpend)
:CudaService(deviceId){
	setFs(fs);
	setTimeSpend(timeSpend);
	setPathNum(DEFAULT_PATH_NUM);
	setMaxFd(DEFAULT_MAX_FD);
	setDeltaOmega(DEFAULT_DELTA_OMEGA);
}

CudaSoSService::CudaSoSService(const unsigned int &deviceId, const float &fs, const float &timeSpend, 
	const unsigned int &pathNum, const float &maxFd, const float &deltaOmega)
	:CudaService(deviceId){
	setFs(fs);
	setTimeSpend(timeSpend);
	setPathNum(pathNum);
	setDeltaOmega(deltaOmega);
	setMaxFd(maxFd);
}

CudaSoSService::~CudaSoSService(){
}

std::string CudaSoSService::getClassName(){
	return MacroUtils_ClassName(CudaSoSService);
}

unsigned int CudaSoSService::setPathNum(const unsigned int &pathNum){
	unsigned int tmp = MIN_PATH_NUM;
	while (tmp < pathNum)
		tmp <<= 1;
	return this->pathNum = tmp;
}

unsigned int CudaSoSService::getPathNum() const{
	return this->pathNum;
}

std::string CudaSoSService::getPathNumStr() const{
	return MacroUtils_VariableName(pathNum);
}

void CudaSoSService::setFs(const float &fs){
	this->fs = fs;
}

void CudaSoSService::setFs(const float &fs, const FrequencyUnit &unit){
	this->fs = fs * unit;
}

float CudaSoSService::getFs() const{
	return this->fs;
}

std::string CudaSoSService::getFsStr() const{
	return MacroUtils_VariableName(fs);
}

void CudaSoSService::setTimeSpend(const float &timeSpend){
	this->timeSpend = timeSpend;
}

float CudaSoSService::getTimeSpend() const{
	return this->timeSpend;
}

std::string CudaSoSService::getTimeSpendStr() const{
	return MacroUtils_VariableName(timeSpend);
}

void CudaSoSService::setMaxFd(const float &maxFd){
	this->maxFd = maxFd;
}

void CudaSoSService::setMaxFd(const float &maxFd, const FrequencyUnit &unit){
	this->maxFd = maxFd * unit;
}

float CudaSoSService::getMaxFd() const{
	return this->maxFd;
}

std::string CudaSoSService::getMaxFdStr() const{
	return MacroUtils_VariableName(maxFd);
}

void CudaSoSService::setDeltaOmega(const float &deltaOmega){
	this->deltaOmega = deltaOmega;
}

float CudaSoSService::getDeltaOmega() const{
	return this->deltaOmega;
}

std::string CudaSoSService::getDeltaOmegaStr() const{
	return MacroUtils_VariableName(deltaOmega);
}

std::string CudaSoSService::toString() const{
	return StringUtils::format(getClassName().append("[").append(CudaService::toString()).append(", ")
		.append(getPathNumStr()).append(" = %d, ")
		.append(getFsStr()).append(" = %f, ")
		.append(getTimeSpendStr()).append(" = %f, ")
		.append(getMaxFdStr()).append(" = %f, ")
		.append(getDeltaOmegaStr()).append(" = %f]"),
		getPathNum(), getFs(), getTimeSpend(), getMaxFd(), getDeltaOmega());
}