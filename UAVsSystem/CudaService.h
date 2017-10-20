#pragma once

#include <string>
#include <sstream>
#include "UtilsDeclaration.h"

using namespace std;

class CudaService{
private:
	static const unsigned int MIN_PATH_NUM = 8;
	static const float DEFAULT_FS;
	static const float DEFAULT_TIME_SPEND;
	static const float DEFAULT_MAX_FD;
	static const float DEFAULT_DELTA_OMEGA;

	int deviceId;

	unsigned int pathNum;
	float fs;
	float timeSpend;
	float maxFd;
	float deltaOmega;
private:
	void setDeviceId(int deviceId);
public:
	CudaService(int deviceId);
	CudaService(int deviceId, float fs, float timeSpend);
	CudaService(int deviceId, unsigned int pathNum, float fs, float timeSpend, float maxFd, float deltaOmega);
	virtual ~CudaService();

	int getDeviceId();
	unsigned int setPathNum(unsigned int pathNum);
	unsigned int getPathNum();
	void setFs(float fs);
	void setFs(float fs, FrequencyUnit unit);
	float getFs();
	void setTimeSpend(float timeSpend);
	float getTimeSpend();
	void setMaxFd(float maxFd);
	void setMaxFd(float maxFd, FrequencyUnit unit);
	float getMaxFd();
	void setDeltaOmega(float deltaOmega);
	float getDeltaOmega();

	virtual string toString();
};

const float CudaService::DEFAULT_FS = 1000000;
const float CudaService::DEFAULT_TIME_SPEND = 1;
const float CudaService::DEFAULT_MAX_FD = 50;
const float CudaService::DEFAULT_DELTA_OMEGA = 0;