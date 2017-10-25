#pragma once

#include <string>
#include <sstream>
#include "common.h"
using namespace std;

class CudaAlgorithmService{
private:
	static const unsigned int MIN_PATH_NUM;
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
	void setDeviceId(const int &deviceId);
public:
	CudaAlgorithmService(const int &deviceId);
	CudaAlgorithmService(const int &deviceId, const float &fs, const float &timeSpend);
	CudaAlgorithmService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega);
	virtual ~CudaAlgorithmService();

	int getDeviceId() const;
	unsigned int setPathNum(const unsigned int &pathNum);
	unsigned int getPathNum() const;
	void setFs(const float &fs);
	void setFs(const float &fs, const FrequencyUnit &unit);
	float getFs() const;
	void setTimeSpend(const float &timeSpend);
	float getTimeSpend() const;
	void setMaxFd(const float &maxFd);
	void setMaxFd(const float &maxFd, const FrequencyUnit &unit);
	float getMaxFd() const;
	void setDeltaOmega(const float &deltaOmega);
	float getDeltaOmega() const;

	virtual string toString() const;
};