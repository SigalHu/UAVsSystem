#pragma once
#include <string>
#include "CudaService.h"

enum FrequencyUnit;

class CudaSoSService:public CudaService{
private:
	static const unsigned int MIN_PATH_NUM;
	static const unsigned int DEFAULT_PATH_NUM;
	static const float DEFAULT_FS;
	static const float DEFAULT_TIME_SPEND;
	static const float DEFAULT_MAX_FD;
	static const float DEFAULT_DELTA_OMEGA;

	unsigned int pathNum;
	float fs;
	float timeSpend;
	float maxFd;
	float deltaOmega;
private:
	static std::string getClassName();

	std::string getPathNumStr() const;
	std::string getFsStr() const;
	std::string getTimeSpendStr() const;
	std::string getMaxFdStr() const;
	std::string getDeltaOmegaStr() const;
public:
	CudaSoSService();
	CudaSoSService(const unsigned int &deviceId);
	CudaSoSService(const unsigned int &deviceId, const float &fs, const float &timeSpend);
	CudaSoSService(const unsigned int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega);
	~CudaSoSService();

	virtual unsigned int setPathNum(const unsigned int &pathNum);
	virtual unsigned int getPathNum() const;

	virtual void setFs(const float &fs);
	virtual void setFs(const float &fs, const FrequencyUnit &unit);
	virtual float getFs() const;

	virtual void setTimeSpend(const float &timeSpend);
	virtual float getTimeSpend() const;

	virtual void setMaxFd(const float &maxFd);
	virtual void setMaxFd(const float &maxFd, const FrequencyUnit &unit);
	virtual float getMaxFd() const;

	virtual void setDeltaOmega(const float &deltaOmega);
	virtual float getDeltaOmega() const;

	virtual std::string toString() const;
};