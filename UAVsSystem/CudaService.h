#pragma once

#include <string>
#include <sstream>
#include "UtilsDeclaration.h"

using namespace std;

class CudaService{
private:
	static const unsigned int MIN_PATH_NUM = 8;

	unsigned int pathNum;
	float fs;
	float timeSpend;
	float maxFd;
	float deltaOmega;
public:
	CudaService(float fs, float timeSpend);
	CudaService(unsigned int pathNum, float fs, float timeSpend, float maxFd, float deltaOmega);
	virtual ~CudaService();

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
