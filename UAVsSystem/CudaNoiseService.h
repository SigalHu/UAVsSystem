#pragma once

#include "CudaAlgorithmService.h"
#include <string>
#include <sstream>

class CudaNoiseService:public CudaAlgorithmService{
private:
	static const float DEFAULT_NOISE_POWER;

	float noisePower;
public:
	CudaNoiseService(const int &deviceId);
	CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend);
	CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega);
	CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega, const float &noisePower);
	~CudaNoiseService();

	void setNoisePower(const float &noisePower);
	float getNoisePower() const;
	string toString() const;
};
