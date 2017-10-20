#pragma once

#include "CudaService.h"

class CudaNoiseService:public CudaService{
private:
	static const float DEFAULT_NOISE_POWER;

	float noisePower;
public:
	CudaNoiseService(int deviceId);
	CudaNoiseService(int deviceId, float fs, float timeSpend);
	CudaNoiseService(int deviceId, unsigned int pathNum, float fs, float timeSpend, float maxFd, float deltaOmega);
	CudaNoiseService(int deviceId, unsigned int pathNum, float fs, float timeSpend, float maxFd, float deltaOmega, float noisePower);
	~CudaNoiseService();

	void setNoisePower(float noisePower);
	float getNoisePower();
	string toString();
};

const float CudaNoiseService::DEFAULT_NOISE_POWER = 1;
