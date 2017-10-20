#pragma once

#include "CudaService.h"

class CudaNoiseService:public CudaService{
private:
	float noisePower;
public:
	CudaNoiseService(float fs, float timeSpend);
	CudaNoiseService(unsigned int pathNum, float fs, float timeSpend, float maxFd, float deltaOmega);
	CudaNoiseService(unsigned int pathNum, float fs, float timeSpend, float maxFd, float deltaOmega, float noisePower);
	~CudaNoiseService();

	void setNoisePower(float noisePower);
	float getNoisePower();
	string toString();
};

