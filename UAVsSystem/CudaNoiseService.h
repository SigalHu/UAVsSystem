#pragma once
#include "CudaSoSService.h"

class CudaNoiseService:public CudaSoSService{
private:
	static const float DEFAULT_NOISE_POWER;

	float noisePower;
private:
	static std::string getClassName();

	std::string getNoisePowerStr() const;
public:
	CudaNoiseService();
	CudaNoiseService(const int &deviceId);
	CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend);
	CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega);
	CudaNoiseService(const int &deviceId, const float &fs, const float &timeSpend, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega, const float &noisePower);
	~CudaNoiseService();

	void setNoisePower(const float &noisePower);
	float getNoisePower() const;
	std::string toString() const;
};
