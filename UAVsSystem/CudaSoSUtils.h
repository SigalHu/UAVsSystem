#pragma once
#include "CudaUtils.h"
#include "DeviceVector.h"
#include "CudaLaunchParam.h"

class CudaSoSUtils :protected CudaUtils{
private:
	static const int THREAD_NUM_PER_BLOCK = 256;
private:
	static std::string getClassName();
public:
	static void noiseGene(CudaLaunchParam cudaLaunchParam, DeviceVector<float> &noiseI, DeviceVector<float> &noiseQ, const float &fs = 1000000,
		const float &avgPower = 1, const unsigned int &pathNum = 32, const float &maxFd = 50, const float &deltaOmega = 0);

	//template<class _Alloc>
	//static void noiseOmegaCulc(const dim3 &blockNum, const dim3 &threadNum,
	//	DeviceVector<float, _Alloc> &devOmegaI, DeviceVector<float, _Alloc> &devOmegaQ,
	//	const float &maxFd, const float &deltaOmega);

	//static bool noiseSoSCulc(float* const &devCosValue, float* const &devSinValue, const dim3 &blockNum, const dim3 &threadNum,
	//	const unsigned int &pitchWidth, const unsigned int &width, const unsigned int &heigth, const float &deltaT,
	//	float* const &devOmegaI, float* const &devOmegaQ, float* const &devPhi);

	//static bool noiseSoSSum(float* const &devCosValue, float* const &devSinValue, const dim3 &blockNum, const dim3 &threadNum,
	//	const unsigned int &pitchWidth, const unsigned int &width, const unsigned int &heigth, const float &sumAmp);

	//static bool cudaNoiseGeneWithSoS(float *noise_I, float *noise_Q, float fs, float time_spend,
	//	float power_avg = 1, unsigned int path_num = 16, float fd_max = 50, float delta_omega = 0);
};