#pragma once

#include "CudaUtils.h"

class CudaAlgorithmUtils :protected CudaUtils{
public:
	static bool noiseOmegaCulc(float* const &devOmegaI, float* const &devOmegaQ, const dim3 &blockNum, const dim3 &threadNum,
		const unsigned int &pathNum, const float &maxFd, const float &deltaOmega);

	static bool noiseSoSCulc(float* const &devCosValue, float* const &devSinValue, const dim3 &blockNum, const dim3 &threadNum,
		const unsigned int &pitchWidth, const unsigned int &width, const unsigned int &heigth, const float &deltaT,
		float* const &devOmegaI, float* const &devOmegaQ, float* const &devPhi);

	static bool noiseSoSSum(float* const &devCosValue, float* const &devSinValue, const dim3 &blockNum, const dim3 &threadNum,
		const unsigned int &pitchWidth, const unsigned int &width, const unsigned int &heigth, const float &sumAmp);

	static bool cudaNoiseGeneWithSoS(float *noise_I, float *noise_Q, float fs, float time_spend,
		float power_avg = 1, unsigned int path_num = 16, float fd_max = 50, float delta_omega = 0);
};