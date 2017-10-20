#pragma once

#include "CudaUtils.h"

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8
#define GRID_DIM_X 65535

class CudaAlgorithmUtils :protected CudaUtils{
public:
	template<size_t _Size>
	static void noiseOmegaCulc(ComplexArray<float, _Size> &devOmega, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega);

	static void noiseSoSCulc(float *dev_cos_value, float *dev_sin_value,
		unsigned int pitch_width, unsigned int width, unsigned int heigth, float delta_t,
		float *dev_omega_n_I, float *dev_omega_n_Q, float *dev_phi_n);

	static void noiseSoSSum(float *dev_cos_value, float *dev_sin_value,
		unsigned int pitch_width, unsigned int width, unsigned int heigth, float sum_amp);

	static bool cudaNoiseGeneWithSoS(float *noise_I, float *noise_Q, float fs, float time_spend,
		float power_avg = 1, unsigned int path_num = 16, float fd_max = 50, float delta_omega = 0);
};