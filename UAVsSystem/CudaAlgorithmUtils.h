#pragma once

#include "CudaUtils.h"

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8
#define GRID_DIM_X 65535

class CudaAlgorithmUtils :public CudaUtils{
public:
	static bool cudaNoiseGeneWithSoS(float *noise_I, float *noise_Q, float fs, float time_spend,
		float power_avg = 1, unsigned int path_num = 16, float fd_max = 50, float delta_omega = 0);
};