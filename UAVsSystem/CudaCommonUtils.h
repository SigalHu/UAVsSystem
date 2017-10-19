#pragma once

#include "CudaUtils.h"

class CudaCommonUtils :public CudaUtils{
public:
	static bool cudaNoiseGene(float *noise_I, float *noise_Q, size_t length, float mean, float stddev);
};