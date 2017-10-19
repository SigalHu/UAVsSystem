#pragma once

#include "CudaUtils.h"

class CudaCommonUtils :public CudaUtils{
private:
	static bool randCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type);
public:
	static bool randGenerateNormal(float *outputPtr, size_t n, float mean, float stddev);
	static bool randGenerateUniform(float *outputPtr, size_t num);
	static bool cudaNoiseGene(float *noise_I, float *noise_Q, size_t length, float mean, float stddev);
};