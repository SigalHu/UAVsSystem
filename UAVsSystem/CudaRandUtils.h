#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "DeviceVector.h"
#include "CudaUtils.h"

class CudaRandUtils :protected CudaUtils{
private:
	static std::string getStatusString(curandStatus_t status);

	static void createGenerator(curandGenerator_t &generator, curandRngType_t rng_type);
public:
	static void generateNormal(DeviceVector<float> &vector, float mean, float stddev);
	static void generateUniform(DeviceVector<float> &vector);
	//static bool cudaNoiseGene(float *noise_I, float *noise_Q, size_t length, float mean, float stddev);
};