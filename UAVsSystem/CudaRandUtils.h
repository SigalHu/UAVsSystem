#pragma once
#include <string>
#include "curand.h"
#include "CudaUtils.h"

template<class _T, class _Alloc>
class DeviceVector;

class CudaRandUtils :protected CudaUtils{
private:
	static std::string getClassName();
	static std::string getStatusStr(curandStatus_t status);

	static void createGenerator(curandGenerator_t &generator, curandRngType_t rng_type);
public:
	template<class _Alloc>
	static void generateNormal(DeviceVector<float, _Alloc> &vector, float mean, float stddev);

	template<class _Alloc>
	static void generateUniform(DeviceVector<float, _Alloc> &vector);

	static bool cudaNoiseGene(float *noise_I, float *noise_Q, size_t length, float mean, float stddev);
};