#include <time.h>
#include <string>
#include "common.h"
#include "DeviceVector.hpp"
#include "CudaRandUtils.h"

std::string CudaRandUtils::getClassName(){
	return MacroUtils_ClassName(CudaRandUtils);
}

std::string CudaRandUtils::getStatusStr(curandStatus_t status){
	switch (status){
	case CURAND_STATUS_SUCCESS:
		return "No errors.";
	case CURAND_STATUS_VERSION_MISMATCH:
		return "Header file and linked library version do not match.";
	case CURAND_STATUS_NOT_INITIALIZED:
		return "Generator not initialized.";
	case CURAND_STATUS_ALLOCATION_FAILED:
		return "Memory allocation failed.";
	case CURAND_STATUS_TYPE_ERROR:
		return "Generator is wrong type.";
	case CURAND_STATUS_OUT_OF_RANGE:
		return "Argument out of range.";
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "Length requested is not a multple of dimension.";
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "GPU does not have double precision required by MRG32k3a.";
	case CURAND_STATUS_LAUNCH_FAILURE:
		return "Kernel launch failure.";
	case CURAND_STATUS_PREEXISTING_FAILURE:
		return "Preexisting failure on library entry.";
	case CURAND_STATUS_INITIALIZATION_FAILED:
		return "Initialization of CUDA failed.";
	case CURAND_STATUS_ARCH_MISMATCH:
		return "Architecture mismatch, GPU does not support requested feature.";
	case CURAND_STATUS_INTERNAL_ERROR:
		return "Internal library error.";
	default:
		return "unrecognized error code.";
	}
}

void CudaRandUtils::createGenerator(curandGenerator_t &generator, curandRngType_t rng_type){
	curandStatus_t status = curandCreateGenerator(&generator, rng_type);
	if (CURAND_STATUS_SUCCESS != status)
		throw SystemException(SystemCodeEnum::CUDA_RUNTIME_ERROR,
		getClassName(), MacroUtils_CurFunctionName(),
		MacroUtils_FunctionName(curandCreateGenerator), getStatusStr(status));

	status = curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
	if (CURAND_STATUS_SUCCESS != status)
		throw SystemException(SystemCodeEnum::CUDA_RUNTIME_ERROR,
		getClassName(), MacroUtils_CurFunctionName(),
		MacroUtils_FunctionName(curandSetPseudoRandomGeneratorSeed), getStatusStr(status));
}

void CudaRandUtils::generateNormal(DeviceVector<float> &vector, float mean, float stddev){
	curandGenerator_t generator;
	createGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

	curandStatus_t status = curandGenerateNormal(generator, raw_pointer_cast(vector.data()), vector.size(), mean, stddev);
	if (CURAND_STATUS_SUCCESS != status)
		throw SystemException(SystemCodeEnum::CUDA_RUNTIME_ERROR,
		getClassName(), MacroUtils_CurFunctionName(),
		MacroUtils_FunctionName(curandGenerateNormal), getStatusStr(status));
}

void CudaRandUtils::generateUniform(DeviceVector<float> &vector){
	curandGenerator_t generator;
	createGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

	curandStatus_t status = curandGenerateUniform(generator, raw_pointer_cast(vector.data()), vector.size());
	if (CURAND_STATUS_SUCCESS != status)
		throw SystemException(SystemCodeEnum::CUDA_RUNTIME_ERROR,
		getClassName(), MacroUtils_CurFunctionName(),
		MacroUtils_FunctionName(curandGenerateUniform), getStatusStr(status));
}

bool CudaRandUtils::cudaNoiseGene(float *noise_I, float *noise_Q, size_t length, float mean, float stddev){
	bool isSucceed = true;
	cudaError_t cudaStatus;
	float *dev_rand = NULL;

	try{
		curandGenerator_t gen;
		curandStatus_t cuRandStatus;

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		// hu ∑÷≈‰ø’º‰
		cudaStatus = cudaMalloc((void **)&dev_rand, 2 * length*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		cuRandStatus = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		if (cuRandStatus != CURAND_STATUS_SUCCESS){
			throw false;
		}

		cuRandStatus = curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
		if (cuRandStatus != CURAND_STATUS_SUCCESS){
			throw false;
		}

		cuRandStatus = curandGenerateNormal(gen, dev_rand, 2 * length, mean, stddev);
		if (cuRandStatus != CURAND_STATUS_SUCCESS){
			throw false;
		}

		cudaStatus = cudaMemcpy((void *)noise_I, (void *)dev_rand, length*sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
		cudaStatus = cudaMemcpy((void *)noise_Q, (void *)(dev_rand + length), length*sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
	}
	catch (bool &msg){
		isSucceed = msg;
	}

	if (dev_rand)
		cudaFree(dev_rand);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		return false;
	}
	return isSucceed;
}