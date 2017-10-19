#include "CudaCommonUtils.h"

bool CudaCommonUtils::cudaNoiseGene(float *noise_I, float *noise_Q, size_t length, float mean, float stddev){
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