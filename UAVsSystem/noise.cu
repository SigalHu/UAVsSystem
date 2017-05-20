
#include "dev_noise.cuh"

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

bool cudaNoiseGene(float *noise_I, float *noise_Q, size_t length, float mean, float stddev){
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

		// hu 分配空间
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
	catch(bool &msg){
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

bool cudaNoiseGeneWithSoS(float *noise_I, float *noise_Q,float fs, float time_spend, 
	float power_avg, unsigned int path_num, float fd_max, float delta_omega){
	bool isSucceed = true;
	cudaError_t cudaStatus;
	float *dev_omega_n_I = NULL, 
		*dev_omega_n_Q = NULL, 
		*dev_phi_n = NULL,
		*dev_cos_value = NULL,
		*dev_sin_value = NULL;

	try{
		curandStatus_t cuRandStatus;
		curandGenerator_t gen;
		size_t blockNum, threadNum;
		dim3 blockNum2D, threadNum2D;
		size_t col_num = fs*time_spend;

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		threadNum = path_num <= THREADS_EACH_BLOCK ? path_num : THREADS_EACH_BLOCK;
		blockNum = path_num % threadNum ?
			path_num / threadNum + 1 :
			path_num / threadNum;

		cudaStatus = cudaMalloc((void **)&dev_omega_n_I, path_num*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
		cudaStatus = cudaMalloc((void **)&dev_omega_n_Q, path_num*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		noiseOmegaCulc<<<blockNum, threadNum>>>(dev_omega_n_I, dev_omega_n_Q, path_num,
			2 * M_PI*fd_max, (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1), delta_omega);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		cudaStatus = cudaMalloc((void **)&dev_phi_n, 2*path_num*sizeof(float));
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
		cuRandStatus = curandGenerateUniform(gen, dev_phi_n, 2 * path_num);
		if (cuRandStatus != CURAND_STATUS_SUCCESS){
			throw false;
		}

		threadNum2D.x = BLOCK_DIM_X_32;
		threadNum2D.y = THREADS_EACH_BLOCK / BLOCK_DIM_X_32;
		blockNum2D.x = col_num % threadNum2D.x ? 
			col_num / threadNum2D.x + 1 : 
			col_num / threadNum2D.x;
		if (blockNum2D.x > GRID_DIM_LIMIT){
			unsigned int gridNum = blockNum2D.x % GRID_DIM_LIMIT ?
				blockNum2D.x / GRID_DIM_LIMIT + 1 :
				blockNum2D.x / GRID_DIM_LIMIT;
			blockNum2D.x = blockNum2D.x % gridNum ?
				blockNum2D.x / gridNum + 1 :
				blockNum2D.x / gridNum;
		}
		blockNum2D.y = path_num % threadNum2D.y ?
			path_num / threadNum2D.y + 1 :
			path_num / threadNum2D.y;

		size_t pitch;
		cudaStatus = cudaMallocPitch((void **)&dev_cos_value, &pitch, col_num*sizeof(float), blockNum2D.y);
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
		cudaStatus = cudaMallocPitch((void **)&dev_sin_value, &pitch, col_num*sizeof(float), blockNum2D.y);
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
		noiseSoSCulc<<<blockNum2D, threadNum2D>>>(dev_cos_value, dev_sin_value, pitch / sizeof(float), 
			col_num, path_num, 1 / fs, dev_omega_n_I, dev_omega_n_Q, dev_phi_n);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		threadNum = col_num <= THREADS_EACH_BLOCK ? col_num : THREADS_EACH_BLOCK;
		blockNum = col_num % threadNum ?
			col_num / threadNum + 1 :
			col_num / threadNum;
		if (blockNum > GRID_DIM_LIMIT){
			unsigned int gridNum = blockNum % GRID_DIM_LIMIT ?
				blockNum / GRID_DIM_LIMIT + 1 :
				blockNum / GRID_DIM_LIMIT;
			blockNum = blockNum % gridNum ?
				blockNum / gridNum + 1 :
				blockNum / gridNum;
		}
		noiseSoSSum<<<blockNum, threadNum >>>(dev_cos_value, dev_sin_value, pitch / sizeof(float),
			col_num, blockNum2D.y, sqrtf(power_avg / path_num));
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		cudaStatus = cudaMemcpy((void *)noise_I, (void *)dev_cos_value, col_num*sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
		cudaStatus = cudaMemcpy((void *)noise_Q, (void *)dev_sin_value, col_num*sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
	}
	catch (bool &msg){
		isSucceed = msg;
	}

	if (dev_omega_n_I)
		cudaFree(dev_omega_n_I);
	if (dev_omega_n_Q)
		cudaFree(dev_omega_n_Q);
	if (dev_phi_n)
		cudaFree(dev_phi_n);
	if (dev_cos_value)
		cudaFree(dev_cos_value);
	if (dev_sin_value)
		cudaFree(dev_sin_value);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
	    return false;
	}
	return isSucceed;
}

bool cudaNoiseGeneWithSoSBaseCol(float *noise_I, float *noise_Q, float fs, float time_spend,
	float power_avg, unsigned int path_num, float fd_max, float delta_omega){
	bool isSucceed = true;
	cudaError_t cudaStatus;
	float *dev_omega_n_I = NULL,
		*dev_omega_n_Q = NULL,
		*dev_phi_n = NULL,
		*dev_cos_value = NULL,
		*dev_sin_value = NULL;

	try{
		curandStatus_t cuRandStatus;
		curandGenerator_t gen;
		size_t blockNum, threadNum;
		dim3 blockNum2D, threadNum2D;
		size_t time_num = fs*time_spend;
		unsigned int time_num_eachrow, time_num_eachblock;

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		threadNum = path_num <= THREADS_EACH_BLOCK ? path_num : THREADS_EACH_BLOCK;
		blockNum = path_num % threadNum ?
			path_num / threadNum + 1 :
			path_num / threadNum;

		cudaStatus = cudaMalloc((void **)&dev_omega_n_I, path_num*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
		cudaStatus = cudaMalloc((void **)&dev_omega_n_Q, path_num*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		noiseOmegaCulc<<<blockNum, threadNum>>>(dev_omega_n_I, dev_omega_n_Q, path_num,
			2 * M_PI*fd_max, (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1), delta_omega);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			throw false;
		}

		cudaStatus = cudaMalloc((void **)&dev_phi_n, 2 * path_num*sizeof(float));
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
		cuRandStatus = curandGenerateUniform(gen, dev_phi_n, 2 * path_num);
		if (cuRandStatus != CURAND_STATUS_SUCCESS){
			throw false;
		}

		if (path_num < BLOCK_DIM_X_32){
			time_num_eachrow = BLOCK_DIM_X_32 / path_num;
			threadNum2D.x = BLOCK_DIM_X_32;
		}
		else{
			time_num_eachrow = 1;
			if (path_num < BLOCK_DIM_X_64){
				threadNum2D.x = BLOCK_DIM_X_32;
			}
			else if (path_num < BLOCK_DIM_X_128){
				threadNum2D.x = BLOCK_DIM_X_64;
			}
			else if (path_num < BLOCK_DIM_X_256){
				threadNum2D.x = BLOCK_DIM_X_128;
			}
			else{
				threadNum2D.x = BLOCK_DIM_X_256;
			}
		}
		threadNum2D.y = THREADS_EACH_BLOCK / threadNum2D.x;
		time_num_eachblock = time_num_eachrow*threadNum2D.y;
		blockNum2D.x = 1;
		blockNum2D.y = time_num%time_num_eachblock ? 
			time_num / time_num_eachblock + 1 : 
			time_num / time_num_eachblock;
		if (blockNum2D.y > GRID_DIM_LIMIT){
			unsigned int gridNum = blockNum2D.y % GRID_DIM_LIMIT ?
				blockNum2D.y / GRID_DIM_LIMIT + 1 :
				blockNum2D.y / GRID_DIM_LIMIT;
			blockNum2D.y = blockNum2D.y % gridNum ?
				blockNum2D.y / gridNum + 1 :
				blockNum2D.y / gridNum;
		}

		cudaStatus = cudaMalloc((void **)&dev_cos_value, time_num*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
		cudaStatus = cudaMalloc((void **)&dev_sin_value, time_num*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			throw false;
		}
		//noiseSoSCulc<<<blockNum2D, threadNum2D>>>(dev_cos_value, dev_sin_value, pitch / sizeof(float),
		//	col_num, path_num, 1 / fs, dev_omega_n_I, dev_omega_n_Q, dev_phi_n);
		//cudaStatus = cudaGetLastError();
		//if (cudaStatus != cudaSuccess) {
		//	throw false;
		//}

		//threadNum = col_num <= threadNumLimit ? col_num : threadNumLimit;
		//blockNum = col_num % threadNum ?
		//	col_num / threadNum + 1 :
		//	col_num / threadNum;
		//if (blockNum > GRID_DIM_LIMIT){
		//	unsigned int gridNum = blockNum % GRID_DIM_LIMIT ?
		//		blockNum / GRID_DIM_LIMIT + 1 :
		//		blockNum / GRID_DIM_LIMIT;
		//	blockNum = blockNum % gridNum ?
		//		blockNum / gridNum + 1 :
		//		blockNum / gridNum;
		//}
		//noiseSoSSum << <blockNum, threadNum >> >(dev_cos_value, dev_sin_value, pitch / sizeof(float),
		//	col_num, blockNum2D.y, sqrtf(power_avg / path_num));
		//cudaStatus = cudaGetLastError();
		//if (cudaStatus != cudaSuccess) {
		//	throw false;
		//}

		//cudaStatus = cudaMemcpy((void *)noise_I, (void *)dev_cos_value, col_num*sizeof(float), cudaMemcpyDeviceToHost);
		//if (cudaStatus != cudaSuccess) {
		//	throw false;
		//}
		//cudaStatus = cudaMemcpy((void *)noise_Q, (void *)dev_sin_value, col_num*sizeof(float), cudaMemcpyDeviceToHost);
		//if (cudaStatus != cudaSuccess) {
		//	throw false;
		//}
	}
	catch (bool &msg){
		isSucceed = msg;
	}

	if (dev_omega_n_I)
		cudaFree(dev_omega_n_I);
	if (dev_omega_n_Q)
		cudaFree(dev_omega_n_Q);
	if (dev_phi_n)
		cudaFree(dev_phi_n);
	if (dev_cos_value)
		cudaFree(dev_cos_value);
	if (dev_sin_value)
		cudaFree(dev_sin_value);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		return false;
	}
	return isSucceed;
}