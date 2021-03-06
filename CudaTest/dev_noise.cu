﻿#include "dev_noise.cuh"

__global__ void noiseOmegaCulc(float *dev_omega_n_I, float *dev_omega_n_Q, unsigned int length,
	float omega_amp, float delta_alpha, float delta_omega){
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < length){
		dev_omega_n_I[tid] = omega_amp*cosf(delta_alpha*tid) + delta_omega;
		dev_omega_n_Q[tid] = omega_amp*sinf(delta_alpha*tid) + delta_omega;
	}
}

__global__ void noiseSoSCulc(float *dev_cos_value, float *dev_sin_value, 
	unsigned int pitch_width, unsigned int width, unsigned int heigth, float delta_t,
	float *dev_omega_n_I, float *dev_omega_n_Q, float *dev_phi_n){
	unsigned int x = threadIdx.x,
		y = threadIdx.y,
		tidy = blockIdx.y * blockDim.y + y;
	__shared__ float sha_cos_value[THREADS_EACH_BLOCK / BLOCK_DIM_X_32][BLOCK_DIM_X_32],
		sha_sin_value[THREADS_EACH_BLOCK / BLOCK_DIM_X_32][BLOCK_DIM_X_32];
	sha_cos_value[y][x] = 0;
	sha_sin_value[y][x] = 0;
	__syncthreads();

	if (tidy < heigth){
		__shared__ float sha_omega_n_I[THREADS_EACH_BLOCK / BLOCK_DIM_X_32], 
			sha_omega_n_Q[THREADS_EACH_BLOCK / BLOCK_DIM_X_32],
			sha_phi_n_I[THREADS_EACH_BLOCK / BLOCK_DIM_X_32], 
			sha_phi_n_Q[THREADS_EACH_BLOCK / BLOCK_DIM_X_32];

		if (x == 0){
			sha_omega_n_I[y] = dev_omega_n_I[tidy];
			sha_omega_n_Q[y] = dev_omega_n_Q[tidy];
			sha_phi_n_I[y] = dev_phi_n[tidy];
			sha_phi_n_Q[y] = dev_phi_n[heigth + tidy];
		}
		__syncthreads();

		for (unsigned int tidx = blockIdx.x * blockDim.x + x;
			tidx < width; tidx += gridDim.x*blockDim.x){
			sha_cos_value[y][x] = cosf(sha_omega_n_I[y] * delta_t*tidx + 2 * CR_CUDART_PI*sha_phi_n_I[y]);
			sha_sin_value[y][x] = sinf(sha_omega_n_Q[y] * delta_t*tidx + 2 * CR_CUDART_PI*sha_phi_n_Q[y]);
			__syncthreads();

			for (unsigned int heigth_ii = blockDim.y / 2, extra = blockDim.y % 2;
				heigth_ii > 0; extra = heigth_ii % 2, heigth_ii /= 2){
				if (y < heigth_ii){
					sha_cos_value[y][x] += sha_cos_value[heigth_ii + extra + y][x];
					sha_sin_value[y][x] += sha_sin_value[heigth_ii + extra + y][x];
				}
				heigth_ii += extra;
				__syncthreads();
			}
			if (y == 0){
				unsigned int loc = blockIdx.y*pitch_width + tidx;
				dev_cos_value[loc] = sha_cos_value[0][x];
				dev_sin_value[loc] = sha_sin_value[0][x];
			}
		}
	}
}

__global__ void noiseSoSSum(float *dev_cos_value, float *dev_sin_value,
	unsigned int pitch_width, unsigned int width, unsigned int heigth, float sum_amp){
	unsigned int loc;
	float reg_cos_value, reg_sin_value;

	for (unsigned int tidx = blockIdx.x*blockDim.x + threadIdx.x;
		tidx < width; tidx += gridDim.x*blockDim.x){
		reg_cos_value = 0;
		reg_sin_value = 0;
		for (unsigned int heigth_ii = 0; heigth_ii < heigth; heigth_ii++){
			loc = heigth_ii*pitch_width + tidx;
			reg_cos_value += dev_cos_value[loc];
			reg_sin_value += dev_sin_value[loc];
		}

		dev_cos_value[tidx] = sum_amp * reg_cos_value;
		dev_sin_value[tidx] = sum_amp * reg_sin_value;
	}
}

__global__ void noiseSoSCulcBaseCol(float *dev_cos_value, float *dev_sin_value,
	unsigned int path_num, unsigned int col_num, unsigned int row_num,
	float delta_t, float *dev_omega_n_I, float *dev_omega_n_Q, float *dev_phi_n){
	unsigned int tidx = threadIdx.x,
		tidy = threadIdx.y,
		bidy = blockIdx.y;


	//unsigned int x = threadIdx.x,
	//	y = threadIdx.y,
	//	tidy = blockIdx.y * blockDim.y + y;
	//__shared__ float sha_cos_value[THREADS_EACH_BLOCK / BLOCK_DIM_X_32][BLOCK_DIM_X_32],
	//	sha_sin_value[THREADS_EACH_BLOCK / BLOCK_DIM_X_32][BLOCK_DIM_X_32];
	//sha_cos_value[y][x] = 0;
	//sha_sin_value[y][x] = 0;
	//__syncthreads();

	//if (tidy < heigth){
	//	__shared__ float sha_omega_n_I[THREADS_EACH_BLOCK / BLOCK_DIM_X_32],
	//		sha_omega_n_Q[THREADS_EACH_BLOCK / BLOCK_DIM_X_32],
	//		sha_phi_n_I[THREADS_EACH_BLOCK / BLOCK_DIM_X_32],
	//		sha_phi_n_Q[THREADS_EACH_BLOCK / BLOCK_DIM_X_32];

	//	if (x == 0){
	//		sha_omega_n_I[y] = dev_omega_n_I[tidy];
	//		sha_omega_n_Q[y] = dev_omega_n_Q[tidy];
	//		sha_phi_n_I[y] = dev_phi_n[tidy];
	//		sha_phi_n_Q[y] = dev_phi_n[heigth + tidy];
	//	}
	//	__syncthreads();

	//	for (unsigned int tidx = blockIdx.x * blockDim.x + x;
	//		tidx < width; tidx += gridDim.x*blockDim.x){
	//		sha_cos_value[y][x] = cosf(sha_omega_n_I[y] * delta_t*tidx + 2 * CR_CUDART_PI*sha_phi_n_I[y]);
	//		sha_sin_value[y][x] = sinf(sha_omega_n_Q[y] * delta_t*tidx + 2 * CR_CUDART_PI*sha_phi_n_Q[y]);
	//		__syncthreads();

	//		for (unsigned int heigth_ii = blockDim.y / 2, extra = blockDim.y % 2;
	//			heigth_ii > 0; extra = heigth_ii % 2, heigth_ii /= 2){
	//			if (y < heigth_ii){
	//				sha_cos_value[y][x] += sha_cos_value[heigth_ii + extra + y][x];
	//				sha_sin_value[y][x] += sha_sin_value[heigth_ii + extra + y][x];
	//			}
	//			heigth_ii += extra;
	//			__syncthreads();
	//		}
	//		if (y == 0){
	//			unsigned int loc = blockIdx.y*pitch_width + tidx;
	//			dev_cos_value[loc] = sha_cos_value[0][x];
	//			dev_sin_value[loc] = sha_sin_value[0][x];
	//		}
	//	}
	//}
}