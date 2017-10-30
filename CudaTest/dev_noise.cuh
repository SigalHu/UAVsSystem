#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#define THREADS_EACH_BLOCK 256
#define BLOCK_DIM_X_32 32
#define BLOCK_DIM_X_64 64
#define BLOCK_DIM_X_128 128
#define BLOCK_DIM_X_256 256
#define GRID_DIM_LIMIT 65535

__global__ void noiseOmegaCulc(float *dev_omega_n_I, float *dev_omega_n_Q, unsigned int tid_max,
	float omega_amp, float delta_alpha, float delta_omega);

__global__ void noiseSoSCulc(float *dev_cos_value, float *dev_sin_value,
	unsigned int pitch_width, unsigned int width, unsigned int heigth, float delta_t,
	float *dev_omega_n_I, float *dev_omega_n_Q, float *dev_phi_n);

__global__ void noiseSoSSum(float *dev_cos_value, float *dev_sin_value,
	unsigned int pitch_width, unsigned int width, unsigned int heigth, float sum_amp);

__global__ void noiseSoSCulcBaseCol(float *dev_cos_value, float *dev_sin_value,
	unsigned int path_num, unsigned int col_num, unsigned int row_num,
	float delta_t, float *dev_omega_n_I, float *dev_omega_n_Q, float *dev_phi_n);