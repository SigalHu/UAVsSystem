#pragma once
#include "cuda_runtime.h"

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8
#define GRID_DIM_X 65535

__global__ void cudaNoiseGeneWithSoS(float *dev_cos_value, float *dev_sin_value, unsigned int length, unsigned int path_num,
	unsigned long long uniform_seed, float omega_amp, float delta_alpha, float delta_omega,
	float delta_t);

__global__ void cudaNoiseOmegaCulc(float *dev_omega_n_I, float *dev_omega_n_Q, unsigned int length,
	float omega_amp, float delta_alpha, float delta_omega);

__global__ void cudaNoiseSoSCulc(float *dev_cos_value, float *dev_sin_value,
	unsigned int pitch_width, unsigned int width, unsigned int heigth, float delta_t,
	float *dev_omega_n_I, float *dev_omega_n_Q, float *dev_phi_n);

__global__ void cudaNoiseSoSSum(float *dev_cos_value, float *dev_sin_value,
	unsigned int pitch_width, unsigned int width, unsigned int heigth, float sum_amp);