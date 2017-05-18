#pragma once

bool cudaNoiseGene(float *noise_I, float *noise_Q, size_t length, float mean, float stddev);
bool cudaNoiseGeneWithSoS(float *noise_I, float *noise_Q, float fs, float time_spend,
	float power_avg = 1, unsigned int path_num = 16, float fd_max = 50, float delta_omega = 0);