#define _USE_MATH_DEFINES
#include "device_launch_parameters.h"
#include "curand.h"
#include <math.h>
#include <time.h>
#include "common.h"
#include "cuda_definition.h"
#include "cuda_api.h"
#include "CudaSoSUtils.h"

std::string CudaSoSUtils::getClassName(){
	return MacroUtils_ClassName(CudaSoSUtils);
}

template<class _Alloc>
void CudaSoSUtils::noiseGene(DeviceVector<float, _Alloc> &noiseI, DeviceVector<float, _Alloc> &noiseQ, const float &fs,
	const float &avgPower, const unsigned int &pathNum, const float &maxFd, const float &deltaOmega){
	if (noiseI.empty() || noiseQ.empty())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE, getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(noiseI),
		MacroUtils_VariableName(noiseI).append(" and ").append(MacroUtils_VariableName(noiseQ)).append(" can not be empty."));
	if (noiseI.size() != noiseQ.size())
		throw SystemException(SystemCodeEnum::NOT_EQUAL, getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(noiseI),
		StringUtils::format(MacroUtils_VariableName(noiseI).append("(size=%d) is not equal to ").append(MacroUtils_VariableName(noiseQ)).append("(size=%d)."),
		noiseI.size(), noiseQ.size()));
	unsigned int tmp = 1;
	while (tmp < pathNum)
		tmp <<= 1;
	if (pathNum != tmp)
		throw SystemException(SystemCodeEnum::NOT_EQUAL, getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(pathNum),
		StringUtils::format(MacroUtils_VariableName(pathNum).append("(=%d) is not equal to pow(2,n)."), pathNum));

	float omegaAmp = 2 * M_PI*maxFd;
	float deltaAlpha = (2 * M_PI - 2 * M_PI / (pathNum + 1)) / (pathNum - 1);
	float deltaT = 1 / fs;
	float sumAmp = sqrtf(avgPower / pathNum);

	size_t len = noiseI.size();
	dim3 blockNum, threadNum(THREAD_NUM_PER_BLOCK);
	if (pathNum <= THREAD_NUM_PER_BLOCK){
		blockNum.y = len;
	}
	else{
		blockNum.x = pathNum / THREAD_NUM_PER_BLOCK;
		blockNum.y = len / blockNum.x + (len % blockNum.x) != 0;
	}
	if (blockNum.y > gridDim.y){
		blockNum.y = gridDim.y;
	}

	cudaNoiseGeneWithSoS << <blockNum, threadNum, 2 * pathNum*sizeof(float) >> >(
		raw_pointer_cast(noiseI.data()), raw_pointer_cast(noiseQ.data()),
		len, pathNum, time(NULL), omegaAmp, deltaAlpha, deltaOmega, deltaT, sumAmp);

	cudaError_t error = cudaGetLastError();
	if (cudaSuccess != error)
		throw SystemException(SystemCodeEnum::CUDA_CALL_ERROR, getClassName(), MacroUtils_CurFunctionName(),
		MacroUtils_FunctionName(cudaNoiseGeneWithSoS), cudaGetErrorString(error));
}
//
//template<class _Alloc>
//void CudaSoSUtils::noiseOmegaCulc(const dim3 &blockNum, const dim3 &threadNum,
//	DeviceVector<float, _Alloc> &devOmegaI, DeviceVector<float, _Alloc> &devOmegaQ,
//	const float &maxFd, const float &deltaOmega){
//	if (devOmegaI.empty() || devOmegaQ.empty())
//		throw SystemException(SystemCodeEnum::OUT_OF_RANGE, MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(), MacroUtils_VariableName(devOmegaI),
//		MacroUtils_VariableName(devOmegaI).append(" and ").append(MacroUtils_VariableName(devOmegaQ)).append(" can not be empty."));
//	if (devOmegaI.size() != devOmegaQ.size())
//		throw SystemException(SystemCodeEnum::NOT_EQUAL, MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(), MacroUtils_VariableName(devOmegaI),
//		StringUtils::format(MacroUtils_VariableName(devOmegaI).append("(size=%d) is not equal to ").append(MacroUtils_VariableName(devOmegaQ)).append("(size=%d)."),
//		devOmegaI.size(), devOmegaQ.size()));
//	
//	unsigned int pathNum = devOmegaI.size();
//	cudaNoiseOmegaCulc << <blockNum, threadNum >> >(raw_pointer_cast(devOmegaI.data()), raw_pointer_cast(devOmegaQ.data()),
//		pathNum, 2 * M_PI*maxFd, (2 * M_PI - 2 * M_PI / (pathNum + 1)) / (pathNum - 1), deltaOmega);
//
//	cudaError_t error = cudaGetLastError();
//	if (cudaSuccess != error)
//		throw SystemException(SystemCodeEnum::CUDA_CALL_ERROR, MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(),
//		MacroUtils_FunctionName(cudaNoiseOmegaCulc), cudaGetErrorString(error));
//}
//
//bool CudaSoSUtils::noiseSoSCulc(float* const &devCosValue, float* const &devSinValue, const dim3 &blockNum, const dim3 &threadNum,
//	const unsigned int &pitchWidth, const unsigned int &width, const unsigned int &heigth, const float &deltaT,
//	float* const &devOmegaI, float* const &devOmegaQ, float* const &devPhi) {
//	//if (devCosValue == nullptr)
//	//	throw SystemException(SystemCodeEnum::NULL_POINTER,
//	//		StringUtils::format(SystemCodeEnum::NULL_POINTER.getInfo(),
//	//		MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(), MacroUtils_VariableName(devCosValue), "null."));
//	//if (devSinValue == nullptr)
//	//	throw SystemException(SystemCodeEnum::NULL_POINTER,
//	//		StringUtils::format(SystemCodeEnum::NULL_POINTER.getInfo(),
//	//		MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(), MacroUtils_VariableName(devSinValue), "null."));
//	//if (devOmegaI == nullptr)
//	//	throw SystemException(SystemCodeEnum::NULL_POINTER,
//	//		StringUtils::format(SystemCodeEnum::NULL_POINTER.getInfo(),
//	//		MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(), MacroUtils_VariableName(devOmegaI), "null."));
//	//if (devOmegaQ == nullptr)
//	//	throw SystemException(SystemCodeEnum::NULL_POINTER,
//	//		StringUtils::format(SystemCodeEnum::NULL_POINTER.getInfo(),
//	//		MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(), MacroUtils_VariableName(devOmegaQ), "null."));
//	//if (devPhi == nullptr)
//	//	throw SystemException(SystemCodeEnum::NULL_POINTER,
//	//		StringUtils::format(SystemCodeEnum::NULL_POINTER.getInfo(),
//	//		MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(), MacroUtils_VariableName(devPhi), "null."));
//
//	cudaNoiseSoSCulc << <blockNum, threadNum >> >(devCosValue, devSinValue, 
//		pitchWidth, width, heigth, deltaT, devOmegaI, devOmegaQ, devPhi);
//	return cudaSuccess == cudaGetLastError();
//}
//
//bool CudaSoSUtils::noiseSoSSum(float* const &devCosValue, float* const &devSinValue, const dim3 &blockNum, const dim3 &threadNum,
//	const unsigned int &pitchWidth, const unsigned int &width, const unsigned int &heigth, const float &sumAmp){
//	//if (devCosValue == nullptr)
//	//	throw SystemException(SystemCodeEnum::NULL_POINTER,
//	//		StringUtils::format(SystemCodeEnum::NULL_POINTER.getInfo(),
//	//		MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(), MacroUtils_VariableName(devCosValue), "null."));
//	//if (devSinValue == nullptr)
//	//	throw SystemException(SystemCodeEnum::NULL_POINTER,
//	//		StringUtils::format(SystemCodeEnum::NULL_POINTER.getInfo(),
//	//		MacroUtils_ClassName(CudaSoSUtils), MacroUtils_CurFunctionName(), MacroUtils_VariableName(devSinValue), "null."));
//
//	cudaNoiseSoSSum << <blockNum, threadNum >> >(devCosValue, devSinValue, pitchWidth, width, heigth, sumAmp);
//	return cudaSuccess == cudaGetLastError();
//}
//
//bool CudaSoSUtils::cudaNoiseGeneWithSoS(float *noise_I, float *noise_Q, float fs, float time_spend,
//	float power_avg, unsigned int path_num, float fd_max, float delta_omega){
//	bool isSucceed = true;
//	cudaError_t cudaStatus;
//	float *dev_omega_n_I = NULL,
//		*dev_omega_n_Q = NULL,
//		*dev_phi_n = NULL,
//		*dev_cos_value = NULL,
//		*dev_sin_value = NULL;
//
//	try{
//		curandStatus_t cuRandStatus;
//		curandGenerator_t gen;
//		const unsigned int threadNumLimit = 256;
//		size_t blockNum, threadNum;
//		dim3 blockNum2D, threadNum2D;
//		size_t col_num = fs*time_spend;
//
//		cudaStatus = cudaSetDevice(0);
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//
//		threadNum = path_num <= threadNumLimit ? path_num : threadNumLimit;
//		blockNum = path_num % threadNum ?
//			path_num / threadNum + 1 :
//			path_num / threadNum;
//
//		cudaStatus = cudaMalloc((void **)&dev_omega_n_I, path_num*sizeof(float));
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//		cudaStatus = cudaMalloc((void **)&dev_omega_n_Q, path_num*sizeof(float));
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//
//		cudaNoiseOmegaCulc << <blockNum, threadNum >> >(dev_omega_n_I, dev_omega_n_Q, path_num,
//			2 * M_PI*fd_max, (2 * M_PI - 2 * M_PI / (path_num + 1)) / (path_num - 1), delta_omega);
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//
//		cudaStatus = cudaMalloc((void **)&dev_phi_n, 2 * path_num*sizeof(float));
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//
//		cuRandStatus = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//		if (cuRandStatus != CURAND_STATUS_SUCCESS){
//			throw false;
//		}
//		cuRandStatus = curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
//		if (cuRandStatus != CURAND_STATUS_SUCCESS){
//			throw false;
//		}
//		cuRandStatus = curandGenerateUniform(gen, dev_phi_n, 2 * path_num);
//		if (cuRandStatus != CURAND_STATUS_SUCCESS){
//			throw false;
//		}
//
//		threadNum2D.x = BLOCK_DIM_X;
//		threadNum2D.y = BLOCK_DIM_Y;
//		blockNum2D.x = col_num % threadNum2D.x ?
//			col_num / threadNum2D.x + 1 :
//			col_num / threadNum2D.x;
//		if (blockNum2D.x > GRID_DIM_X){
//			unsigned int gridNum = blockNum2D.x % GRID_DIM_X ?
//				blockNum2D.x / GRID_DIM_X + 1 :
//				blockNum2D.x / GRID_DIM_X;
//			blockNum2D.x = blockNum2D.x % gridNum ?
//				blockNum2D.x / gridNum + 1 :
//				blockNum2D.x / gridNum;
//		}
//		blockNum2D.y = path_num % threadNum2D.y ?
//			path_num / threadNum2D.y + 1 :
//			path_num / threadNum2D.y;
//
//
//		size_t pitch;
//		cudaStatus = cudaMallocPitch((void **)&dev_cos_value, &pitch, col_num*sizeof(float), blockNum2D.y);
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//		cudaStatus = cudaMallocPitch((void **)&dev_sin_value, &pitch, col_num*sizeof(float), blockNum2D.y);
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//		cudaNoiseSoSCulc << <blockNum2D, threadNum2D >> >(dev_cos_value, dev_sin_value, pitch / sizeof(float),
//			col_num, path_num, 1 / fs, dev_omega_n_I, dev_omega_n_Q, dev_phi_n);
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//
//		threadNum = col_num <= threadNumLimit ? col_num : threadNumLimit;
//		blockNum = col_num % threadNum ?
//			col_num / threadNum + 1 :
//			col_num / threadNum;
//		if (blockNum > GRID_DIM_X){
//			unsigned int gridNum = blockNum % GRID_DIM_X ?
//				blockNum / GRID_DIM_X + 1 :
//				blockNum / GRID_DIM_X;
//			blockNum = blockNum % gridNum ?
//				blockNum / gridNum + 1 :
//				blockNum / gridNum;
//		}
//		cudaNoiseSoSSum << <blockNum, threadNum >> >(dev_cos_value, dev_sin_value, pitch / sizeof(float),
//			col_num, blockNum2D.y, sqrtf(power_avg / path_num));
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//
//		cudaStatus = cudaMemcpy((void *)noise_I, (void *)dev_cos_value, col_num*sizeof(float), cudaMemcpyDeviceToHost);
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//		cudaStatus = cudaMemcpy((void *)noise_Q, (void *)dev_sin_value, col_num*sizeof(float), cudaMemcpyDeviceToHost);
//		if (cudaStatus != cudaSuccess) {
//			throw false;
//		}
//	}
//	catch (bool &msg){
//		isSucceed = msg;
//	}
//
//	if (dev_omega_n_I)
//		cudaFree(dev_omega_n_I);
//	if (dev_omega_n_Q)
//		cudaFree(dev_omega_n_Q);
//	if (dev_phi_n)
//		cudaFree(dev_phi_n);
//	if (dev_cos_value)
//		cudaFree(dev_cos_value);
//	if (dev_sin_value)
//		cudaFree(dev_sin_value);
//
//	cudaStatus = cudaDeviceReset();
//	if (cudaStatus != cudaSuccess) {
//		return false;
//	}
//	return isSucceed;
//}
