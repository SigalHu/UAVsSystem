#pragma once

#include "UtilsDeclaration.h"
#include <stdexcept>
using namespace std;

template<class _T>
class CudaMemoryService final{
	friend class CudaAlgorithmUtils;
private:
	void *ptr;
	size_t width;
	size_t height;
	size_t pitch;

	size_t pos;
private:
	_T *getPtr() const{
		return this->ptr;
	}
public:
	CudaMemoryService(const size_t &width) throw(bad_alloc)
		:width(width * sizeof(_T)), height(1), pitch(0){
		if (!CudaCoreUtils::malloc(&(this->ptr), this->width)){
			this->ptr = nullptr;
			throw bad_alloc();
		}
	}

	CudaMemoryService(const size_t &width, const size_t &height) throw(bad_alloc)
		:width(width * sizeof(_T)), height(height){
		if (!CudaCoreUtils::mallocPitch(&(this->ptr), &(this->pitch), this->width, this->height)){
			this->ptr = nullptr;
			throw bad_alloc();
		}
	}

	~CudaMemoryService(){
		if (this->ptr)
			CudaCoreUtils::free(this->ptr);
	}

	size_t getWidth() const{
		return this->width / sizeof(_T);
	}

	size_t getWidthBytes() const{
		return this->width;
	}

	size_t getPitchBytes() const{
		return this->pitch;
	}

	size_t getHeight() const{
		return this->height;
	}

	void resetPos(){
		this->pos = 0;
	}

	//size_t read(void* const &dest, const size_t &nBytes) const throw(invalid_argument){
	//	if (dest == nullptr)
	//		throw invalid_argument(string(varName(dest)) + " can not be nullptr.");
	//	
	//	if (this->height == 1){
	//		if (cudaSuccess != cudaMemcpy((void *)noise_I, (void *)dev_cos_value, col_num*sizeof(float), cudaMemcpyDeviceToHost))

	//	}
	//	else {
	//      cudaMemcpy2D
	//	}
	//}

	//size_t read(void* const &dest, size_t nBytes, size_t pitchBytes) const throw(invalid_argument){
	//	if (dest == nullptr)
	//		throw invalid_argument(string(varName(dest)) + " can not be nullptr.");
	//}
};

