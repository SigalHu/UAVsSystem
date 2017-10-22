#pragma once

#include "UtilsDeclaration.h"
#include <stdexcept>
#include "CudaTask.h"
using namespace std;

template<class _T>
class CudaMemoryService final{
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
		:width(width * sizeof(_T)), height(1), pitch(width * sizeof(_T)), pos(0){
		if (!CudaCoreUtils::malloc(&(this->ptr), this->width)){
			this->ptr = nullptr;
			throw bad_alloc();
		}
	}

	CudaMemoryService(const size_t &width, const size_t &height) throw(bad_alloc)
		:width(width * sizeof(_T)), height(height),pos(0){
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

	size_t getSizeBytes() const{
		return this->width*this->height;
	}

	size_t getCapacityBytes() const{
		return this->pitch*this->height;
	}

	void initRead(){
		this->pos = 0;
	}

	bool hasNext() const{
		return (this->pos < this->pitch*this->height);
	}

	size_t read(void* const &dest, size_t nBytes) const throw(invalid_argument){
		if (dest == nullptr)
			throw invalid_argument(string(varName(dest)) + " can not be nullptr.");

		if (this->height == 1){
			int _Size = this->getSizeBytes();

			if (this->pos >= _Size)
				return 0;
			if (this->pos + nBytes > _Size)
				nBytes = _Size - this->pos;
			if (cudaSuccess != cudaMemcpy(dest, this->ptr + this->pos, nBytes, cudaMemcpyDeviceToHost))
				throw invalid_argument("GPU memory reading failed.");
			this->pos += nBytes;
		}
		else {
			int _MaxPos = this->getCapacityBytes() - (this->pitch - this->width);

			if (this->pos >= _MaxPos)
				return 0;

			int _Height = nBytes / this->width;
			int _CurrentHeight = this->pos / this->pitch;
			if (_Height == 0)
				throw invalid_argument(string(varName(nBytes)) + " can not be less than " + string(varName(width)) + ".");
			else if (_CurrentHeight + _Height > this->height)
				_Height = this->height - _CurrentHeight;
			if (cudaSuccess != cudaMemcpy2D(dest, this->width, this->ptr + this->pos, this->pitch, this->width, _Height, cudaMemcpyDeviceToHost))
				throw invalid_argument("GPU memory reading failed.");
			nBytes = this->width*_Height;
			this->pos += this->pitch*_Height;
		}
		return nBytes;
	}

	template<class _T1>
	void call(_T1 cudaTask){
		static_assert(is_base_of<CudaTask, _T1>::value, "cudaTask must be a subclass of CudaTask.");

		cudaTask();
	}
};

