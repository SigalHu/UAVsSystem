#pragma once

#include <stdexcept>
#include <memory>
#include "cuda.h"
using namespace std;

template<class _T>
class CudaMemoryService final{
	static_assert(is_arithmetic<_T>::value, "'_T' must be a arithmetic type.");
private:
	static unsigned int useCount;
	const unsigned int deviceId;

	void *ptr;
	size_t width;
	size_t height;
	size_t pitch;

	size_t pos;
private:
	CudaMemoryService() = default;
	CudaMemoryService(const CudaMemoryService& item) = default;
	CudaMemoryService<_T>& operator=(const CudaMemoryService& _Right) = default;

	_T* getPtr() const{
		return static_cast<_T*>(this->ptr);
	}
public:
	CudaMemoryService(const unsigned int &deviceId, const size_t &width) throw(bad_alloc)
		:deviceId(deviceId), width(width * sizeof(_T)), height(1), pitch(width * sizeof(_T)), pos(0){
		if (this->deviceId >= CudaUtils::getDeviceCount())
			throw bad_alloc();
		if (!CudaUtils::setDevice(this->deviceId))
			throw bad_alloc();
		if (!CudaCoreUtils::malloc(&(this->ptr), this->width)){
			throw bad_alloc();
		}
	}

	CudaMemoryService(const unsigned int &deviceId, const size_t &width, const size_t &height) throw(bad_alloc)
		:deviceId(deviceId), width(width * sizeof(_T)), height(height), pos(0){
		if (this->deviceId >= CudaUtils::getDeviceCount())
			throw bad_alloc();
		if (!CudaUtils::setDevice(this->deviceId))
			throw bad_alloc();
		if (!CudaCoreUtils::mallocPitch(&(this->ptr), &(this->pitch), this->width, this->height)){
			throw bad_alloc();
		}
	}

	~CudaMemoryService(){
		if (this->ptr)
			CudaCoreUtils::free(this->ptr);
		if (--CudaMemoryService::deviceId == 0)
			CudaUtils::resetDevice();
	}

	int getDeviceId() const{
		return this->deviceId;
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
			throw invalid_argument(MacroUtils_VariableName(dest) + " can not be nullptr.");
	//	if (!CudaUtils::setDevice(this->deviceId))
			//throw runtime_error("GPU can not be set to " + 0 + ".");

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
				throw invalid_argument(MacroUtils_VariableName(nBytes) + " can not be less than " + MacroUtils_VariableName(width) + ".");
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
	void call(typename remove_reference<_T1>::type& cudaTask, 
		initializer_list<shared_ptr<CudaMemoryService>> others = {}) throw(invalid_argument){
		static_assert(is_base_of<CudaTask, _T1>::value, "cudaTask must be a subclass of CudaTask.");

		vector<void*> otherPtrs;
		otherPtrs.reserve(others.size());
		for (const shared_ptr<CudaMemoryService> &item : others){
			otherPtrs.emplace_back(item->getPtr());
		}
		
		if (!CudaUtils::setDevice(this->deviceId))
			throw invalid_argument("GPU memory reading failed.");
		cudaTask(this->getPtr(), otherPtrs);
	}
};

template<class _T>
unsigned int CudaMemoryService<_T>::useCount = 0;

