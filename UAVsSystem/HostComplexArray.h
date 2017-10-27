#pragma once

#include <memory>
#include <complex>
#include "common.h"
#include "HostVector.h"
using namespace std;

template<class _T, class _Alloc = std::allocator<_T>>
class HostComplexArray {
	static_assert(is_arithmetic<_T>::value, "'_T' must be a arithmetic type.");
private:
	HostVector<_T, _Alloc> realVector;
	HostVector<_T, _Alloc> imagVector;
	unique_ptr<ComplexRef<_T>> upCurrentItem;
public:
	HostComplexArray(const size_t &size){
		this->resize(size);
	}

	HostComplexArray(const size_t &size, const _T &value){
		this->resize(size, value);
	}

	HostComplexArray(const size_t &size, const _T &real, const _T &imag){
		this->resize(size, real, imag);
	}

	~HostComplexArray(){
	}

	ComplexRef<_T>& operator[](const size_t& _Off) {
		if (_Off >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			MacroUtils_ClassName(*this), MacroUtils_FunctionName(), MacroUtils_VariableName(_Off), "null.");
		upCurrentItem.reset(new ComplexRef<_T>(this->realVector[_Off], this->imagVector[_Off]));
		return (*upCurrentItem);
	}

	void set(const size_t& index, const complex<_T>& value) {
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			MacroUtils_ClassName(*this), MacroUtils_FunctionName(), MacroUtils_VariableName(index), "null.");
		this->realVector[index] = value.real();
		this->imagVector[index] = value.imag();
	}

	complex<_T> get(const size_t& index) const {
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			MacroUtils_ClassName(*this), MacroUtils_FunctionName(), MacroUtils_VariableName(index), "null.");
		return complex<_T>(this->realVector[index], this->imagVector[index]);
	}

	void setReal(const size_t& index, const _T &value) {
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			MacroUtils_ClassName(*this), MacroUtils_FunctionName(), MacroUtils_VariableName(index), "null.");
		this->realVector[index] = value;
	}

	_T getReal(const size_t& index) const {
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			MacroUtils_ClassName(*this), MacroUtils_FunctionName(), MacroUtils_VariableName(index), "null.");
		return this->realVector[index];
	}

	void setImag(const size_t& index, const _T &value) {
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			MacroUtils_ClassName(*this), MacroUtils_FunctionName(), MacroUtils_VariableName(index), "null.");
		this->imagVector[index] = value;
	}

	_T getImag(const size_t& index) const {
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			MacroUtils_ClassName(*this), MacroUtils_FunctionName(), MacroUtils_VariableName(index), "null.");
		return this->imagVector[index];
	}

	HostVector<_T, _Alloc>& getRealArray() {
		return this->realVector;
	}

	HostVector<_T, _Alloc>& getImagArray() {
		return this->imagVector;
	}

	size_t size() const{
		size_t realSize = this->realVector.size();
		size_t imagSize = this->imagVector.size();
		return realSize < imagSize ? realSize : imagSize;
	}

	void resize(const size_t &size){
		this->realVector.resize(size, 0);
		this->imagVector.resize(size, 0);

		this->realVector.shrink_to_fit();
		this->imagVector.shrink_to_fit();
	}

	void resize(const size_t &size, const _T &value){
		this->realVector.resize(size, value);
		this->imagVector.resize(size, value);

		this->realVector.shrink_to_fit();
		this->imagVector.shrink_to_fit();
	}

	void resize(const size_t &size, const _T &real, const _T &imag){
		this->realVector.resize(size, real);
		this->imagVector.resize(size, imag);

		this->realVector.shrink_to_fit();
		this->imagVector.shrink_to_fit();
	}

	void fit2MaxSize(){
		size_t realSize = this->realVector.size();
		size_t imagSize = this->imagVector.size();

		if (realSize > imagSize){
			size_t _Size = realSize - imagSize;
			this->imagVector.reserve(_Size);
			this->imagVector.append(_Size);
		}
		else if (realSize < imagSize){
			size_t _Size = imagSize - realSize;
			this->realVector.reserve(_Size);
			this->realVector.append(_Size);
		}
	}

	void fit2MinSize(){
		size_t realSize = this->realVector.size();
		size_t imagSize = this->imagVector.size();

		if (realSize > imagSize){
			this->realVector.erase(this->realVector.begin() + imageSize, this->realVector.end());
			this->realVector.shrink_to_fit();
		}
		else if (realSize < imagSize){
			this->imagVector.erase(this->imagVector.begin() + realSize, this->imagVector.end());
			this->imagVector.shrink_to_fit();
		}
	}

	void fit2RealSize(){
		size_t realSize = this->realVector.size();
		size_t imagSize = this->imagVector.size();

		if (realSize > imagSize){
			size_t _Size = realSize - imagSize;
			this->imagVector.reserve(_Size);
			this->imagVector.append(_Size);
		}
		else if (realSize < imagSize){
			this->imagVector.erase(this->imagVector.begin() + realSize, this->imagVector.end());
			this->imagVector.shrink_to_fit();
		}
	}

	void fit2ImagSize(){
		size_t realSize = this->realVector.size();
		size_t imagSize = this->imagVector.size();

		if (realSize > imagSize){
			this->realVector.erase(this->realVector.begin() + imageSize, this->realVector.end());
			this->realVector.shrink_to_fit();
		}
		else if (realSize < imagSize){
			size_t _Size = imagSize - realSize;
			this->realVector.reserve(_Size);
			this->realVector.append(_Size);
		}
	}
};