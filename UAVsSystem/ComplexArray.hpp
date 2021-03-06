#pragma once
#include <valarray>
#include <memory>
#include <complex>
#include "common_exception.h"
#include "common_utils.h"
#include "ComplexRef.h"

template<class _T>
class ComplexArray {
	static_assert(std::is_arithmetic<_T>::value, "'_T' must be a arithmetic type.");
private:
	std::valarray<_T> realArray;
	std::valarray<_T> imagArray;
	std::unique_ptr<ComplexRef<_T>> upCurrentItem;
private:
	static std::string getClassName(){
		return MacroUtils_ClassName(ComplexArray);
	}
public:
	ComplexArray(const size_t &size){
		this->resize(size);
	}

	ComplexArray(const size_t &size, const _T &value){
		this->resize(size, value);
	}

	ComplexArray(const size_t &size, const _T &real, const _T &imag){
		this->resize(size, real, imag);
	}

	ComplexRef<_T>& operator[](const size_t& _Off){
		if (_Off >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(_Off), "null.");
		upCurrentItem.reset(new ComplexRef<_T>(this->realArray[_Off], this->imagArray[_Off]));
		return (*upCurrentItem);
	}

	void set(const size_t& index, const std::complex<_T>& value){
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
		this->realArray[index] = value.real();
		this->imagArray[index] = value.imag();
	}

	std::complex<_T> get(const size_t& index) const{
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
		return std::complex<_T>(this->realArray[index], this->imagArray[index]);
	}

	void setReal(const size_t& index, const _T &value){
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
		this->realArray[index] = value;
	}

	_T getReal(const size_t& index) const{
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
		return this->realArray[index];
	}

	void setImag(const size_t& index, const _T &value){
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
		this->imagArray[index] = value;
	}

	_T getImag(const size_t& index) const{
		if (index >= this->size())
			throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
			getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
		return this->imagArray[index];
	}

	std::valarray<_T>& getRealArray() const{
		return this->realArray;
	}

	std::valarray<_T>& getImagArray() const{
		return this->imagArray;
	}

	size_t size() const{
		size_t realSize = this->realArray.size();
		size_t imagSize = this->imagArray.size();
		return realSize < imagSize ? realSize : imagSize;
	}

	bool checkSize() const{
		return this->realArray.size() == this->imagArray.size();
	}

	void resize(const size_t &size){
		this->realArray.resize(size, 0);
		this->imagArray.resize(size, 0);
	}

	void resize(const size_t &size, const _T &value){
		this->realArray.resize(size, value);
		this->imagArray.resize(size, value);
	}

	void resize(const size_t &size, const _T &real, const _T &imag){
		this->realArray.resize(size, real);
		this->imagArray.resize(size, imag);
	}

	void fit2MaxSize(){
		size_t realSize = this->realArray.size();
		size_t imagSize = this->imagArray.size();

		if (realSize > imagSize){
			std::valarray<_T> _Array(0, realSize);
			for (int ii = 0; ii < imagSize; ++ii){
				_Array[ii] = this->imagArray[ii];
			}
			this->imagArray = _Array;
		}
		else if (realSize < imagSize){
			std::valarray<_T> _Array(0, imagSize);
			for (int ii = 0; ii < realSize; ++ii){
				_Array[ii] = this->realArray[ii];
			}
			this->realArray = _Array;
		}
	}

	void fit2MinSize(){
		size_t realSize = this->realArray.size();
		size_t imagSize = this->imagArray.size();

		if (realSize > imagSize){
			std::valarray<_T> _Array(&(this->realArray[0]), imagSize);
			this->realArray = _Array;
		}
		else if (realSize < imagSize){
			std::valarray<_T> _Array(&(this->imagArray[0]), realSize);
			this->imagArray = _Array;
		}
	}

	void fit2RealSize(){
		size_t realSize = this->realArray.size();
		size_t imagSize = this->imagArray.size();

		if (realSize > imagSize){
			std::valarray<_T> _Array(0, realSize);
			for (int ii = 0; ii < imagSize; ++ii){
				_Array[ii] = this->imagArray[ii];
			}
			this->imagArray = _Array;
		}
		else if (realSize < imagSize){
			std::valarray<_T> _Array(&(this->imagArray[0]), realSize);
			this->imagArray = _Array;
		}
	}

	void fit2ImagSize(){
		size_t realSize = this->realArray.size();
		size_t imagSize = this->imagArray.size();

		if (realSize > imagSize){
			std::valarray<_T> _Array(&(this->realArray[0]), imagSize);
			this->realArray = _Array;
		}
		else if (realSize < imagSize){
			std::valarray<_T> _Array(0, imagSize);
			for (int ii = 0; ii < realSize; ++ii){
				_Array[ii] = this->realArray[ii];
			}
			this->realArray = _Array;
		}
	}
};