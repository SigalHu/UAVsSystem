#include "common_exception.h"
#include "common_utils.h"
#include "ComplexRef.h"
#include "ComplexArray.h"

template<class _T>
ComplexArray<_T>::ComplexArray(const size_t &size){
	this->resize(size);
}

template<class _T>
ComplexArray<_T>::ComplexArray(const size_t &size, const _T &value){
	this->resize(size, value);
}

template<class _T>
ComplexArray<_T>::ComplexArray(const size_t &size, const _T &real, const _T &imag){
	this->resize(size, real, imag);
}

template<class _T>
std::string ComplexArray<_T>::getClassName(){
	return MacroUtils_ClassName(ComplexArray);
}

template<class _T>
ComplexRef<_T>& ComplexArray<_T>::operator[](const size_t& _Off) {
	if (_Off >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(_Off), "null.");
	upCurrentItem.reset(new ComplexRef<_T>(this->realArray[_Off], this->imagArray[_Off]));
	return (*upCurrentItem);
}

template<class _T>
void ComplexArray<_T>::set(const size_t& index, const std::complex<_T>& value) {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	this->realArray[index] = value.real();
	this->imagArray[index] = value.imag();
}

template<class _T>
std::complex<_T> ComplexArray<_T>::get(const size_t& index) const {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	return std::complex<_T>(this->realArray[index], this->imagArray[index]);
}

template<class _T>
void ComplexArray<_T>::setReal(const size_t& index, const _T &value) {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	this->realArray[index] = value;
}

template<class _T>
_T ComplexArray<_T>::getReal(const size_t& index) const {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	return this->realArray[index];
}

template<class _T>
void ComplexArray<_T>::setImag(const size_t& index, const _T &value) {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	this->imagArray[index] = value;
}

template<class _T>
_T ComplexArray<_T>::getImag(const size_t& index) const {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	return this->imagArray[index];
}

template<class _T>
std::valarray<_T>& ComplexArray<_T>::getRealArray() const{
	return this->realArray;
}

template<class _T>
std::valarray<_T>& ComplexArray<_T>::getImagArray() const{
	return this->imagArray;
}

template<class _T>
size_t ComplexArray<_T>::size() const{
	size_t realSize = this->realArray.size();
	size_t imagSize = this->imagArray.size();
	return realSize < imagSize ? realSize : imagSize;
}

template<class _T>
bool ComplexArray<_T>::checkSize() const{
	return this->realArray.size() == this->imagArray.size();
}

template<class _T>
void ComplexArray<_T>::resize(const size_t &size){
	this->realArray.resize(size, 0);
	this->imagArray.resize(size, 0);
}

template<class _T>
void ComplexArray<_T>::resize(const size_t &size, const _T &value){
	this->realArray.resize(size, value);
	this->imagArray.resize(size, value);
}

template<class _T>
void ComplexArray<_T>::resize(const size_t &size, const _T &real, const _T &imag){
	this->realArray.resize(size, real);
	this->imagArray.resize(size, imag);
}

template<class _T>
void ComplexArray<_T>::fit2MaxSize(){
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

template<class _T>
void ComplexArray<_T>::fit2MinSize(){
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

template<class _T>
void ComplexArray<_T>::fit2RealSize(){
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

template<class _T>
void ComplexArray<_T>::fit2ImagSize(){
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