#include "common.h"
#include "HostVector.h"
#include "HostComplexArray.h"

template<class _T, class _Alloc>
std::string HostComplexArray<_T, _Alloc>::getClassName(){
	return MacroUtils_ClassName(HostComplexArray);
}

template<class _T, class _Alloc>
HostComplexArray<_T, _Alloc>::HostComplexArray(const size_t &size){
	this->resize(size);
}

template<class _T, class _Alloc>
HostComplexArray<_T, _Alloc>::HostComplexArray(const size_t &size, const _T &value){
	this->resize(size, value);
}

template<class _T, class _Alloc>
HostComplexArray<_T, _Alloc>::HostComplexArray(const size_t &size, const _T &real, const _T &imag){
	this->resize(size, real, imag);
}

template<class _T, class _Alloc>
ComplexRef<_T>& HostComplexArray<_T, _Alloc>::operator[](const size_t& _Off) {
	if (_Off >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(_Off), "null.");
	upCurrentItem.reset(new ComplexRef<_T>(this->realVector[_Off], this->imagVector[_Off]));
	return (*upCurrentItem);
}

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::set(const size_t& index, const std::complex<_T>& value) {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	this->realVector[index] = value.real();
	this->imagVector[index] = value.imag();
}

template<class _T, class _Alloc>
std::complex<_T> HostComplexArray<_T, _Alloc>::get(const size_t& index) const {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	return std::complex<_T>(this->realVector[index], this->imagVector[index]);
}

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::setReal(const size_t& index, const _T &value) {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	this->realVector[index] = value;
}

template<class _T, class _Alloc>
_T HostComplexArray<_T, _Alloc>::getReal(const size_t& index) const {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	return this->realVector[index];
}

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::setImag(const size_t& index, const _T &value) {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	this->imagVector[index] = value;
}

template<class _T, class _Alloc>
_T HostComplexArray<_T, _Alloc>::getImag(const size_t& index) const {
	if (index >= this->size())
		throw SystemException(SystemCodeEnum::OUT_OF_RANGE,
		getClassName(), MacroUtils_CurFunctionName(), MacroUtils_VariableName(index), "null.");
	return this->imagVector[index];
}

template<class _T, class _Alloc>
HostVector<_T, _Alloc>& HostComplexArray<_T, _Alloc>::getRealArray() const{
	return this->realVector;
}

template<class _T, class _Alloc>
HostVector<_T, _Alloc>& HostComplexArray<_T, _Alloc>::getImagArray() const{
	return this->imagVector;
}

template<class _T, class _Alloc>
size_t HostComplexArray<_T, _Alloc>::size() const{
	size_t realSize = this->realVector.size();
	size_t imagSize = this->imagVector.size();
	return realSize < imagSize ? realSize : imagSize;
}

template<class _T, class _Alloc>
bool HostComplexArray<_T, _Alloc>::checkSize() const{
	return this->realVector.size() == this->imagVector.size();
}

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::resize(const size_t &size){
	this->realVector.resize(size, 0);
	this->imagVector.resize(size, 0);

	this->realVector.shrink_to_fit();
	this->imagVector.shrink_to_fit();
}

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::resize(const size_t &size, const _T &value){
	this->realVector.resize(size, value);
	this->imagVector.resize(size, value);

	this->realVector.shrink_to_fit();
	this->imagVector.shrink_to_fit();
}

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::resize(const size_t &size, const _T &real, const _T &imag){
	this->realVector.resize(size, real);
	this->imagVector.resize(size, imag);

	this->realVector.shrink_to_fit();
	this->imagVector.shrink_to_fit();
}

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::fit2MaxSize(){
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

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::fit2MinSize(){
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

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::fit2RealSize(){
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

template<class _T, class _Alloc>
void HostComplexArray<_T, _Alloc>::fit2ImagSize(){
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