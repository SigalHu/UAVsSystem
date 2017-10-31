#include "ComplexRef.h"

template<class _T>
ComplexRef<_T>::ComplexRef(_Tref realValue, _Tref imagValue)
:std::complex<_T>(realValue, imagValue), realValue(realValue), imagValue(imagValue){
}

template<class _T>
_T ComplexRef<_T>::real(){
	return (this->realValue);
}

template<class _T>
_T ComplexRef<_T>::imag(){
	return (this->imagValue);
}

template<class _T>
ComplexRef<_T>& ComplexRef<_T>::operator=(const _Tref _Right){
	std::complex<_T>::operator=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
ComplexRef<_T>& ComplexRef<_T>::operator+=(const _Tref _Right){
	std::complex<_T>::operator+=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
ComplexRef<_T>& ComplexRef<_T>::operator-=(const _Tref _Right){
	std::complex<_T>::operator-=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
ComplexRef<_T>& ComplexRef<_T>::operator*=(const _Tref _Right){
	std::complex<_T>::operator*=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
ComplexRef<_T>& ComplexRef<_T>::operator/=(const _Tref _Right){
	std::complex<_T>::operator/=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator=(const std::complex<_Other>& _Right){
	std::complex<_T>::operator=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator+=(const std::complex<_Other>& _Right){
	std::complex<_T>::operator+=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator-=(const std::complex<_Other>& _Right){
	std::complex<_T>::operator-=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator*=(const std::complex<_Other>& _Right){
	std::complex<_T>::operator*=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator/=(const std::complex<_Other>& _Right){
	std::complex<_T>::operator/=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator=(const ComplexRef<_Other>& _Right){
	std::complex<_T>::operator=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator+=(const ComplexRef<_Other>& _Right){
	std::complex<_T>::operator+=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator-=(const ComplexRef<_Other>& _Right){
	std::complex<_T>::operator-=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator*=(const ComplexRef<_Other>& _Right){
	std::complex<_T>::operator*=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}

template<class _T>
template<class _Other> inline
ComplexRef<_T>& ComplexRef<_T>::operator/=(const ComplexRef<_Other>& _Right){
	std::complex<_T>::operator/=(_Right);
	this->realValue = std::complex<_T>::real();
	this->imagValue = std::complex<_T>::imag();
	return (*this);
}