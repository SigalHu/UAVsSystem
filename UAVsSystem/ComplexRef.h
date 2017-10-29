#pragma once
#include <complex>

template<class _T>
class ComplexRef final:public std::complex<_T>{
	template<class _T1>
	friend class ComplexArray;

	template<class _T, class _Alloc>
	friend class HostComplexArray;

	typedef typename std::remove_reference<_T>::type& _Tref;
private:
	_Tref realValue;
	_Tref imagValue;
private:
	ComplexRef() = default;
	ComplexRef(const ComplexRef& item) = default;
	ComplexRef(_Tref realValue, _Tref imagValue);
public:
	_T real();
	_T imag();

	ComplexRef<_T>& operator=(const _Tref _Right);
	ComplexRef<_T>& operator+=(const _Tref _Right);
	ComplexRef<_T>& operator-=(const _Tref _Right);
	ComplexRef<_T>& operator*=(const _Tref _Right);
	ComplexRef<_T>& operator/=(const _Tref _Right);

	template<class _Other> inline
		ComplexRef<_T>& operator=(const std::complex<_Other>& _Right);
	template<class _Other> inline
		ComplexRef<_T>& operator+=(const std::complex<_Other>& _Right);
	template<class _Other> inline
		ComplexRef<_T>& operator-=(const std::complex<_Other>& _Right);
	template<class _Other> inline
		ComplexRef<_T>& operator*=(const std::complex<_Other>& _Right);
	template<class _Other> inline
		ComplexRef<_T>& operator/=(const std::complex<_Other>& _Right);

	template<class _Other> inline
		ComplexRef<_T>& operator=(const ComplexRef<_Other>& _Right);
	template<class _Other> inline
		ComplexRef<_T>& operator+=(const ComplexRef<_Other>& _Right);
	template<class _Other> inline
		ComplexRef<_T>& operator-=(const ComplexRef<_Other>& _Right);
	template<class _Other> inline
		ComplexRef<_T>& operator*=(const ComplexRef<_Other>& _Right);
	template<class _Other> inline
		ComplexRef<_T>& operator/=(const ComplexRef<_Other>& _Right);
};

