#pragma once

#include <complex>
using namespace std;

template<class _T>
class ComplexRef final:public complex<_T>{
	template<class _T1>
	friend class ComplexArray;

	template<class _T, class _Alloc = std::allocator<_T>>
	friend class HostComplexArray;

	typedef typename remove_reference<_T>::type& _Tref;
private:
	_Tref realValue;
	_Tref imagValue;
private:
	ComplexRef() = default;
	ComplexRef(const ComplexRef& item) = default;
	ComplexRef(_Tref realValue, _Tref imagValue) :complex<_T>(realValue, imagValue),realValue(realValue), imagValue(imagValue){}
public:
	_T real(){
		return (this->realValue);
	}

	_T imag(){
		return (this->imagValue);
	}

	ComplexRef<_T>& operator=(const _Tref _Right){
		complex<_T>::operator=(_Right);
		this->realValue = complex<_T>::real();
		this->imagValue = complex<_T>::imag();
		return (*this);
	}

	ComplexRef<_T>& operator+=(const _Tref _Right){
		complex<_T>::operator+=(_Right);
		this->realValue = complex<_T>::real();
		this->imagValue = complex<_T>::imag();
		return (*this);
	}

	ComplexRef<_T>& operator-=(const _Tref _Right){
		complex<_T>::operator-=(_Right);
		this->realValue = complex<_T>::real();
		this->imagValue = complex<_T>::imag();
		return (*this);
	}

	ComplexRef<_T>& operator*=(const _Tref _Right){
		complex<_T>::operator*=(_Right);
		this->realValue = complex<_T>::real();
		this->imagValue = complex<_T>::imag();
		return (*this);
	}

	ComplexRef<_T>& operator/=(const _Tref _Right){
		complex<_T>::operator/=(_Right);
		this->realValue = complex<_T>::real();
		this->imagValue = complex<_T>::imag();
		return (*this);
	}

	template<class _Other> inline
		ComplexRef<_T>& operator=(const complex<_Other>& _Right){
			complex<_T>::operator=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator+=(const complex<_Other>& _Right){
			complex<_T>::operator+=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator-=(const complex<_Other>& _Right){
			complex<_T>::operator-=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator*=(const complex<_Other>& _Right){
			complex<_T>::operator*=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator/=(const complex<_Other>& _Right){
			complex<_T>::operator/=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator=(const ComplexRef<_Other>& _Right){
			complex<_T>::operator=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator+=(const ComplexRef<_Other>& _Right){
			complex<_T>::operator+=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator-=(const ComplexRef<_Other>& _Right){
			complex<_T>::operator-=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator*=(const ComplexRef<_Other>& _Right){
			complex<_T>::operator*=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator/=(const ComplexRef<_Other>& _Right){
			complex<_T>::operator/=(_Right);
			this->realValue = complex<_T>::real();
			this->imagValue = complex<_T>::imag();
			return (*this);
		}
};

