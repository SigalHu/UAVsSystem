#pragma once
#include <complex>

template<class _T>
class ComplexRef final:public std::complex<_T>{
	template<class _T1>
	friend class ComplexArray;

	template<class _T, class _Alloc = std::allocator<_T>>
	friend class HostComplexArray;

	typedef typename std::remove_reference<_T>::type& _Tref;
private:
	_Tref realValue;
	_Tref imagValue;
private:
	ComplexRef() = default;
	ComplexRef(const ComplexRef& item) = default;
	ComplexRef(_Tref realValue, _Tref imagValue) :std::complex<_T>(realValue, imagValue),realValue(realValue), imagValue(imagValue){}
public:
	_T real(){
		return (this->realValue);
	}

	_T imag(){
		return (this->imagValue);
	}

	ComplexRef<_T>& operator=(const _Tref _Right){
		std::complex<_T>::operator=(_Right);
		this->realValue = std::complex<_T>::real();
		this->imagValue = std::complex<_T>::imag();
		return (*this);
	}

	ComplexRef<_T>& operator+=(const _Tref _Right){
		std::complex<_T>::operator+=(_Right);
		this->realValue = std::complex<_T>::real();
		this->imagValue = std::complex<_T>::imag();
		return (*this);
	}

	ComplexRef<_T>& operator-=(const _Tref _Right){
		std::complex<_T>::operator-=(_Right);
		this->realValue = std::complex<_T>::real();
		this->imagValue = std::complex<_T>::imag();
		return (*this);
	}

	ComplexRef<_T>& operator*=(const _Tref _Right){
		std::complex<_T>::operator*=(_Right);
		this->realValue = std::complex<_T>::real();
		this->imagValue = std::complex<_T>::imag();
		return (*this);
	}

	ComplexRef<_T>& operator/=(const _Tref _Right){
		std::complex<_T>::operator/=(_Right);
		this->realValue = std::complex<_T>::real();
		this->imagValue = std::complex<_T>::imag();
		return (*this);
	}

	template<class _Other> inline
		ComplexRef<_T>& operator=(const std::complex<_Other>& _Right){
			std::complex<_T>::operator=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator+=(const std::complex<_Other>& _Right){
			std::complex<_T>::operator+=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator-=(const std::complex<_Other>& _Right){
			std::complex<_T>::operator-=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator*=(const std::complex<_Other>& _Right){
			std::complex<_T>::operator*=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator/=(const std::complex<_Other>& _Right){
			std::complex<_T>::operator/=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator=(const ComplexRef<_Other>& _Right){
			std::complex<_T>::operator=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator+=(const ComplexRef<_Other>& _Right){
			std::complex<_T>::operator+=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator-=(const ComplexRef<_Other>& _Right){
			std::complex<_T>::operator-=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator*=(const ComplexRef<_Other>& _Right){
			std::complex<_T>::operator*=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}

	template<class _Other> inline
		ComplexRef<_T>& operator/=(const ComplexRef<_Other>& _Right){
			std::complex<_T>::operator/=(_Right);
			this->realValue = std::complex<_T>::real();
			this->imagValue = std::complex<_T>::imag();
			return (*this);
		}
};

