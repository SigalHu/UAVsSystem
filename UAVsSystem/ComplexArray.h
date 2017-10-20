#pragma once

#include <valarray>
#include <complex>
#include <array>
using namespace std;

template<class _T, size_t _Size>
class ComplexArray {
private:
	class Complex :public complex<_T>{
	private:
		_T& real;
		_T& imag;
	private:
		Complex() = default;
	public:
		Complex(_T& real, _T& imag) :complex<_T>(real,imag),real(real), imag(imag){}

		Complex& operator=(const _T& _Right){
			complex<_T>::operator=(_Right);
			this->real = complex<_T>::real();
			this->imag = complex<_T>::imag();
			return (*this);
		}

		Complex& operator+=(const _T& _Right){
			complex<_T>::operator+=(_Right);
			this->real = complex<_T>::real();
			this->imag = complex<_T>::imag();
			return (*this);
		}

		Complex& operator-=(const _T& _Right){
			complex<_T>::operator-=(_Right);
			this->real = complex<_T>::real();
			this->imag = complex<_T>::imag();
			return (*this);
		}

		Complex& operator*=(const _T& _Right){
			complex<_T>::operator*=(_Right);
			this->real = complex<_T>::real();
			this->imag = complex<_T>::imag();
			return (*this);
		}

		Complex& operator/=(const _T& _Right){
			complex<_T>::operator/=(_Right);
			this->real = complex<_T>::real();
			this->imag = complex<_T>::imag();
			return (*this);
		}

		Complex& operator+=(const Complex& _Right){
			complex<_T>::operator+=(_Right);
			this->real = complex<_T>::real();
			this->imag = complex<_T>::imag();
			return (*this);
		}

		Complex& operator-=(const Complex& _Right){
			complex<_T>::operator-=(_Right);
			this->real = complex<_T>::real();
			this->imag = complex<_T>::imag();
			return (*this);
		}

		Complex& operator*=(const Complex& _Right){
			complex<_T>::operator*=(_Right);
			this->real = complex<_T>::real();
			this->imag = complex<_T>::imag();
			return (*this);
		}

		Complex& operator/=(const Complex& _Right){
			complex<_T>::operator/=(_Right);
			this->real = complex<_T>::real();
			this->imag = complex<_T>::imag();
			return (*this);
		}

		template<class _Other> inline
			Complex& operator=(const complex<_Other>& _Right){
				complex<_T>::operator=(_Right);
				this->real = complex<_T>::real();
				this->imag = complex<_T>::imag();
				return (*this);
			}

		template<class _Other> inline
			Complex& operator+=(const complex<_Other>& _Right){
				complex<_T>::operator+=(_Right);
				this->real = complex<_T>::real();
				this->imag = complex<_T>::imag();
				return (*this);
			}

		template<class _Other> inline
			Complex& operator-=(const complex<_Other>& _Right){
				complex<_T>::operator-=(_Right);
				this->real = complex<_T>::real();
				this->imag = complex<_T>::imag();
				return (*this);
			}

		template<class _Other> inline
			Complex& operator*=(const complex<_Other>& _Right){
				complex<_T>::operator*=(_Right);
				this->real = complex<_T>::real();
				this->imag = complex<_T>::imag();
				return (*this);
			}

		template<class _Other> inline
			Complex& operator/=(const complex<_Other>& _Right){
				complex<_T>::operator/=(_Right);
				this->real = complex<_T>::real();
				this->imag = complex<_T>::imag();
				return (*this);
			}
	};
private:
	valarray<_T> realArray;
	valarray<_T> imagArray;
public:
	ComplexArray(){
		this->realArray.resize(_Size,0);
		this->imagArray.resize(_Size,0);
	}

	ComplexArray(const _T &value){
		this->realArray.resize(_Size,value);
		this->imagArray.resize(_Size,value);
	}

	~ComplexArray(){
	}

	void set(size_t index, const complex<_T> &value) throw(out_of_range){
		if (index >= _Size)
			throw out_of_range("越界访问！");
		this->realArray[index] = value.real();
		this->imagArray[index] = value.imag();
	}

	complex<_T> get(size_t index) throw(out_of_range){
		if (index >= _Size)
			throw out_of_range("越界访问！");
		return complex<_T>(this->realArray[index], this->imagArray[index]);
	}

	void setReal(size_t index, const _T &value) throw(out_of_range){
		if (index >= _Size)
			throw out_of_range("越界访问！");
		this->realArray[index] = value;
	}

	_T getReal(size_t index) throw(out_of_range){
		if (index >= _Size)
			throw out_of_range("越界访问！");
		return this->realArray[index];
	}

	void setImag(size_t index, const _T &value) throw(out_of_range){
		if (index >= _Size)
			throw out_of_range("越界访问！");
		this->imagArray[index] = value;
	}

	_T getImag(size_t index) throw(out_of_range){
		if (index >= _Size)
			throw out_of_range("越界访问！");
		return this->imagArray[index];
	}

	_T* getRealPtr(){
		return &(this->realArray[0]);
	}

	_T* getImagPtr(){
		return &(this->imagArray[0]);
	}

	size_t size() const{
		return _Size;
	}
};

