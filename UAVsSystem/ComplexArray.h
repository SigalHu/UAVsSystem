#pragma once

#include <valarray>
#include <complex>
#include <memory>
#include "ComplexRef.h"
using namespace std;

template<class _T, size_t _Size>
class ComplexArray {
	static_assert(is_arithmetic<_T>::value, "'_T' must be a arithmetic type.");
private:
	valarray<_T> realArray;
	valarray<_T> imagArray;
	unique_ptr<ComplexRef<_T>> upCurrentItem;
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

	ComplexRef<_T>& operator[](const size_t& _Off) throw(out_of_range){
		if (_Off >= _Size)
			throw out_of_range();
		upCurrentItem.reset(new ComplexRef<_T>(this->realArray[_Off], this->imagArray[_Off]));
		return (*upCurrentItem);
	}

	void set(const size_t& index, const complex<_T>& value) throw(out_of_range){
		if (index >= _Size)
			throw out_of_range();
		this->realArray[index] = value.real();
		this->imagArray[index] = value.imag();
	}

	complex<_T> get(const size_t& index) const throw(out_of_range){
		if (index >= _Size)
			throw out_of_range();
		return complex<_T>(this->realArray[index], this->imagArray[index]);
	}

	void setReal(const size_t& index, const _T &value) throw(out_of_range){
		if (index >= _Size)
			throw out_of_range();
		this->realArray[index] = value;
	}

	_T getReal(const size_t& index) const throw(out_of_range){
		if (index >= _Size)
			throw out_of_range();
		return this->realArray[index];
	}

	void setImag(const size_t& index, const _T &value) throw(out_of_range){
		if (index >= _Size)
			throw out_of_range();
		this->imagArray[index] = value;
	}

	_T getImag(const size_t& index) const throw(out_of_range){
		if (index >= _Size)
			throw out_of_range();
		return this->imagArray[index];
	}

	_T* getRealPtr() const{
		return &(this->realArray[0]);
	}

	_T* getImagPtr() const{
		return &(this->imagArray[0]);
	}

	valarray<_T>& getRealArray() {
		return (this->realArray);
	}

	valarray<_T>& getImagArray() {
		return (this->imagArray);
	}

	size_t size() const{
		return _Size;
	}
};

