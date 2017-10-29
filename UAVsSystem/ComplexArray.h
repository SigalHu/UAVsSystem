#pragma once
#include <valarray>
#include <memory>
#include <complex>

template<class _T>
class ComplexRef;

template<class _T>
class ComplexArray {
	static_assert(std::is_arithmetic<_T>::value, "'_T' must be a arithmetic type.");
private:
	std::valarray<_T> realArray;
	std::valarray<_T> imagArray;
	std::unique_ptr<ComplexRef<_T>> upCurrentItem;
public:
	ComplexArray(const size_t &size);
	ComplexArray(const size_t &size, const _T &value);
	ComplexArray(const size_t &size, const _T &real, const _T &imag);

	ComplexRef<_T>& operator[](const size_t& _Off);

	void set(const size_t& index, const std::complex<_T>& value);
	std::complex<_T> get(const size_t& index) const;

	void setReal(const size_t& index, const _T &value);
	_T getReal(const size_t& index) const;

	void setImag(const size_t& index, const _T &value);
	_T getImag(const size_t& index) const;

	std::valarray<_T>& getRealArray() const;
	std::valarray<_T>& getImagArray() const;

	size_t size() const;
	bool checkSize() const;

	void resize(const size_t &size);
	void resize(const size_t &size, const _T &value);
	void resize(const size_t &size, const _T &real, const _T &imag);

	void fit2MaxSize();
	void fit2MinSize();
	void fit2RealSize();
	void fit2ImagSize();
};

