#pragma once
#include <memory>
#include <complex>

template<class _OtherT, class _OtherAlloc>
class HostVector;
template<class _T>
class ComplexRef;

template<class _T, class _Alloc = std::allocator<_T>>
class HostComplexArray {
	static_assert(std::is_arithmetic<_T>::value, "'_T' must be a arithmetic type.");
private:
	HostVector<_T, _Alloc> realVector;
	HostVector<_T, _Alloc> imagVector;
	std::unique_ptr<ComplexRef<_T>> upCurrentItem;
public:
	HostComplexArray(const size_t &size);
	HostComplexArray(const size_t &size, const _T &value);
	HostComplexArray(const size_t &size, const _T &real, const _T &imag);

	ComplexRef<_T>& operator[](const size_t& _Off);

	void set(const size_t& index, const std::complex<_T>& value);
	std::complex<_T> get(const size_t& index) const;

	void setReal(const size_t& index, const _T &value);
	_T getReal(const size_t& index) const;

	void setImag(const size_t& index, const _T &value);
	_T getImag(const size_t& index) const;

	HostVector<_T, _Alloc>& getRealArray() const;
	HostVector<_T, _Alloc>& getImagArray() const;

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