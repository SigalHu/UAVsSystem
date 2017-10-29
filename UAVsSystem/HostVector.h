#pragma once
#include <vector>
#include "thrust\host_vector.h"

template<class _T, class _Alloc>
class DeviceVector;

template<class _T, class _Alloc = std::allocator<_T>>
class HostVector :public thrust::host_vector < _T, _Alloc > {
public:
	HostVector();
	HostVector(const size_t &n);
	HostVector(const size_t &n, const _T &value);
	HostVector(const HostVector &v);
	HostVector(HostVector &&v);

	HostVector& operator=(const HostVector &v);
	HostVector& operator=(HostVector &&v);

	template<class _OtherT, class _OtherAlloc>
	HostVector(const HostVector<_OtherT, _OtherAlloc> &v);

	template<class _OtherT, class _OtherAlloc>
	HostVector& operator=(const HostVector<_OtherT, _OtherAlloc> &v);

	template<class _OtherT, class _OtherAlloc>
	HostVector(const std::vector<_OtherT, _OtherAlloc> &v);

	template<class _OtherT, class _OtherAlloc>
	HostVector &operator=(const std::vector<_OtherT, _OtherAlloc> &v);

	template<class _OtherT, class _OtherAlloc>
	HostVector(const DeviceVector<_OtherT, _OtherAlloc> &v);

	template<class _OtherT, class _OtherAlloc>
	HostVector &operator=(const DeviceVector<_OtherT, _OtherAlloc> &v);

	template<class InputIterator>
	HostVector(InputIterator first, InputIterator last);
};

