#pragma once
#include <vector>
#include "thrust\host_vector.h"
#include "DeviceVector.h"
using namespace std;
using namespace thrust;

template<class _T, class _Alloc = std::allocator<_T>>
class HostVector :public host_vector<_T, _Alloc>{
public:
	HostVector() :host_vector(){}
	HostVector(const size_t &n) :host_vector(n){}
	HostVector(const size_t &n, const _T &value) :host_vector(n, value){}
	HostVector(const HostVector &v) :host_vector(v){}

	template<class _OtherT, class _OtherAlloc>
		HostVector(const HostVector<_OtherT, _OtherAlloc> &v) : host_vector(v){}

	template<class _OtherT, class _OtherAlloc>
		HostVector(const vector<_OtherT, _OtherAlloc> &v) : host_vector(v){}

	template<class _OtherT, class _OtherAlloc>
		HostVector(const DeviceVector<_OtherT, _OtherAlloc> &v) : host_vector(v){}

	template<class InputIterator>
		HostVector(InputIterator first, InputIterator last) : host_vector(first, last){}

	~HostVector(){}
};

