#pragma once
#include <iostream>
#include <vector>
#include "thrust\device_vector.h"
//#include "HostVector.h"
using namespace std;
using namespace thrust;

template<class _T, class _Alloc>
class HostVector;

template<class _T, class _Alloc = device_malloc_allocator<_T>>
class DeviceVector :public device_vector<_T, _Alloc>{
public:
	DeviceVector() :device_vector(){}
	DeviceVector(const size_t &n) :device_vector(n){}
	DeviceVector(const size_t &n,const _T &value) :device_vector(n,value){}
	DeviceVector(const DeviceVector &v) :device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
		DeviceVector(const DeviceVector<_OtherT, _OtherAlloc> &v) : device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
		DeviceVector(const vector<_OtherT, _OtherAlloc> &v) : device_vector(v){}

	template<class _OtherT, class _OtherAlloc>
		DeviceVector(const HostVector<_OtherT, _OtherAlloc> &v) : device_vector(v){}

	template<class InputIterator>
		DeviceVector(InputIterator first, InputIterator last) : device_vector(first, last){}

	~DeviceVector(){}
};

