#pragma once
#include <vector>
#include "thrust\host_vector.h"
#include "DeviceVector.h"
using namespace std;
using namespace thrust;

template<class _T, class _Alloc = std::allocator<_T>>
class HostVector :public host_vector < _T, _Alloc > {
public:
	HostVector(){}
	HostVector(const size_t &n) :host_vector(n){}
	HostVector(const size_t &n, const _T &value) :host_vector(n, value){}
	HostVector(const HostVector &v) :host_vector(v){}
	HostVector(HostVector &&v) :host_vector(move(v)){}

	HostVector &operator=(const HostVector &v){
		host_vector::operator=(v);
		return *this;
	}

	HostVector &operator=(HostVector &&v){
		host_vector::operator=(move(v));
		return *this;
	}

	template<class _OtherT, class _OtherAlloc>
	HostVector(const HostVector<_OtherT, _OtherAlloc> &v) : host_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	HostVector &operator=(const HostVector<_OtherT, _OtherAlloc> &v){
		host_vector::operator=(v);
		return *this;
	}

	template<class _OtherT, class _OtherAlloc>
	HostVector(const vector<_OtherT, _OtherAlloc> &v) : host_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	HostVector &operator=(const vector<_OtherT, _OtherAlloc> &v){
		host_vector::operator=(v);
		return *this;
	}

	template<class _OtherT, class _OtherAlloc>
	HostVector(const DeviceVector<_OtherT, _OtherAlloc> &v) : host_vector(v){}

	template<class _OtherT, class _OtherAlloc>
	HostVector &operator=(const DeviceVector<_OtherT, _OtherAlloc> &v){
		host_vector::operator=(v);
		return *this;
	}

	template<class InputIterator>
	HostVector(InputIterator first, InputIterator last) : host_vector(first, last){}

	~HostVector(){}
};

